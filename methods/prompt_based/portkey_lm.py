"""Portkey adapters used by both GEPA and ACE/OE paths.

Two surfaces are exposed:

  - PortkeyChatLM   — DSPy LM subclass used by GEPA (litellm bypass).
  - PortkeyOpenAIShim — duck-typed openai.OpenAI() client used by the
    harness (`_call_llm` in openevolve_runner.py) and ACE's internal
    generator/reflector/curator (`ace/llm.py:timed_llm_call`).

Mirrors methods/prompt_based/sagemaker_lm.py — same shim shape so
the wiring (setup_run, _build_ace_instance, initialize_clients) is
symmetric. Both bypass litellm to avoid provider-routing quirks; here
the route IS the model string (e.g. "@fireworks/accounts/fireworks/
models/qwen3-8b") which Portkey interprets directly.

Why use the Portkey SDK directly instead of OpenAI SDK + base_url?
The Portkey SDK accepts `strict_open_ai_compliance=False`, which lets
the gateway pass responses through without coercing them into strict
OpenAI shape — needed for some providers (e.g. Fireworks Qwen3) whose
response payloads carry extra fields that break stricter parsers.

Auth: PORTKEY_API_KEY env var (loaded via python-dotenv in scripts/run.py).
"""

from __future__ import annotations

import json
import os
import re
import time

import dspy
from portkey_ai import Portkey


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# Fireworks-via-Portkey rejects non-streaming requests with max_tokens > 4096
# ("Requests with max_tokens > 4096 must have stream=true"). The harness
# doesn't stream, so we clamp at the API ceiling for that provider only.
_PORTKEY_FIREWORKS_MAX_TOKENS_CAP = 4096
# Non-Fireworks upstreams (OpenAI, Google, ...) don't share the 4096
# non-streaming limit. Reasoning models like GPT-5.2 also need far more
# headroom because reasoning tokens count against max_tokens, so a tiny cap
# would starve the visible completion. Clamp generously instead.
_PORTKEY_DEFAULT_MAX_TOKENS_CAP = 32768


def _max_tokens_cap(route: str) -> int:
    """Per-provider non-streaming max_tokens ceiling, keyed off the route.

    Only the Fireworks route carries the hard 4096 non-streaming limit; every
    other upstream gets the generous default. `route` is the Portkey routing
    string, e.g. "@fireworks/accounts/fireworks/models/qwen3-8b" or
    "@openai/gpt-5.2".
    """
    if "fireworks" in (route or "").lower():
        return _PORTKEY_FIREWORKS_MAX_TOKENS_CAP
    return _PORTKEY_DEFAULT_MAX_TOKENS_CAP


def _build_portkey_client():
    """Construct a Portkey client; PORTKEY_API_KEY is required."""
    api_key = os.environ.get("PORTKEY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "PORTKEY_API_KEY env var not set. Export it or add to .env."
        )
    return Portkey(api_key=api_key, strict_open_ai_compliance=False)


def _inject_no_think(messages):
    """Inject `/no_think` directives so Qwen3 skips reasoning mode.

    Qwen3's `/no_think` is a *soft* directive — long, instruction-heavy
    contexts (e.g. an ACE playbook prepended to a task prompt) can override a
    single injection and cause the model to emit thinking-only output, which
    `_ShapedResponse` then strips to "". To make the signal robust we inject
    in two places:
      1. The system message (prepended to it, or a new one created): the
         model sees this first as a meta-directive.
      2. The last user message: legacy injection point, kept as belt-and-
         braces in case a provider's chat template ignores the system slot.
    Both injections are idempotent — repeated calls don't duplicate the marker.
    """
    msgs = [dict(m) for m in messages]
    # System slot: prepend /no_think into the first system message; create
    # one at index 0 if none exists.
    has_system = False
    for i, m in enumerate(msgs):
        if m.get("role") == "system":
            content = m.get("content", "") or ""
            if "/no_think" not in content:
                msgs[i]["content"] = (
                    f"/no_think\n\n{content}" if content else "/no_think"
                )
            has_system = True
            break
    if not has_system:
        msgs.insert(0, {"role": "system", "content": "/no_think"})
    # Last user slot: append /no_think to the most recent user message.
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].get("role") == "user":
            content = msgs[i].get("content", "") or ""
            if "/no_think" not in content:
                msgs[i]["content"] = content.rstrip() + "\n\n/no_think"
            break
    return msgs


# OpenAI's GPT-5 / o-series reasoning models reject the legacy `max_tokens`
# field (they require `max_completion_tokens`) and only accept temperature == 1
# (any explicit other value 400s). Detect them off the route so we send a
# compliant payload; everything else keeps the classic chat-completions shape.
_OPENAI_REASONING_RE = re.compile(r"(^|[/@])(gpt-5|o[1-4])", re.IGNORECASE)


def _is_openai_reasoning_route(route: str) -> bool:
    r = (route or "").lower()
    return "openai" in r and bool(_OPENAI_REASONING_RE.search(r))


def _call_with_retry(client, *, model, messages, temperature, max_tokens,
                     num_retries=3):
    kwargs = {"model": model, "messages": messages}
    if _is_openai_reasoning_route(model):
        # GPT-5 / o-series: max_completion_tokens, and omit temperature so the
        # API uses its fixed default of 1 instead of 400ing on an explicit value.
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens
        kwargs["temperature"] = temperature
    last_err = None
    for attempt in range(num_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            last_err = e
            if attempt == num_retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise last_err  # pragma: no cover


# ---------------------------------------------------------------------------
#  Duck-typed OpenAI client for ACE / openevolve_runner._call_llm
# ---------------------------------------------------------------------------

class _Msg:
    __slots__ = ("content", "role", "raw_content")
    def __init__(self, content: str, role: str = "assistant",
                 raw_content: str | None = None):
        self.content = content
        self.role = role
        # Pre-<think>-strip content, preserved for diagnostics. None when no
        # stripping occurred. A consumer that sees content == "" can check
        # raw_content to distinguish "model returned nothing" from
        # "model returned thinking-only output that we stripped to empty".
        self.raw_content = raw_content


class _Choice:
    __slots__ = ("message", "finish_reason", "index")
    def __init__(self, message: _Msg, finish_reason: str = "stop", index: int = 0):
        self.message = message
        self.finish_reason = finish_reason
        self.index = index


class _Usage:
    __slots__ = ("prompt_tokens", "completion_tokens", "total_tokens")
    def __init__(self, raw):
        # Portkey SDK returns either a usage object or a dict — handle both.
        def _get(o, k):
            if o is None:
                return 0
            if isinstance(o, dict):
                return int(o.get(k, 0) or 0)
            return int(getattr(o, k, 0) or 0)
        self.prompt_tokens = _get(raw, "prompt_tokens")
        self.completion_tokens = _get(raw, "completion_tokens")
        self.total_tokens = _get(raw, "total_tokens")


class _ShapedResponse:
    """Wraps a Portkey response so .choices[0].message.content / .usage.* work
    consistently regardless of whether Portkey returns dict or object shapes."""
    def __init__(self, raw):
        choices_raw = getattr(raw, "choices", None)
        if choices_raw is None and isinstance(raw, dict):
            choices_raw = raw.get("choices") or []
        choices_raw = choices_raw or []
        self.choices = []
        for i, c in enumerate(choices_raw):
            msg = getattr(c, "message", None)
            if msg is None and isinstance(c, dict):
                msg = c.get("message") or {}
            raw_content = (
                getattr(msg, "content", None)
                if not isinstance(msg, dict)
                else msg.get("content")
            ) or ""
            content = _THINK_RE.sub("", raw_content).strip()
            role = (
                getattr(msg, "role", "assistant")
                if not isinstance(msg, dict)
                else msg.get("role", "assistant")
            )
            finish = (
                getattr(c, "finish_reason", "stop")
                if not isinstance(c, dict)
                else c.get("finish_reason", "stop")
            )
            # Diagnostic: when PORTKEY_LOG_RAW_PATH is set, append a one-line
            # record per response with pre-strip vs post-strip metadata so we
            # can prove whether Qwen3 is silently emitting thinking-only
            # output. Off by default (zero cost when the env var is unset).
            _log_path = os.environ.get("PORTKEY_LOG_RAW_PATH")
            if _log_path:
                try:
                    with open(_log_path, "a") as _f:
                        _f.write(json.dumps({
                            "raw_chars": len(raw_content),
                            "stripped_chars": len(content),
                            "stripped_empty": content == "",
                            "all_thinking": (
                                content == "" and "<think>" in raw_content
                            ),
                            "finish_reason": finish,
                            "raw_preview": raw_content[:600],
                        }) + "\n")
                except OSError:
                    pass  # never let logging break a run
            # Only stash raw_content when it differs from stripped — keeps
            # memory low for the common "no thinking" case.
            self.choices.append(_Choice(
                _Msg(content, role=role,
                     raw_content=raw_content if raw_content != content else None),
                finish_reason=finish, index=i,
            ))
        self.usage = _Usage(
            getattr(raw, "usage", None)
            if not isinstance(raw, dict)
            else raw.get("usage")
        )
        self.model = (
            getattr(raw, "model", "")
            if not isinstance(raw, dict)
            else raw.get("model", "")
        )
        self.id = (
            getattr(raw, "id", "")
            if not isinstance(raw, dict)
            else raw.get("id", "")
        )


class _PortkeyCompletions:
    """Implements `client.chat.completions.create(...)`."""

    def __init__(self, route: str, client, no_think: bool = False,
                 num_retries: int = 3):
        self._route = route
        self._client = client
        self._no_think = no_think
        self._num_retries = num_retries

    def create(self, *, messages, model=None, temperature=0.0, max_tokens=8192,
               extra_body=None, response_format=None,
               max_completion_tokens=None, **_kwargs):
        # `model`: caller may pass our "portkey:<route>" string OR the bare
        # route. Either way, we send the configured Portkey route. The
        # portkey: prefix is harness convention; Portkey expects "@<provider>/...".
        # `extra_body`, `response_format`: Portkey SDK accepts both verbatim
        # but they're route-specific; drop to keep the path provider-neutral.
        # `max_completion_tokens`: OpenAI's newer name — coerce.
        msgs = list(messages)
        if self._no_think:
            msgs = _inject_no_think(msgs)
        out_max = max_completion_tokens if max_completion_tokens is not None else max_tokens
        cap = _max_tokens_cap(self._route)
        out_max = min(int(out_max) if out_max else cap, cap)
        raw = _call_with_retry(
            self._client, model=self._route, messages=msgs,
            temperature=float(temperature) if temperature is not None else 0.0,
            max_tokens=out_max,
            num_retries=self._num_retries,
        )
        return _ShapedResponse(raw)


class _Chat:
    def __init__(self, completions):
        self.completions = completions


class PortkeyOpenAIShim:
    """Duck-types `openai.OpenAI(...)` for the harness + ACE call sites.

    Used in two places:
      1. openevolve_runner.setup_run — replaces the `OpenAI(...)` client
         that `_call_llm` calls into.
      2. ace/utils.py:initialize_clients (portkey branch) — returns three
         instances (generator/reflector/curator).
    """

    def __init__(self, route: str, no_think: bool = False, num_retries: int = 3):
        client = _build_portkey_client()
        self._route = route
        self._client = client
        self.chat = _Chat(_PortkeyCompletions(
            route=route, client=client,
            no_think=no_think, num_retries=num_retries,
        ))


# ---------------------------------------------------------------------------
#  DSPy LM for GEPA
# ---------------------------------------------------------------------------

class PortkeyChatLM(dspy.LM):
    """DSPy LM backed by the Portkey SDK (bypasses litellm).

    `route` is the Portkey model-routing string, e.g.
    "@fireworks/accounts/fireworks/models/qwen3-8b".
    """

    def __init__(
        self,
        route: str,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        no_think: bool = True,
        num_retries: int = 3,
        **kwargs,
    ):
        kwargs.pop("api_key", None)
        kwargs.pop("api_base", None)
        kwargs.pop("extra_body", None)
        super().__init__(
            model=f"portkey:{route}",
            model_type="chat",
            temperature=temperature,
            max_tokens=max_tokens,
            cache=False,
            **kwargs,
        )
        self._client = _build_portkey_client()
        self._route = route
        self._no_think = no_think
        self._num_retries = num_retries

    def __call__(self, prompt=None, messages=None, **kwargs):
        if messages is None:
            messages = [{"role": "user", "content": prompt or ""}]
        if self._no_think:
            messages = _inject_no_think(messages)

        cap = _max_tokens_cap(self._route)
        max_tokens = kwargs.get("max_tokens", self.kwargs.get("max_tokens", cap))
        temperature = kwargs.get("temperature", self.kwargs.get("temperature", 0.0))
        max_tokens = min(int(max_tokens) if max_tokens else cap, cap)
        raw = _call_with_retry(
            self._client, model=self._route, messages=messages,
            temperature=float(temperature) if temperature is not None else 0.0,
            max_tokens=max_tokens,
            num_retries=self._num_retries,
        )
        # Pull text via the shaped wrapper to share the <think> stripper.
        shaped = _ShapedResponse(raw)
        text = shaped.choices[0].message.content if shaped.choices else ""

        self.history.append({
            "prompt": prompt,
            "messages": messages,
            "kwargs": {"model": self._route, "temperature": temperature, "max_tokens": max_tokens},
            "response": raw if isinstance(raw, dict) else None,
            "outputs": [text],
        })
        return [text]
