"""AWS Bedrock `Converse` adapters used by both GEPA and ACE/OE paths.

Two surfaces are exposed:

  - BedrockChatLM   — DSPy LM subclass used by GEPA (litellm bypass).
  - BedrockOpenAIShim — duck-typed openai.OpenAI() client used by the
    harness (`_call_llm` in openevolve_runner.py) and ACE's internal
    generator/reflector/curator (`ace/llm.py:timed_llm_call`).

Mirrors methods/prompt_based/sagemaker_lm.py — same shim shape so
the wiring (setup_run, _build_ace_instance, initialize_clients) is symmetric.
Both bypass litellm to avoid provider-routing quirks; here the route IS the
Bedrock modelId (e.g. "qwen.qwen3-32b-v1:0").

Why a dedicated shim instead of going through litellm/Portkey/Bedrock? The
Bedrock `Converse` API has its own message envelope:

    system   = [{"text": "..."}]
    messages = [{"role": "user|assistant", "content": [{"text": "..."}]}]

i.e. a separate `system` array plus content-as-list-of-blocks. That doesn't
match the OpenAI chat-completions shape the rest of the harness speaks, so we
translate on the way in and translate the response on the way out.

Auth: AWS_BEARER_TOKEN_BEDROCK env var. boto3's bedrock-runtime client picks
it up automatically and switches to bearer auth (bypassing SigV4 / IAM); no
AWS_PROFILE / IAM permission grant required. Region defaults to us-east-1
and can be overridden via the `aws_region_name` config field.
"""

from __future__ import annotations

import os
import re
import time

import boto3
import dspy
from botocore.config import Config as BotoConfig


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _build_bedrock_client(region_name: str, read_timeout: int = 300,
                          connect_timeout: int = 10):
    """Construct a bedrock-runtime client with hard timeouts.

    `retries={"max_attempts": 1}` disables botocore's auto-retry layer so
    callers see ReadTimeoutError directly. BedrockChatLM and BedrockOpenAIShim
    do their own bounded retry loops with backoff.
    """
    cfg = BotoConfig(
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
        retries={"max_attempts": 1, "mode": "standard"},
    )
    return boto3.client("bedrock-runtime", region_name=region_name, config=cfg)


def _inject_no_think(messages):
    """Mirror of portkey_lm._inject_no_think.

    Qwen3's `/no_think` is *soft*; we inject in two places (system + last
    user message) so dense playbook prompts don't silently push the model
    into reasoning mode and produce thinking-only output.
    """
    msgs = [dict(m) for m in messages]
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
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].get("role") == "user":
            content = msgs[i].get("content", "") or ""
            if "/no_think" not in content:
                msgs[i]["content"] = content.rstrip() + "\n\n/no_think"
            break
    return msgs


def _split_openai_messages(messages):
    """Translate OpenAI chat messages into Bedrock Converse shape.

    Bedrock wants `system` as a separate array of `{"text": str}` blocks
    (concatenated if multiple system messages were supplied) and `messages`
    with only user/assistant roles, each `content` wrapped as `[{"text": str}]`.
    Consecutive same-role messages are merged so Bedrock's strict alternation
    rule doesn't reject the payload — uncommon in our usage but cheap to guard.
    """
    system_blocks = []
    out_msgs = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "") or ""
        if not isinstance(content, str):
            # Defensive: stringify non-text content rather than crashing.
            content = str(content)
        if role == "system":
            if content:
                system_blocks.append({"text": content})
            continue
        if role not in ("user", "assistant"):
            continue
        if out_msgs and out_msgs[-1]["role"] == role:
            out_msgs[-1]["content"][0]["text"] += "\n\n" + content
        else:
            out_msgs.append({"role": role, "content": [{"text": content}]})
    # Bedrock requires the first message to be from `user`. If somehow we
    # have nothing or start with assistant, inject a placeholder user turn.
    if not out_msgs or out_msgs[0]["role"] != "user":
        out_msgs.insert(0, {"role": "user", "content": [{"text": ""}]})
    return system_blocks, out_msgs


def _extract_text(bedrock_response):
    """Pull the assistant text out of a Converse response."""
    msg = bedrock_response.get("output", {}).get("message", {})
    parts = []
    for block in msg.get("content", []):
        if "text" in block:
            parts.append(block["text"])
    return "".join(parts)


_FINISH_REASON_MAP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "content_filtered": "content_filter",
    "guardrail_intervened": "content_filter",
}


def _converse_with_retry(client, *, model_id, messages, temperature,
                         max_tokens, num_retries=3):
    """Translate + call bedrock.converse with bounded retries."""
    system_blocks, msgs = _split_openai_messages(messages)
    kwargs = {
        "modelId": model_id,
        "messages": msgs,
        "inferenceConfig": {
            "maxTokens": int(max_tokens),
            "temperature": float(temperature) if temperature is not None else 0.0,
        },
    }
    if system_blocks:
        kwargs["system"] = system_blocks
    last_err = None
    for attempt in range(num_retries):
        try:
            return client.converse(**kwargs)
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
        # Pre-<think>-strip text, for diagnostics. None when no stripping
        # was needed. Mirrors portkey_lm._Msg.raw_content.
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
        usage = raw.get("usage", {}) if isinstance(raw, dict) else {}
        self.prompt_tokens = int(usage.get("inputTokens", 0) or 0)
        self.completion_tokens = int(usage.get("outputTokens", 0) or 0)
        self.total_tokens = int(usage.get("totalTokens", 0) or 0)


class _ShapedResponse:
    """Wraps a Bedrock Converse response so .choices[0].message.content /
    .usage.* work like a stock OpenAI ChatCompletion."""
    def __init__(self, raw):
        raw_text = _extract_text(raw)
        content = _THINK_RE.sub("", raw_text).strip()
        bedrock_reason = raw.get("stopReason", "end_turn")
        finish = _FINISH_REASON_MAP.get(bedrock_reason, "stop")
        # Optional raw-content diagnostic, mirrors portkey_lm.
        import json as _json
        _log_path = os.environ.get("BEDROCK_LOG_RAW_PATH")
        if _log_path:
            try:
                with open(_log_path, "a") as _f:
                    _f.write(_json.dumps({
                        "raw_chars": len(raw_text),
                        "stripped_chars": len(content),
                        "stripped_empty": content == "",
                        "all_thinking": (
                            content == "" and "<think>" in raw_text
                        ),
                        "stop_reason": bedrock_reason,
                        "raw_preview": raw_text[:600],
                    }) + "\n")
            except OSError:
                pass
        self.choices = [_Choice(
            _Msg(content, role="assistant",
                 raw_content=raw_text if raw_text != content else None),
            finish_reason=finish,
            index=0,
        )]
        self.usage = _Usage(raw)
        self.model = raw.get("ResponseMetadata", {}).get("RequestId", "")
        self.id = self.model


class _BedrockCompletions:
    """Implements `client.chat.completions.create(...)`."""

    def __init__(self, model_id: str, client, no_think: bool = False,
                 num_retries: int = 3):
        self._model_id = model_id
        self._client = client
        self._no_think = no_think
        self._num_retries = num_retries

    def create(self, *, messages, model=None, temperature=0.0, max_tokens=8192,
               extra_body=None, response_format=None,
               max_completion_tokens=None, **_kwargs):
        # `model`: caller may pass the bare model_id or our "bedrock:<id>"
        # convention; we always send the configured modelId. `extra_body`
        # and `response_format` are dropped — Bedrock's Converse API has its
        # own inferenceConfig/additionalModelRequestFields surface; keep
        # this shim provider-neutral for the harness's use cases.
        msgs = list(messages)
        if self._no_think:
            msgs = _inject_no_think(msgs)
        out_max = max_completion_tokens if max_completion_tokens is not None else max_tokens
        raw = _converse_with_retry(
            self._client, model_id=self._model_id, messages=msgs,
            temperature=temperature, max_tokens=int(out_max) if out_max else 4096,
            num_retries=self._num_retries,
        )
        return _ShapedResponse(raw)


class _Chat:
    def __init__(self, completions):
        self.completions = completions


class BedrockOpenAIShim:
    """Duck-types `openai.OpenAI(...)` for the harness + ACE call sites.

    Used in two places:
      1. openevolve_runner.setup_run — replaces the `OpenAI(...)` client.
      2. ace/utils.py:initialize_clients (bedrock branch) — returns three
         instances (generator/reflector/curator).
    """

    def __init__(self, model_id: str, region_name: str = "us-east-1",
                 no_think: bool = False, num_retries: int = 3):
        client = _build_bedrock_client(region_name=region_name)
        self._model_id = model_id
        self._region = region_name
        self._client = client
        self.chat = _Chat(_BedrockCompletions(
            model_id=model_id, client=client,
            no_think=no_think, num_retries=num_retries,
        ))


# ---------------------------------------------------------------------------
#  DSPy LM for GEPA
# ---------------------------------------------------------------------------

class BedrockChatLM(dspy.LM):
    """DSPy LM backed by Bedrock Converse (bypasses litellm).

    `model_id` is the Bedrock foundation-model identifier, e.g.
    "qwen.qwen3-32b-v1:0". Constructor signature mirrors PortkeyChatLM /
    SageMakerChatLM so `_build_lm` can route by prefix without branching on
    arg shapes.
    """

    def __init__(
        self,
        model_id: str,
        region_name: str = "us-east-1",
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
            model=f"bedrock:{model_id}",
            model_type="chat",
            temperature=temperature,
            max_tokens=max_tokens,
            cache=False,
            **kwargs,
        )
        self._client = _build_bedrock_client(region_name=region_name)
        self._model_id = model_id
        self._region = region_name
        self._no_think = no_think
        self._num_retries = num_retries

    def __call__(self, prompt=None, messages=None, **kwargs):
        if messages is None:
            messages = [{"role": "user", "content": prompt or ""}]
        if self._no_think:
            messages = _inject_no_think(messages)

        max_tokens = kwargs.get("max_tokens", self.kwargs.get("max_tokens", 8192))
        temperature = kwargs.get("temperature", self.kwargs.get("temperature", 0.0))
        raw = _converse_with_retry(
            self._client, model_id=self._model_id, messages=messages,
            temperature=temperature, max_tokens=int(max_tokens) if max_tokens else 8192,
            num_retries=self._num_retries,
        )
        shaped = _ShapedResponse(raw)
        text = shaped.choices[0].message.content if shaped.choices else ""

        self.history.append({
            "prompt": prompt,
            "messages": messages,
            "kwargs": {
                "model": self._model_id, "region": self._region,
                "temperature": temperature, "max_tokens": max_tokens,
            },
            "response": None,  # raw boto3 response isn't JSON-serializable as-is
            "outputs": [text],
        })
        return [text]
