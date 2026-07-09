"""SageMaker boto3 adapters used by both GEPA and ACE/OE paths.

Two surfaces are exposed:

  - SageMakerChatLM   — DSPy LM subclass used by GEPA (litellm bypass).
  - SageMakerOpenAIShim — duck-typed openai.OpenAI() client used by the
    harness (`_call_llm` in openevolve_runner.py) and ACE's internal
    generator/reflector/curator (`ace/llm.py:timed_llm_call`).

Both share boto3 client construction with hard timeouts and disabled
botocore auto-retries (we manage retries ourselves so timeouts aren't
silently masked).

AWS auth follows boto3's standard credential chain: AWS_PROFILE,
AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, or ~/.aws/credentials.
"""

from __future__ import annotations

import json
import re
import time

import boto3
import dspy
from botocore.config import Config as BotoConfig


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _build_boto3_client(region_name: str, read_timeout: int = 120,
                        connect_timeout: int = 10):
    """Construct a sagemaker-runtime client with hard timeouts.

    `retries={"max_attempts": 1}` disables botocore's auto-retry layer so
    callers see ReadTimeoutError directly. Both SageMakerChatLM and
    SageMakerOpenAIShim do their own bounded retry loops with backoff.
    """
    cfg = BotoConfig(
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
        retries={"max_attempts": 1, "mode": "standard"},
    )
    return boto3.client("sagemaker-runtime", region_name=region_name, config=cfg)


def _inject_no_think(messages):
    """Append `/no_think` to the last user message (idempotent)."""
    msgs = [dict(m) for m in messages]
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].get("role") == "user":
            content = msgs[i].get("content", "") or ""
            if "/no_think" not in content:
                msgs[i]["content"] = content.rstrip() + "\n\n/no_think"
            break
    return msgs


def _invoke_with_retry(sm_client, endpoint, body, num_retries=3):
    last_err = None
    for attempt in range(num_retries):
        try:
            resp = sm_client.invoke_endpoint(
                EndpointName=endpoint,
                ContentType="application/json",
                Body=json.dumps(body),
            )
            return json.loads(resp["Body"].read())
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
    __slots__ = ("content", "role")
    def __init__(self, content: str, role: str = "assistant"):
        self.content = content
        self.role = role


class _Choice:
    __slots__ = ("message", "finish_reason", "index")
    def __init__(self, message: _Msg, finish_reason: str = "stop", index: int = 0):
        self.message = message
        self.finish_reason = finish_reason
        self.index = index


class _Usage:
    __slots__ = ("prompt_tokens", "completion_tokens", "total_tokens")
    def __init__(self, d: dict):
        d = d or {}
        self.prompt_tokens = int(d.get("prompt_tokens", 0) or 0)
        self.completion_tokens = int(d.get("completion_tokens", 0) or 0)
        self.total_tokens = int(d.get("total_tokens", 0) or 0)


class _OpenAIShapedResponse:
    """Looks enough like an OpenAI ChatCompletion response for our callers.

    The two callers we need to satisfy:
      - openevolve_runner._call_llm: reads response.choices[0].message.content
      - ace/llm.py:timed_llm_call:    reads response.choices, .message.content,
                                      and (optionally) usage fields
    """
    def __init__(self, raw: dict):
        choices_raw = raw.get("choices") or []
        choices = []
        for i, c in enumerate(choices_raw):
            msg = c.get("message") or {}
            content = msg.get("content") or ""
            content = _THINK_RE.sub("", content).strip()
            choices.append(_Choice(
                message=_Msg(content, role=msg.get("role", "assistant")),
                finish_reason=c.get("finish_reason", "stop"),
                index=i,
            ))
        self.choices = choices
        self.usage = _Usage(raw.get("usage") or {})
        self.model = raw.get("model", "")
        self.id = raw.get("id", "")


class _SageMakerCompletions:
    """Implements `client.chat.completions.create(...)`."""

    def __init__(self, endpoint: str, sm_client, no_think: bool = False,
                 num_retries: int = 3):
        self._endpoint = endpoint
        self._sm = sm_client
        self._no_think = no_think
        self._num_retries = num_retries

    def create(self, *, messages, model=None, temperature=0.0, max_tokens=8192,
               extra_body=None, response_format=None,
               max_completion_tokens=None, **_kwargs):
        # `model` is encoded by the endpoint URL on SageMaker — drop it.
        # `extra_body` (provider-pin / reasoning toggles) is OpenRouter/
        # DashScope-specific and meaningless for SageMaker — drop it.
        # `response_format` (json_object) is honored only if the server
        # supports it; we just drop it to avoid 4xx on stricter servers.
        # `max_completion_tokens` is OpenAI's newer name — coerce.
        msgs = list(messages)
        if self._no_think:
            msgs = _inject_no_think(msgs)
        out_max = max_completion_tokens if max_completion_tokens is not None else max_tokens
        body = {
            "messages": msgs,
            "temperature": float(temperature) if temperature is not None else 0.0,
            "max_tokens": int(out_max) if out_max else 8192,
        }
        result = _invoke_with_retry(
            self._sm, self._endpoint, body, num_retries=self._num_retries,
        )
        return _OpenAIShapedResponse(result)


class _Chat:
    def __init__(self, completions):
        self.completions = completions


class SageMakerOpenAIShim:
    """Duck-types `openai.OpenAI(...)` for the harness + ACE call sites.

    Used in two places:
      1. openevolve_runner.setup_run — replaces the `OpenAI(...)` client
         that `_call_llm` calls into. Same `client.chat.completions.create()`
         interface, but routes to SageMaker.
      2. ace/utils.py:initialize_clients (sagemaker branch) — returns three
         instances (generator/reflector/curator) so ACE's internal loop
         calls into SageMaker via the same shape.
    """

    def __init__(self, endpoint_name: str, region_name: str = "us-east-1",
                 no_think: bool = False, num_retries: int = 3,
                 read_timeout: int = 120, connect_timeout: int = 10):
        sm = _build_boto3_client(region_name, read_timeout=read_timeout,
                                 connect_timeout=connect_timeout)
        self._endpoint = endpoint_name
        self._sm = sm
        self.chat = _Chat(_SageMakerCompletions(
            endpoint=endpoint_name, sm_client=sm,
            no_think=no_think, num_retries=num_retries,
        ))


# ---------------------------------------------------------------------------
#  DSPy LM for GEPA
# ---------------------------------------------------------------------------

class SageMakerChatLM(dspy.LM):
    """DSPy LM backed by sagemaker-runtime.invoke_endpoint.

    `endpoint_name` is the SageMaker endpoint deployed by the colleague
    (e.g. "qwen3-8b-32k-east"). When `no_think=True`, "/no_think" is
    appended to the last user message so qwen3 suppresses its reasoning
    block; the empty `<think></think>` shell the server still emits is
    stripped from the returned text.
    """

    def __init__(
        self,
        endpoint_name: str,
        region_name: str = "us-east-1",
        temperature: float = 0.0,
        max_tokens: int = 8192,
        no_think: bool = True,
        num_retries: int = 3,
        read_timeout: int = 120,
        connect_timeout: int = 10,
        **kwargs,
    ):
        kwargs.pop("api_key", None)
        kwargs.pop("api_base", None)
        kwargs.pop("extra_body", None)
        super().__init__(
            model=f"sagemaker:{endpoint_name}",
            model_type="chat",
            temperature=temperature,
            max_tokens=max_tokens,
            cache=False,
            **kwargs,
        )
        self._client = _build_boto3_client(
            region_name, read_timeout=read_timeout, connect_timeout=connect_timeout,
        )
        self._endpoint = endpoint_name
        self._no_think = no_think
        self._num_retries = num_retries

    def __call__(self, prompt=None, messages=None, **kwargs):
        if messages is None:
            messages = [{"role": "user", "content": prompt or ""}]
        if self._no_think:
            messages = _inject_no_think(messages)

        body = {
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.kwargs.get("max_tokens", 8192)),
            "temperature": kwargs.get("temperature", self.kwargs.get("temperature", 0.0)),
        }
        result = _invoke_with_retry(self._client, self._endpoint, body,
                                    num_retries=self._num_retries)

        text = (result.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
        text = _THINK_RE.sub("", text).strip()

        self.history.append({
            "prompt": prompt,
            "messages": messages,
            "kwargs": body,
            "response": result,
            "outputs": [text],
        })
        return [text]

    @staticmethod
    def _inject_no_think(messages):
        return _inject_no_think(messages)
