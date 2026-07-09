import os

import dspy


class HotpotQA(dspy.Signature):
    """Answer the question based on the provided context paragraphs."""
    # the above docstring becomes the instruction text in the prompt

    question = dspy.InputField()
    context = dspy.InputField()
    answer = dspy.OutputField()


def build_program():
    """Build an unoptimized chain-of-thought program for HotpotQA."""
    return dspy.ChainOfThought(HotpotQA)


def _thinking_extra_body(api_base: str | None):
    """Return the provider-specific extra_body payload that disables reasoning.

    DashScope (Alibaba) qwen3 uses `{"enable_thinking": False}`. OpenRouter and
    similar OpenAI-compatible passthroughs use `{"reasoning": {"enabled": False}}`.
    """
    base = (api_base or "").lower()
    if "dashscope" in base or "aliyuncs" in base:
        return {"enable_thinking": False}
    return {"reasoning": {"enabled": False}}


def _merge_extra_body(base: dict | None, *extras: dict | None) -> dict | None:
    """Shallow-merge extra_body dicts; later args override earlier ones."""
    merged = {}
    for part in (base, *extras):
        if part:
            merged.update(part)
    return merged or None


def _build_lm(model: str, *, temperature: float, max_tokens: int,
              thinking: bool | None, api_key: str | None, api_base: str | None,
              extra_body: dict | None, aws_region_name: str):
    """Construct a single DSPy LM for `model`.

    Routes `sagemaker:<endpoint>` strings to the boto3-backed
    `SageMakerChatLM` and `portkey:<route>` strings to the Portkey-SDK-backed
    `PortkeyChatLM`. Everything else flows through dspy.LM (litellm), so
    existing model profiles (OpenRouter+Alibaba, Gemini-via-Portkey-gateway,
    etc.) keep working without any config changes.
    """
    if model.startswith("sagemaker:"):
        from methods.prompt_based.sagemaker_lm import SageMakerChatLM
        return SageMakerChatLM(
            endpoint_name=model.split(":", 1)[1],
            region_name=aws_region_name,
            temperature=temperature,
            max_tokens=max_tokens,
            no_think=(thinking is False),
        )

    if model.startswith("bedrock:"):
        from methods.prompt_based.bedrock_lm import BedrockChatLM
        return BedrockChatLM(
            model_id=model.split(":", 1)[1],
            region_name=aws_region_name,
            temperature=temperature,
            max_tokens=max_tokens,
            no_think=(thinking is False),
        )

    if model.startswith("portkey:"):
        from methods.prompt_based.portkey_lm import PortkeyChatLM
        return PortkeyChatLM(
            route=model.split(":", 1)[1],
            temperature=temperature,
            max_tokens=max_tokens,
            no_think=(thinking is False),
        )

    eb = _merge_extra_body(
        extra_body,
        _thinking_extra_body(api_base) if thinking is False else None,
    )
    kwargs = {}
    if api_base:
        kwargs["api_base"] = api_base
    if eb is not None:
        kwargs["extra_body"] = eb
    return dspy.LM(
        model, api_key=api_key, temperature=temperature,
        max_tokens=max_tokens, **kwargs,
    )


def configure_lms(task_model: str, reflection_model: str, api_key: str | None = None,
                  api_base: str | None = None, api_key_env: str = "PORTKEY_API_KEY",
                  task_temperature: float = 0.7, task_max_tokens: int = 8192,
                  reflection_temperature: float = 1.0, reflection_max_tokens: int = 32000,
                  task_thinking: bool | None = None,
                  reflection_thinking: bool | None = None,
                  extra_body: dict | None = None,
                  aws_region_name: str = "us-east-1",
                  reflection_api_key: str | None = None,
                  reflection_api_key_env: str | None = None,
                  reflection_api_base: str | None = None):
    """Configure DSPy with a task LM and return a reflection LM for GEPA.

    Args:
        task_model: litellm model string ("openai/...", "openrouter/...") OR
            a SageMaker route ("sagemaker:<endpoint_name>") that bypasses
            litellm and calls boto3.invoke_endpoint directly.
        reflection_model: same conventions as task_model.
        api_key: API key (falls back to api_key_env environment variable).
            Skipped for SageMaker routes — AWS auth comes from boto3's
            standard credential chain (AWS_PROFILE / AWS_ACCESS_KEY_ID /
            ~/.aws/credentials).
        api_base: Base URL for the API (e.g. Portkey gateway, OpenRouter).
        api_key_env: Name of the env var holding the API key.
        task_temperature: Temperature for the task LM.
        task_max_tokens: Max tokens for the task LM.
        reflection_temperature: Temperature for the reflection LM.
        reflection_max_tokens: Max tokens for the reflection LM.
        task_thinking: If False, suppress reasoning/thinking mode on the task
            LM. For litellm models this is done via `extra_body`; for
            SageMaker it appends `/no_think` to messages and strips
            `<think>...</think>` blocks from responses.
        reflection_thinking: Same, for the reflection LM.
        aws_region_name: AWS region for SageMaker endpoints (default us-east-1).
            Ignored for litellm-routed models.
        reflection_api_key: Explicit API key for the reflection LM. When None
            (and the env-var/fallback below also yield nothing reflection-
            specific), the reflection LM reuses the task LM's key — i.e. the
            original single-provider behaviour is preserved.
        reflection_api_key_env: Env var holding the reflection LM's key. Set
            this to run the reflector on a *different* provider/account than
            the task LM (e.g. task via Portkey->Fireworks, reflector direct to
            OpenAI). Ignored if reflection_api_key is given explicitly.
        reflection_api_base: Base URL for the reflection LM. Falls back to
            api_base when None.
    """
    # --- Resolve the task LM's API key ---
    # SageMaker routes carry no key (boto3 credential chain handles auth).
    # Bedrock routes use AWS_BEARER_TOKEN_BEDROCK (or AWS creds) picked up by
    # boto3 directly — also no harness-side key needed.
    _is_aws_route = lambda m: m.startswith("sagemaker:") or m.startswith("bedrock:")
    if _is_aws_route(task_model):
        task_api_key = None
    elif api_key is not None:
        task_api_key = api_key
    else:
        task_api_key = os.environ[api_key_env]

    # --- Resolve the reflection LM's API key/base ---
    # Precedence: explicit reflection_api_key > reflection_api_key_env >
    # fall back to the task LM's key. The fallback is what keeps existing
    # single-provider configs working unchanged.
    if _is_aws_route(reflection_model):
        reflection_api_key = None
    elif reflection_api_key is not None:
        pass  # explicit override wins
    elif reflection_api_key_env is not None:
        reflection_api_key = os.environ[reflection_api_key_env]
    else:
        # No reflection-specific override: reuse the task key. If the task is
        # SageMaker/Bedrock (task_api_key is None), still resolve from
        # api_key_env so a non-AWS reflector gets a usable key.
        reflection_api_key = (
            task_api_key if task_api_key is not None
            else os.environ[api_key_env]
        )
    reflection_api_base = (
        reflection_api_base if reflection_api_base is not None else api_base
    )

    task_lm = _build_lm(
        task_model, temperature=task_temperature, max_tokens=task_max_tokens,
        thinking=task_thinking, api_key=task_api_key, api_base=api_base,
        extra_body=extra_body, aws_region_name=aws_region_name,
    )
    dspy.configure(lm=task_lm)

    reflection_lm = _build_lm(
        reflection_model, temperature=reflection_temperature,
        max_tokens=reflection_max_tokens, thinking=reflection_thinking,
        api_key=reflection_api_key, api_base=reflection_api_base,
        extra_body=extra_body, aws_region_name=aws_region_name,
    )
    return reflection_lm


def run_gepa(program, trainset, valset, metric, reflection_lm, budget="light",
             max_metric_calls=None, num_threads=2, reflection_minibatch_size=3):
    """Run GEPA optimization and return the optimized program.

    Exactly one of budget or max_metric_calls should be set.
    budget="light" auto-calculates ~420 calls for 200 examples.
    max_metric_calls directly caps the number of example evaluations.
    """
    if max_metric_calls is not None:
        optimizer = dspy.GEPA(
            metric=metric,
            max_metric_calls=max_metric_calls,
            reflection_lm=reflection_lm,
            num_threads=num_threads,
            track_stats=True,
            reflection_minibatch_size=reflection_minibatch_size,
        )
    else:
        optimizer = dspy.GEPA(
            metric=metric,
            auto=budget,
            reflection_lm=reflection_lm,
            num_threads=num_threads,
            track_stats=True,
            reflection_minibatch_size=reflection_minibatch_size,
        )
    return optimizer.compile(program, trainset=trainset, valset=valset)
