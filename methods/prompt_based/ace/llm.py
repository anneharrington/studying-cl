"""
==============================================================================
llm.py
==============================================================================

This file contains the LLM class for the project.

"""
import json
import os
import re
import time
import random
from datetime import datetime
import openai
from logger import log_llm_call, log_problematic_request


# OpenAI's GPT-5 / o-series reasoning models reject an explicit `temperature`
# (only the fixed default of 1 is allowed) and require `max_completion_tokens`
# instead of `max_tokens`. Match on the model name so a split-provider run that
# points the reflector at e.g. "gpt-5.2" sends a compliant payload.
_REASONING_MODEL_RE = re.compile(r"(^|[/@])(gpt-5|o[1-4])", re.IGNORECASE)


def _is_reasoning_model(model: str) -> bool:
    return bool(_REASONING_MODEL_RE.search(model or ""))


# Tokens the Alibaba content-safety filter uses in 400 responses. We match on
# substrings so error-format changes don't silently turn into hard crashes.
_CONTENT_FILTER_TOKENS = (
    "data_inspection_failed",
    "inappropriate content",
    "Input data may contain inappropriate content",
)

# Sentinel a censored sample emits in place of a real model response. Shaped
# as a parseable JSON object so ACE's downstream parsers (extract_answer,
# extract_bullet_ids) handle it gracefully — the sample is graded incorrect
# (which it would be anyway) and contributes no bullets to the playbook.
_CONTENT_FILTER_SENTINEL = (
    '{"final_answer": "[CONTENT_FILTER_SKIP]", "bullet_ids": [], '
    '"reasoning": "Input rejected by provider content-safety filter; '
    'sample skipped."}'
)


def _is_content_filter_error(err) -> bool:
    """True iff `err` looks like an Alibaba/OpenRouter content-safety rejection.

    The error surfaces as `openai.BadRequestError` with the offending tokens
    in either the message or the response body. Match on substring rather
    than exact text to survive minor wording changes from the provider.
    """
    if not isinstance(err, openai.BadRequestError):
        # Some SDK versions wrap in APIError instead — fall through to text match.
        if not (hasattr(openai, "APIError") and isinstance(err, openai.APIError)):
            return False
    blob = str(err).lower()
    body = ""
    resp = getattr(err, "response", None)
    if resp is not None:
        body = (getattr(resp, "text", "") or "").lower()
    haystack = blob + " " + body
    return any(tok.lower() in haystack for tok in _CONTENT_FILTER_TOKENS)


def _log_content_filter_skip(log_dir, call_id, role, model, prompt, error):
    """Append the censored sample to <log_dir>/skipped_samples.jsonl."""
    if not log_dir:
        return
    try:
        os.makedirs(log_dir, exist_ok=True)
        record = {
            "timestamp": datetime.now().isoformat(),
            "call_id": call_id,
            "role": role,
            "model": model,
            "prompt_length": len(prompt or ""),
            "prompt_preview": (prompt or "")[:500],
            "error": str(error)[:500],
        }
        with open(os.path.join(log_dir, "skipped_samples.jsonl"), "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as log_err:
        # Logging failures must never break the run.
        print(f"[CENSOR-SKIP] failed to log to skipped_samples.jsonl: {log_err}")

def _disable_thinking_extra_body(api_provider):
    """Provider-specific payload that disables qwen3's reasoning/thinking mode.

    DashScope (alibaba) qwen3 expects `{"enable_thinking": False}`; OpenRouter's
    passthrough uses `{"reasoning": {"enabled": False}}`. Returns None for
    providers without a known flag — in which case we simply don't inject.
    """
    if api_provider in ("alibaba", "dashscope"):
        return {"enable_thinking": False}
    if api_provider == "openrouter":
        return {"reasoning": {"enabled": False}}
    return None


def timed_llm_call(client, api_provider, model, prompt, role, call_id, max_tokens=4096, log_dir=None,
                   sleep_seconds=15, retries_on_timeout=1000, attempt=1, use_json_mode=False,
                   enable_thinking=True, extra_body=None):
    """
    Make a timed LLM call with error handling and retry logic.
    
    EMPTY RESPONSE HANDLING STRATEGY:
    - Training calls (call_id starts with 'train_'): Skip the entire training sample
    - Test calls (call_id starts with 'test_'): Mark as incorrect (return wrong answers)
    - All empty responses are logged to problematic_requests/ for SambaNova support analysis
    
    For test calls specifically: Returns "INCORRECT_DUE_TO_EMPTY_RESPONSE" repeated 4 times
    (comma-separated) to handle the 4-question format used in financial NER evaluation.
    
    Args:
        client: API client
        model: Model name to use
        prompt: Text prompt to send
        role: Role for logging (generator, reflector, curator)
        call_id: Unique identifier for this call (format: {train|test}_{role}_{details})
        max_tokens: Maximum tokens to generate
        log_dir: Directory for detailed logging
        sleep_seconds: Base sleep time between retries
        retries_on_timeout: Maximum number of retries for timeouts/rate limits/empty responses
        attempt: Current attempt number (for recursive calls)
        use_json_mode: Whether to use JSON mode for structured output
    
    Returns:
        tuple: (response_text, call_info_dict)
        
    Special return values for empty responses:
        - Training: ("INCORRECT_DUE_TO_EMPTY_RESPONSE, INCORRECT_DUE_TO_EMPTY_RESPONSE, ...", call_info)
        - Testing: ("INCORRECT_DUE_TO_EMPTY_RESPONSE, INCORRECT_DUE_TO_EMPTY_RESPONSE, ...", call_info)
    """
    start_time = time.time()
    prompt_time = time.time()
    
    print(f"[{role.upper()}] Starting call {call_id}...")
    
    # Check if we're using API key mixer for dynamic key rotation on retries
    using_key_mixer = False
    
    while True:
        try:
            # Get client
            active_client = client

            # Prepare API call parameters
            is_reasoning = _is_reasoning_model(model)
            # GPT-5 / o-series reasoning models need `max_completion_tokens`;
            # so does the native OpenAI provider for all models.
            if api_provider == "openai" or is_reasoning:
                max_tokens_key = "max_completion_tokens"
            else:
                max_tokens_key = "max_tokens"

            api_params = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                max_tokens_key: max_tokens
            }
            # Reasoning models reject an explicit temperature — omit it so the
            # API uses its fixed default. Everything else keeps temperature 0.0.
            if not is_reasoning:
                api_params["temperature"] = 0.0

            # Add JSON mode if requested
            if use_json_mode:
                api_params["response_format"] = {"type": "json_object"}

            # Merge model-configured extra_body with the provider-specific
            # payload that disables qwen3 thinking/reasoning when requested.
            _merged_eb = {}
            if extra_body:
                _merged_eb.update(extra_body)
            if not enable_thinking:
                _extra = _disable_thinking_extra_body(api_provider)
                if _extra is not None:
                    _merged_eb.update(_extra)
            if _merged_eb:
                api_params["extra_body"] = _merged_eb
            call_start = time.time()
            response = active_client.chat.completions.create(**api_params)
            call_end = time.time()
            
            # Check if response is valid
            if not response or not response.choices or len(response.choices) == 0:
                raise Exception("Empty response from API")
            
            response_time = time.time()
            total_time = response_time - start_time
            response_content = response.choices[0].message.content
            
            if response_content is None:
                raise Exception("API returned None content")
            
            call_info = {
                "role": role,
                "call_id": call_id,
                "model": model,
                "prompt": prompt,
                "response": response_content,
                "prompt_time": prompt_time - start_time,
                "response_time": response_time - prompt_time,
                "total_time": total_time,
                "call_time": call_end - call_start,
                "prompt_length": len(prompt),
                "response_length": len(response_content),
                "prompt_num_tokens": response.usage.prompt_tokens,
                "response_num_tokens": response.usage.completion_tokens,
            }
            
            print(f"[{role.upper()}] Call {call_id} completed in {total_time:.2f}s")
            
            if log_dir:
                log_llm_call(log_dir, call_info)
            
            return response_content, call_info
            
        except Exception as e:
            # Content-filter rejection (Alibaba data_inspection_failed). Skip the
            # sample rather than abort: log it, return a benign sentinel response
            # so the parser sees a syntactically valid (but incorrect) result and
            # ACE moves on to the next sample. The sample is counted as wrong in
            # scoring (which it would be anyway since we can't get a real answer)
            # and emits zero bullets, so the playbook isn't polluted.
            if _is_content_filter_error(e):
                _log_content_filter_skip(log_dir, call_id, role, model, prompt, e)
                print(f"[{role.upper()}] [censor-skip] {call_id}: input rejected by content filter; "
                      f"emitting sentinel + advancing")
                error_time = time.time()
                call_info = {
                    "role": role,
                    "call_id": call_id,
                    "model": model,
                    "prompt_length": len(prompt or ""),
                    "response_length": len(_CONTENT_FILTER_SENTINEL),
                    "total_time": error_time - start_time,
                    "skipped": "content_filter",
                    "skip_reason": str(e)[:200],
                    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3],
                }
                if log_dir:
                    log_llm_call(log_dir, call_info)
                return _CONTENT_FILTER_SENTINEL, call_info

            # Check for both timeout and rate limit errors
            is_timeout = any(k in str(e).lower() for k in ["timeout", "timed out", "connection"])
            is_rate_limit = any(k in str(e).lower() for k in ["rate limit", "429", "rate_limit_exceeded"])
            is_empty_response = "empty response" in str(e).lower() or "api returned none content" in str(e).lower()

            # Malformed response body from provider (truncated stream, HTML error page, etc.)
            # surfaces as a JSONDecodeError from the OpenAI SDK's body parser, or as an
            # openai.APIError/APIResponseValidationError wrapping it. Treat as retryable.
            is_decode_error = isinstance(e, json.JSONDecodeError) or "expecting value" in str(e).lower()
            if hasattr(openai, 'APIResponseValidationError') and isinstance(e, openai.APIResponseValidationError):
                is_decode_error = True
            if hasattr(openai, 'APIError') and isinstance(e, openai.APIError) and is_decode_error:
                is_decode_error = True
            
            # Check for server errors (500, 502, 503, etc.) that should be retried
            is_server_error = False
            if hasattr(e, 'response'):
                try:
                    status_code = getattr(e.response, 'status_code', None)
                    if status_code and status_code >= 500:
                        is_server_error = True
                        print(f"[{role.upper()}] Server error detected: HTTP {status_code}")
                except:
                    pass
            
            # Also check for 500 errors in the error message itself
            if any(k in str(e).lower() for k in ["500 internal server error", "internal server error", "502 bad gateway", "503 service unavailable"]):
                is_server_error = True
                print(f"[{role.upper()}] Server error detected in message: {str(e)[:100]}...")
            
            # Also check for specific OpenAI exceptions
            if hasattr(openai, 'RateLimitError') and isinstance(e, openai.RateLimitError):
                is_rate_limit = True
            
            # Check for OpenAI InternalServerError
            if hasattr(openai, 'InternalServerError') and isinstance(e, openai.InternalServerError):
                is_server_error = True
                print(f"[{role.upper()}] OpenAI InternalServerError detected")
            
            # Debug empty response issues
            if is_empty_response:
                print(f"\n🚨 DEBUG: Empty response detected for {call_id}")
                print(f"📝 Exception type: {type(e).__name__}")
                print(f"📝 Exception message: {str(e)}")
                print(f"📝 Using JSON mode: {use_json_mode}")
                print(f"📝 Model: {model}")
                print(f"📝 Prompt length: {len(prompt)}")
                print(f"📝 Prompt preview (first 500 chars):")
                print(f"    {prompt[:500]}...")
                print(f"📝 Full exception details: {repr(e)}")
                if hasattr(e, 'response'):
                    print(f"📝 Raw response object: {e.response}")
                    if hasattr(e.response, 'text'):
                        print(f"📝 Raw response text: {e.response.text}")
                    if hasattr(e.response, 'content'):
                        print(f"📝 Raw response content: {e.response.content}")
                print("-" * 60)
                
                # Log problematic requests for SambaNova support
                log_problematic_request(call_id, prompt, model, api_params, e, log_dir, using_key_mixer, 
                                       client if using_key_mixer else None)
            
            # For empty responses, we handle differently based on context
            if is_empty_response:
                # Log the problematic request for SambaNova support
                log_problematic_request(call_id, prompt, model, api_params, e, log_dir, using_key_mixer, 
                                       client if using_key_mixer else None)
                
                # Check if this is a training or test call to decide behavior
                if call_id.startswith('train_'):
                    # In training: Mark as incorrect answer (same as testing)
                    print(f"[{role.upper()}] 🚨 Empty response in training - marking as INCORRECT for {call_id}")
                    error_time = time.time()
                    call_info = {
                        "role": role,
                        "call_id": call_id,
                        "model": model,
                        "prompt": prompt,
                        "error": "TRAINING_INCORRECT: " + str(e),
                        "total_time": error_time - start_time,
                        "prompt_length": len(prompt),
                        "response_length": 0,
                        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3],
                        "datetime": datetime.now().isoformat(),
                        "training_marked_incorrect_due_to_empty_response": True
                    }
                    if log_dir:
                        log_llm_call(log_dir, call_info)
                    
                    # Return a response that will be marked as incorrect
                    # For the 4-question format, we return 4 wrong answers
                    incorrect_response = "INCORRECT_DUE_TO_EMPTY_RESPONSE, INCORRECT_DUE_TO_EMPTY_RESPONSE, INCORRECT_DUE_TO_EMPTY_RESPONSE, INCORRECT_DUE_TO_EMPTY_RESPONSE"
                    return incorrect_response, call_info
                
                elif call_id.startswith('test_'):
                    # In testing: Treat as incorrect answer
                    print(f"[{role.upper()}] 🚨 Empty response in testing - marking as INCORRECT for {call_id}")
                    error_time = time.time()
                    call_info = {
                        "role": role,
                        "call_id": call_id,
                        "model": model,
                        "prompt": prompt,
                        "error": "TEST_INCORRECT: " + str(e),
                        "total_time": error_time - start_time,
                        "prompt_length": len(prompt),
                        "response_length": 0,
                        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3],
                        "datetime": datetime.now().isoformat(),
                        "test_marked_incorrect_due_to_empty_response": True
                    }
                    if log_dir:
                        log_llm_call(log_dir, call_info)
                    
                    # Return a response that will be marked as incorrect
                    # For the 4-question format, we return 4 wrong answers
                    incorrect_response = "INCORRECT_DUE_TO_EMPTY_RESPONSE, INCORRECT_DUE_TO_EMPTY_RESPONSE, INCORRECT_DUE_TO_EMPTY_RESPONSE, INCORRECT_DUE_TO_EMPTY_RESPONSE"
                    return incorrect_response, call_info
            
            # Retry logic for timeouts, rate limits, server errors, and malformed bodies
            if (is_timeout or is_rate_limit or is_server_error or is_decode_error) and attempt < retries_on_timeout:
                attempt += 1
                if is_rate_limit:
                    error_type = "rate limited"
                    base_sleep = sleep_seconds * 2
                elif is_server_error:
                    error_type = "server error (500+)"
                    base_sleep = sleep_seconds * 1.5  # Moderate delay for server errors
                elif is_decode_error:
                    error_type = "malformed response body (JSON decode failed)"
                    base_sleep = sleep_seconds * 1.5
                elif is_empty_response:
                    error_type = "returned empty response"
                    base_sleep = sleep_seconds
                else:
                    error_type = "timed out"
                    base_sleep = sleep_seconds
                jitter = random.uniform(0.5, 1.5)  # Add jitter to avoid thundering herd
                sleep_time = base_sleep * jitter
                print(f"[{role.upper()}] Call {call_id} {error_type}, sleeping {sleep_time:.1f}s then retrying "
                      f"({attempt}/{retries_on_timeout})...")
                time.sleep(sleep_time)
                continue
            
            error_time = time.time()
            call_info = {
                "role": role,
                "call_id": call_id,
                "model": model,
                "prompt": prompt,
                "error": str(e),
                "total_time": error_time - start_time,
                "prompt_length": len(prompt),
                "attempt": attempt,
            }
            
            print(f"[{role.upper()}] Call {call_id} failed after {error_time - start_time:.2f}s: {e}")
            
            if log_dir:
                log_llm_call(log_dir, call_info)
            
            raise e
