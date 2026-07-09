#!/usr/bin/env python3
import os
import re
import json
import openai
import tiktoken
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables from .env file
load_dotenv()

def _initialize_clients_base(api_provider):
    """Initialize separate clients for generator, reflector, and curator"""
    if api_provider == "sambanova":
        # Use SambaNova API
        base_url = "https://api.sambanova.ai/v1"
        api_key = os.getenv('SAMBANOVA_API_KEY', '')
        if not api_key:
            raise ValueError("SambaNova api key not found in environment variables")
    elif api_provider == "together":
        # Use Together API
        base_url = "https://api.together.xyz/v1"
        api_key = os.getenv('TOGETHER_API_KEY', '')
        if not api_key:
            raise ValueError("Together api key not found in environment variables")
    elif api_provider == "openai":
        # Use OpenAI API
        base_url = "https://api.openai.com/v1"
        api_key = os.getenv('OPENAI_API_KEY', '')
        if not api_key:
            raise ValueError("OpenAI api key not found in environment variables")
    elif api_provider == "commonstack":
        # Use Commonstack API
        base_url = "https://api.commonstack.ai/v1"
        api_key = os.getenv('COMMONSTACK_API_KEY', '')
        if not api_key:
            raise ValueError("Commonstack api key not found in environment variables")
    elif api_provider == "openrouter":
        # Use OpenRouter API (OpenAI-compatible)
        base_url = "https://openrouter.ai/api/v1"
        api_key = os.getenv('OPENROUTER_API_KEY', '')
        if not api_key:
            raise ValueError("OpenRouter api key not found in environment variables")
    elif api_provider == "alibaba":
        # Alibaba DashScope (OpenAI-compatible international endpoint)
        base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        api_key = os.getenv('DASHSCOPE_API_KEY', '')
        if not api_key:
            raise ValueError("DashScope api key not found in environment variables")
    elif api_provider == "portkey":
        # Portkey SDK (strict_open_ai_compliance=False). The Portkey route is
        # propagated via PORTKEY_ROUTE env (set by
        # methods/prompt_based/runners/ace_runner.py::_build_ace_instance from
        # cfg["model"]["task_lm"]); PORTKEY_API_KEY is read inside the shim
        # via os.environ.
        import sys
        from pathlib import Path
        repo_root = Path(__file__).resolve().parent.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from methods.prompt_based.portkey_lm import PortkeyOpenAIShim
        route = os.getenv("PORTKEY_ROUTE", "")
        if not route:
            raise ValueError(
                "PORTKEY_ROUTE env var must be set when api_provider='portkey'"
            )
        no_think = os.getenv("ACE_NO_THINK", "0") in ("1", "true", "True", "TRUE")
        generator_client = PortkeyOpenAIShim(route=route, no_think=no_think)
        reflector_client = PortkeyOpenAIShim(route=route, no_think=no_think)
        curator_client = PortkeyOpenAIShim(route=route, no_think=no_think)
        print(f"Using portkey API for all models (route={route}, no_think={no_think})")
        return generator_client, reflector_client, curator_client
    elif api_provider == "sagemaker":
        # AWS SageMaker endpoint via boto3. Returns duck-typed openai.OpenAI()
        # shims that route .chat.completions.create to invoke_endpoint.
        # Endpoint and region are propagated through env vars set by the
        # caller (methods/prompt_based/runners/ace_runner.py::_build_ace_instance):
        #   SAGEMAKER_ENDPOINT — required (e.g. "qwen3-8b-32k-east")
        #   AWS_DEFAULT_REGION — optional (defaults to "us-east-1")
        #   ACE_NO_THINK       — "1" to inject /no_think into messages
        # Routed this way so we don't have to widen initialize_clients's
        # signature, which is called from inside ACE's reference impl.
        import sys
        from pathlib import Path
        repo_root = Path(__file__).resolve().parent.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from methods.prompt_based.sagemaker_lm import SageMakerOpenAIShim
        endpoint = os.getenv("SAGEMAKER_ENDPOINT", "")
        if not endpoint:
            raise ValueError(
                "SAGEMAKER_ENDPOINT env var must be set when api_provider='sagemaker'"
            )
        region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        no_think = os.getenv("ACE_NO_THINK", "0") in ("1", "true", "True", "TRUE")
        generator_client = SageMakerOpenAIShim(
            endpoint_name=endpoint, region_name=region, no_think=no_think,
        )
        reflector_client = SageMakerOpenAIShim(
            endpoint_name=endpoint, region_name=region, no_think=no_think,
        )
        curator_client = SageMakerOpenAIShim(
            endpoint_name=endpoint, region_name=region, no_think=no_think,
        )
        print(f"Using sagemaker API for all models (endpoint={endpoint}, region={region}, no_think={no_think})")
        return generator_client, reflector_client, curator_client
    elif api_provider == "bedrock":
        # AWS Bedrock Converse via boto3. Auth via AWS_BEARER_TOKEN_BEDROCK
        # env var (or standard AWS creds as a fallback). Model + region are
        # propagated through env vars set by the caller
        # (methods/prompt_based/runners/ace_runner.py::_build_ace_instance):
        #   BEDROCK_MODEL_ID — required (e.g. "qwen.qwen3-32b-v1:0")
        #   BEDROCK_REGION   — optional (defaults to "us-east-1")
        #   ACE_NO_THINK     — "1" to inject /no_think into messages
        import sys
        from pathlib import Path
        repo_root = Path(__file__).resolve().parent.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from methods.prompt_based.bedrock_lm import BedrockOpenAIShim
        model_id = os.getenv("BEDROCK_MODEL_ID", "")
        if not model_id:
            raise ValueError(
                "BEDROCK_MODEL_ID env var must be set when api_provider='bedrock'"
            )
        region = os.getenv("BEDROCK_REGION", "us-east-1")
        no_think = os.getenv("ACE_NO_THINK", "0") in ("1", "true", "True", "TRUE")
        generator_client = BedrockOpenAIShim(
            model_id=model_id, region_name=region, no_think=no_think,
        )
        reflector_client = BedrockOpenAIShim(
            model_id=model_id, region_name=region, no_think=no_think,
        )
        curator_client = BedrockOpenAIShim(
            model_id=model_id, region_name=region, no_think=no_think,
        )
        print(f"Using bedrock API for all models (model_id={model_id}, region={region}, no_think={no_think})")
        return generator_client, reflector_client, curator_client
    else:
        raise ValueError(
            f"Invalid api_provider name: {api_provider}. Must be 'sambanova', 'together', 'openai', 'commonstack', 'openrouter', 'alibaba', 'sagemaker', or 'portkey'"
        )

    generator_client = openai.OpenAI(api_key=api_key, base_url=base_url)
    reflector_client = openai.OpenAI(api_key=api_key, base_url=base_url)
    curator_client = openai.OpenAI(api_key=api_key, base_url=base_url)

    print(f"Using {api_provider} API for all models")
    return generator_client, reflector_client, curator_client


# Known OpenAI-compatible providers and their default base URLs, used when the
# reflector is overridden onto a different provider than generator/curator.
_REFLECTOR_PROVIDER_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "together": "https://api.together.xyz/v1",
    "alibaba": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "sambanova": "https://api.sambanova.ai/v1",
    "commonstack": "https://api.commonstack.ai/v1",
}


def _maybe_override_reflector_client(default_client):
    """Return a dedicated reflector client when ACE_REFLECTOR_* env vars are set.

    Enables a split-provider ACE run: the generator + curator stay on the task
    provider while the reflector runs on a different one (e.g. task LM via
    Portkey->Fireworks, reflector direct to OpenAI GPT-5.2). The env vars are
    set by methods/prompt_based/runners/ace_runner.py::_build_ace_instance. When no override is
    configured this returns `default_client` unchanged, so single-provider runs
    behave exactly as before.

    Env vars:
      ACE_REFLECTOR_PROVIDER       — 'portkey' or an OpenAI-compatible provider
                                     name ('openai', 'openrouter', ...).
      ACE_REFLECTOR_PORTKEY_ROUTE  — required when provider == 'portkey'.
      ACE_REFLECTOR_API_KEY_ENV    — name of the env var holding the key
                                     (HTTP providers only).
      ACE_REFLECTOR_API_BASE       — optional base URL override (HTTP providers).
      ACE_REFLECTOR_NO_THINK       — '1' to inject /no_think (portkey only).
    """
    provider = os.getenv("ACE_REFLECTOR_PROVIDER", "").strip()
    if not provider:
        return default_client

    if provider == "portkey":
        route = os.getenv("ACE_REFLECTOR_PORTKEY_ROUTE", "").strip()
        if not route:
            raise ValueError(
                "ACE_REFLECTOR_PORTKEY_ROUTE must be set when "
                "ACE_REFLECTOR_PROVIDER='portkey'"
            )
        import sys
        from pathlib import Path
        repo_root = Path(__file__).resolve().parent.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from methods.prompt_based.portkey_lm import PortkeyOpenAIShim
        no_think = os.getenv("ACE_REFLECTOR_NO_THINK", "0") in ("1", "true", "True", "TRUE")
        print(f"Reflector override: portkey route={route} (no_think={no_think})")
        return PortkeyOpenAIShim(route=route, no_think=no_think)

    # OpenAI-compatible HTTP provider.
    base_url = os.getenv("ACE_REFLECTOR_API_BASE", "").strip() or \
        _REFLECTOR_PROVIDER_BASE_URLS.get(provider)
    if not base_url:
        raise ValueError(
            f"Unknown ACE_REFLECTOR_PROVIDER='{provider}'. Set "
            "ACE_REFLECTOR_API_BASE explicitly, or use a known provider: "
            f"{', '.join(sorted(_REFLECTOR_PROVIDER_BASE_URLS))}."
        )
    key_env = os.getenv("ACE_REFLECTOR_API_KEY_ENV", "").strip()
    if not key_env:
        raise ValueError(
            "ACE_REFLECTOR_API_KEY_ENV must name the env var holding the "
            "reflector's API key."
        )
    api_key = os.getenv(key_env, "")
    if not api_key:
        raise ValueError(
            f"Reflector API key env var '{key_env}' is unset or empty."
        )
    print(f"Reflector override: provider={provider} base_url={base_url} "
          f"key_env={key_env}")
    return openai.OpenAI(api_key=api_key, base_url=base_url)


def initialize_clients(api_provider):
    """Initialize generator/reflector/curator clients.

    Thin wrapper over `_initialize_clients_base` that additionally applies an
    optional reflector-only provider override (see
    `_maybe_override_reflector_client`). Generator + curator always use the
    task provider; only the reflector can be redirected.
    """
    generator_client, reflector_client, curator_client = \
        _initialize_clients_base(api_provider)
    reflector_client = _maybe_override_reflector_client(reflector_client)
    return generator_client, reflector_client, curator_client


def get_section_slug(section_name):
    """Convert section name to slug format (3-5 chars)"""
    # Common section mappings - updated to match original sections
    slug_map = {
        "financial_strategies_and_insights": "fin",
        "formulas_and_calculations": "calc",
        "code_snippets_and_templates": "code",
        "common_mistakes_to_avoid": "err",
        "problem_solving_heuristics": "prob",
        "context_clues_and_indicators": "ctx",
        "others": "misc",
        "meta_strategies": "meta"
    }
    
    # Clean and convert to snake_case
    clean_name = section_name.lower().strip().replace(" ", "_").replace("&", "and")
    
    if clean_name in slug_map:
        return slug_map[clean_name]
    
    # Generate slug from first letters
    words = clean_name.split("_")
    if len(words) == 1:
        return words[0][:4]
    else:
        return "".join(w[0] for w in words[:5])

def extract_boxed_content(text):
    """Helper function to extract content from \\boxed{} format"""
    pattern = r'\\boxed\{'
    match = re.search(pattern, text)
    if not match:
        return None
    
    start = match.end() - 1  # Position of opening brace
    brace_count = 0
    i = start
    
    while i < len(text):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                return text[start + 1:i]  # Content between braces
        i += 1
    return None

def extract_answer(response):
    """Extract final answer from model response"""
    try:
        # First try JSON parsing
        parsed = json.loads(response)
        answer = str(parsed.get("final_answer", "No final answer found"))
        return answer  
            
    except (json.JSONDecodeError, KeyError, AttributeError):
        # JSON parsing failed, use fallback logic
        matches = re.findall(r"Finish\[(.*?)\]", response)
        if matches:
            answer = matches[-1]
            return answer
        
        # Try to get final answer from JSON style response with regex matching 
        # Try double quotes first
        matches = re.findall(r'"final_answer"\s*:\s*"([^"]*)"', response)
        if matches:
            answer = matches[-1]
            return answer
        
        # Try single quotes
        matches = re.findall(r"'final_answer'\s*:\s*'([^']*)'", response)
        if matches:
            answer = matches[-1]
            return answer
        
        # Handle JSON format without quotes (for simple expressions)
        matches = re.findall(r'[\'"]final_answer[\'"]\s*:\s*([^,}]+)', response)
        if matches:
            answer = matches[-1].strip()
            # Clean up trailing characters
            answer = re.sub(r'[,}]*$', '', answer)
            return answer
        
        # Fallback for "The final answer is: X" pattern with boxed
        final_answer_pattern = r'[Tt]he final answer is:?\s*\$?\\boxed\{'
        match = re.search(final_answer_pattern, response)
        if match:
            # Extract boxed content starting from this match
            remaining_text = response[match.start():]
            boxed_content = extract_boxed_content(remaining_text)
            if boxed_content:
                return boxed_content
        
        # More general pattern for "final answer is X"
        matches = re.findall(r'[Tt]he final answer is:?\s*([^\n.]+)', response)
        if matches:
            answer = matches[-1].strip()
            # Clean up common formatting
            answer = re.sub(r'^\$?\\boxed\{([^}]+)\}\$?$', r'\1', answer)
            answer = answer.replace('$', '').strip()
            if answer:
                return answer
        
        return "No final answer found"
    
enc = tiktoken.get_encoding("cl100k_base")
def count_tokens(prompt: str) -> int:
    return len(enc.encode(prompt))


def evaluate_single_test_sample(args_tuple, data_processor) -> Tuple[Dict, str]:
    """
    Evaluate a single test sample - task-agnostic implementation.
    
    Args:
        args_tuple: Tuple of (index, task_dict, generator, playbook, max_tokens, log_dir, use_json_mode)
        data_processor: DataProcessor instance with answer_is_correct method
    """
    (i, task_dict, generator, playbook, max_tokens, log_dir, use_json_mode) = args_tuple
    try:
        context = task_dict["context"]
        question = task_dict["question"]
        target = task_dict["target"]

        gen_response, bullet_ids, call_info = generator.generate(
            question=question,
            playbook=playbook,
            context=context,
            reflection="(empty)",
            use_json_mode=use_json_mode,
            call_id=f"test_eval_{i}",
            log_dir=log_dir
        )

        final_answer = extract_answer(gen_response)
        is_correct = data_processor.answer_is_correct(final_answer, target)

        return {
            "index": i,
            "final_answer": final_answer,
            "target": target,
            "is_correct": is_correct,
            "success": True
        }, None

    except Exception as e:
        return None, f"Error evaluating sample {i}: {type(e).__name__}: {str(e)}"


def evaluate_test_set(data_processor, generator, playbook, test_samples,
                      max_tokens=4096, log_dir=None, max_workers=20, 
                      use_json_mode=False) -> Tuple[Dict, Dict]:
    """
    Parallel evaluation of test set - task-agnostic implementation.
    
    Args:
        data_processor: DataProcessor instance with answer_is_correct and evaluate_accuracy methods
        generator: Generator instance
        playbook: Current playbook string
        test_samples: List of test samples
        max_tokens: Max tokens for generation
        log_dir: Directory for logs
        max_workers: Number of parallel workers
        use_json_mode: Whether to use JSON mode
        
    Returns:
        Tuple of (results_dict, error_logs_dict)
    """
    print(f"\n{'='*40}")
    print(f"EVALUATING TEST SET - {len(test_samples)} samples, {max_workers} workers")
    print(f"{'='*40}")

    args_list = [
        (i, sample, generator, playbook, max_tokens, log_dir, use_json_mode)
        for i, sample in enumerate(test_samples)
    ]

    results = {
        "correct": 0, "total": 0, "no_answer": 0,
        "answers": [], "targets": [], "errors": []
    }

    # Use a wrapper to pass data_processor to the evaluation function
    def eval_wrapper(args_tuple):
        return evaluate_single_test_sample(args_tuple, data_processor)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_args = {
            executor.submit(eval_wrapper, args): args 
            for args in args_list
        }

        for i, future in enumerate(as_completed(future_to_args), 1):
            result, error = future.result()
            
            if error:
                print(error)
                continue

            if result and result["success"]:
                results["correct"] += (1 if result["is_correct"] else 0)
                results["total"] += 1
                results["answers"].append(result["final_answer"])
                results["targets"].append(result["target"])
                
                if not result["is_correct"]:
                    results["errors"].append({
                        "index": result["index"],
                        "prediction": result["final_answer"],
                        "ground_truth": result["target"]
                    })
                
                if result["final_answer"] == "No final answer found":
                    results["no_answer"] += 1

            if i % 50 == 0:
                curr_acc = results["correct"] / results["total"] if results["total"] > 0 else 0
                print(f"Progress: {i}/{len(args_list)}, Accuracy: {curr_acc:.3f}")
    
    if results["answers"] and results["targets"]:
        accuracy = data_processor.evaluate_accuracy(results["answers"], results["targets"])
        
        final_results = {
            "accuracy": accuracy,
            "correct": results["correct"],
            "total": results["total"],
            "no_answer": results["no_answer"]
        }
        
        error_logs = {
            "accuracy": accuracy,
            "errors": results["errors"]
        }
        
        print(f"\n📊 Final Accuracy: {accuracy:.3f} ({results['correct']}/{results['total']})")
    else:
        results = {"accuracy": 0.0, "correct": 0, "total": 0}
        error_logs = {}
        print(f"\n📊 No valid results!")
        
    return final_results, error_logs
