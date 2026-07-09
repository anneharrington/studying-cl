"""OpenEvolve evaluator for tooluse prompt optimization."""

import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from openai import OpenAI

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cl.evals.toolalpaca import _parse_actions, _score_actions
from cl.evals.tooluse import load_tooluse_raw
from cl.utils.token_tracker import record_usage_to_file

CONFIG_PATH = os.environ.get("BENCHMARK_CONFIG", "configs/openevolve_tooluse.yaml")
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

API_KEY = os.environ.get(cfg["model"].get("api_key_env", "PORTKEY_API_KEY"), "")
API_BASE = cfg["model"]["api_base"]
TASK_MODEL = cfg["model"]["task_lm"]
MAX_RETRIES = 3
EVAL_NUM_THREADS = cfg.get("eval_num_threads", 8)
THINKING_ENABLED = cfg["model"].get("task_thinking", cfg["model"].get("thinking", True))
USER_EXTRA_BODY = cfg["model"].get("extra_body")


def _disable_thinking_extra():
    base = (API_BASE or "").lower()
    if "dashscope" in base or "aliyuncs" in base:
        return {"enable_thinking": False}
    return {"reasoning": {"enabled": False}}


def _build_extra_body():
    merged = {}
    if USER_EXTRA_BODY:
        merged.update(USER_EXTRA_BODY)
    if not THINKING_ENABLED:
        merged.update(_disable_thinking_extra())
    return merged or None


client = OpenAI(base_url=API_BASE, api_key=API_KEY)
# Loader returns (train, val, eval) on the parquet path and (train, val) on
# legacy jsonl. Keep only train_set; tolerate either arity.
train_set = load_tooluse_raw(
    path=cfg["dataset"]["path"],
    train_n=cfg["dataset"]["train_n"],
    val_n=0,
    seed=cfg["dataset"]["seed"],
)[0]
print(f"[evaluator] Loaded {len(train_set)} train examples, model={TASK_MODEL}, threads={EVAL_NUM_THREADS}")


def _call_llm(prompt_text: str) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            create_kwargs = dict(
                model=TASK_MODEL,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=0.7,
                max_tokens=8192,
            )
            _eb = _build_extra_body()
            if _eb is not None:
                create_kwargs["extra_body"] = _eb
            response = client.chat.completions.create(**create_kwargs)
            record_usage_to_file(response)
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"[evaluator] LLM call failed after {MAX_RETRIES} attempts: {e}")
                return ""
            time.sleep(2 ** attempt)
    return ""


def _score_one(prompt_template: str, ex: dict) -> float:
    try:
        formatted = prompt_template.format(question=ex["question"])
    except (KeyError, ValueError, IndexError):
        return 0.0

    response = _call_llm(formatted)
    pred_actions = _parse_actions(response)
    score, _ = _score_actions(pred_actions, ex["golden_steps"])
    return score


def _score_prompt(prompt_template: str, examples: list[dict]) -> float:
    n = len(examples)
    if n == 0:
        return 0.0

    total_score = 0.0
    done = 0
    lock = threading.Lock()

    def _on_complete(score):
        nonlocal total_score, done
        with lock:
            total_score += score
            done += 1
            if done % 20 == 0 or done == n:
                avg = total_score / done
                print(f"[evaluator] tooluse: {done}/{n} examples, running score={avg:.3f}")

    with ThreadPoolExecutor(max_workers=EVAL_NUM_THREADS) as pool:
        futures = [pool.submit(_score_one, prompt_template, ex) for ex in examples]
        for future in as_completed(futures):
            _on_complete(future.result())

    return total_score / n


def _validate_prompt_structure(prompt_template: str) -> None:
    if "[GENERAL_INSTRUCTIONS]" not in prompt_template:
        raise ValueError("prompt missing [GENERAL_INSTRUCTIONS] opening delimiter")
    if "[/GENERAL_INSTRUCTIONS]" not in prompt_template:
        raise ValueError("prompt missing [/GENERAL_INSTRUCTIONS] closing delimiter")
    if prompt_template.index("[GENERAL_INSTRUCTIONS]") >= prompt_template.index("[/GENERAL_INSTRUCTIONS]"):
        raise ValueError("opening delimiter must appear before closing delimiter")
    if "{question}" not in prompt_template:
        raise ValueError("prompt missing required {question} placeholder")
    try:
        prompt_template.format(question="__probe__")
    except (KeyError, ValueError, IndexError) as e:
        raise ValueError(f"prompt str.format() failed: {e}")


def evaluate(prompt_path: str) -> dict:
    with open(prompt_path) as f:
        prompt_template = f.read().strip()

    _validate_prompt_structure(prompt_template)

    score = _score_prompt(prompt_template, train_set)
    return {"combined_score": score}
