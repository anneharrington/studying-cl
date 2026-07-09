"""OpenEvolve evaluator for SciKnowEval (Chemistry L3) prompt optimization.

OpenEvolve calls evaluate(prompt_path) -> dict with "combined_score".
Configuration is loaded from the YAML file pointed to by the BENCHMARK_CONFIG
environment variable (set by the run script).
"""

import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from openai import OpenAI

# Add project root to path so src/ imports work
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cl.evals.sciknoweval import load_sciknoweval_raw
from cl.utils.token_tracker import record_usage_to_file

# --- Load configuration from YAML ---
CONFIG_PATH = os.environ.get("BENCHMARK_CONFIG", "configs/openevolve_sciknoweval.yaml")
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

API_KEY = os.environ.get(cfg["model"].get("api_key_env", "PORTKEY_API_KEY"), "")
API_BASE = cfg["model"]["api_base"]
TASK_MODEL = cfg["model"]["task_lm"]
MAX_RETRIES = 3
EVAL_NUM_THREADS = cfg.get("eval_num_threads", 8)
THINKING_ENABLED = cfg["model"].get("task_thinking", cfg["model"].get("thinking", True))

# --- Initialize client and data once at import time ---
client = OpenAI(base_url=API_BASE, api_key=API_KEY)
train_set, _ = load_sciknoweval_raw(
    path=cfg["dataset"]["path"],
    train_n=cfg["dataset"]["train_n"],
    val_n=0,
    seed=cfg["dataset"]["seed"],
)
print(f"[evaluator] Loaded {len(train_set)} train examples, model={TASK_MODEL}, threads={EVAL_NUM_THREADS}")


def _call_llm(prompt_text: str) -> str:
    """Call the task LLM and return the response text."""
    for attempt in range(MAX_RETRIES):
        try:
            create_kwargs = dict(
                model=TASK_MODEL,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=0.7,
                max_tokens=8192,
            )
            if not THINKING_ENABLED:
                create_kwargs["extra_body"] = {"reasoning": {"enabled": False}}
            response = client.chat.completions.create(**create_kwargs)
            record_usage_to_file(response)
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"[evaluator] LLM call failed after {MAX_RETRIES} attempts: {e}")
                return ""
            time.sleep(2 ** attempt)
    return ""


def _extract_answer(response: str) -> str:
    """Extract the answer from the model response.

    Looks for text after "Answer:" (case-insensitive). If not found, returns
    the last line as a fallback.
    """
    match = re.search(r"(?i)answer\s*:\s*(.+)", response)
    if match:
        return match.group(1).strip()
    lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
    return lines[-1] if lines else response.strip()


def _score_one(prompt_template: str, ex: dict) -> float:
    """Score a single example. Returns 1.0 for correct, 0.0 for incorrect."""
    try:
        formatted = prompt_template.format(question=ex["question"])
    except (KeyError, ValueError, IndexError):
        return 0.0

    response = _call_llm(formatted)
    pred = _extract_answer(response)
    gold = ex["answer"].strip()
    task_type = ex["task_type"]

    if task_type.startswith("mcq"):
        # Extract single letter A-D
        for pattern in [r'^([A-Da-d])\b', r'\(([A-Da-d])\)', r'^([A-Da-d])\.']:
            m = re.search(pattern, pred)
            if m:
                pred = m.group(1).upper()
                break
        else:
            if pred and pred[0].upper() in "ABCD":
                pred = pred[0].upper()
        return 1.0 if pred == gold.upper() else 0.0
    else:
        # filling (balancing_chemical_equation): substring match
        return 1.0 if gold in pred else 0.0


def _score_prompt(prompt_template: str, examples: list[dict]) -> float:
    """Evaluate a prompt template on examples in parallel, return mean accuracy."""
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
            if done % 50 == 0 or done == n:
                avg = total_score / done
                print(f"[evaluator] sciknoweval: {done}/{n} examples, running acc={avg:.3f}")

    with ThreadPoolExecutor(max_workers=EVAL_NUM_THREADS) as pool:
        futures = [pool.submit(_score_one, prompt_template, ex) for ex in examples]
        for future in as_completed(futures):
            _on_complete(future.result())

    return total_score / n


def _validate_prompt_structure(prompt_template: str) -> None:
    """Reject structurally malformed prompts before scoring.

    Raises ValueError (which openevolve will log and discard) if the prompt is
    missing required delimiters/placeholders or would crash on str.format.
    """
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
    """Main entry point called by OpenEvolve for each candidate prompt."""
    with open(prompt_path) as f:
        prompt_template = f.read().strip()

    _validate_prompt_structure(prompt_template)

    score = _score_prompt(prompt_template, train_set)
    return {"combined_score": score}
