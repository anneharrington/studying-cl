"""Shared OpenEvolve runner logic.

Provides functions for single-task, sequential, append, allorders, and mixed
strategies. Individual scripts become thin wrappers around these functions.
"""

import itertools
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
import openevolve
from openai import OpenAI

from cl.tasks import get_openevolve_tasks, TASK_REGISTRY, get_system_prompt
from cl.utils.token_tracker import TokenTracker, usage_diff
from cl.utils.metrics_logger import log_stage, read_metrics_log
from cl.utils.failure_tracker import FailureTracker
from cl.utils.wandb_logger import WandbLogger
from cl.utils.plotting import plot_sequential, plot_allorders

# All 6 orderings of 3 tasks
ALL_ORDERINGS = [
    ("hotpotqa", "ifeval", "hover"),
    ("hotpotqa", "hover", "ifeval"),
    ("ifeval", "hotpotqa", "hover"),
    ("ifeval", "hover", "hotpotqa"),
    ("hover", "hotpotqa", "ifeval"),
    ("hover", "ifeval", "hotpotqa"),
]

from methods.prompt_based.runners.meta_prompt import get_meta_prompt

# Module-level tracker/logger, set by setup_run()
_failure_tracker = None
_wandb_logger = None
_thinking_enabled = True
_api_base = ""
_extra_body = None
_predictions_log_path = None


def _thinking_extra_body():
    """Provider-specific payload to disable qwen3 reasoning. Inferred from api_base.

    DashScope/Alibaba expects {"enable_thinking": False}; OpenRouter and other
    OpenAI-compatible passthroughs use {"reasoning": {"enabled": False}}.
    """
    base = (_api_base or "").lower()
    if "dashscope" in base or "aliyuncs" in base:
        return {"enable_thinking": False}
    return {"reasoning": {"enabled": False}}


def _build_extra_body():
    """Merge model-configured extra_body with the thinking-disable payload."""
    merged = {}
    if _extra_body:
        merged.update(_extra_body)
    if not _thinking_enabled:
        merged.update(_thinking_extra_body())
    return merged or None


# ---------------------------------------------------------------------------
#  Shared helpers
# ---------------------------------------------------------------------------

def strip_placeholders(text):
    """Remove `{name}` placeholder tokens from text.

    Used before prepending previous-task instructions in append mode, so
    task-specific placeholders (e.g. `{prompt}`, `{context}`) don't leak into
    the next task's general-instructions block and break .format() calls.
    """
    if not text:
        return text
    return re.sub(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}", "", text)


def clean_evolved_text(text):
    """Remove format placeholders and task-specific template parts from evolved text."""
    text = re.sub(r"\{[a-z_]+\}", "", text)
    text = re.sub(r'(?i).*provide your final answer.*after "Answer:".*', "", text)
    text = re.sub(r'(?i).*provide your final verdict.*after "Label:".*', "", text)
    text = re.sub(r"(?i)^Context:\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"(?i)^Question:\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"(?i)^Claim:\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_prompt(general_instructions, task_name):
    """Combine general instructions with a task's fixed template.

    Instructions are natural-language text (GEPA-evolved prompt or ACE playbook)
    and may legitimately contain `{...}` literals — JSON snippets in bullet
    examples, dict-shaped tool call demos, etc. Those would collide with the
    task template's `{question}`/`{prompt}`/... placeholders when
    `_safe_format` calls str.format(), so we escape them. Real placeholders
    in `template` are unaffected since `template` comes from TASK_REGISTRY,
    not from the evolved instructions.

    Failure mode this prevents: GPT-5.2 reflector wrote bullets like
    `... use {"query": "cute kitten"} ...` into ACE playbooks. Without
    escaping, every post-baseline eval call raised KeyError("query") inside
    _safe_format, which returned None and caused _score_and_extract to log
    empty predictions with metric=0 for the whole evaluation stage.
    """
    template = TASK_REGISTRY[task_name].get("template", "")
    safe_instructions = (general_instructions or "").replace("{", "{{").replace("}", "}}")
    return f"{safe_instructions}\n\n{template}"


def apply_thinking_to_oe_dict(oe_dict, cfg):
    """Controls the openevolve mutation/reflection LLM's thinking mode.

    Reads cfg['model']['reflection_thinking'] first, falling back to
    cfg['model']['thinking'] (default True). When disabled, appends '/no_think'
    to prompt.system_message — Qwen3 honors that directive to suppress its
    reasoning traces. This propagates across openevolve's worker-process pool
    because system_message is serialized into the config each worker rebuilds.
    Idempotent; mutates and returns oe_dict.
    """
    model_cfg = cfg.get("model", {})
    enabled = model_cfg.get("reflection_thinking", model_cfg.get("thinking", True))
    if enabled:
        return oe_dict
    prompt_section = oe_dict.setdefault("prompt", {})
    existing = prompt_section.get("system_message") or ""
    if "/no_think" in existing:
        return oe_dict
    prompt_section["system_message"] = (existing + "\n\n/no_think").strip()
    return oe_dict


def _call_llm(client, model, prompt_text, max_retries=3, system_text=None):
    """Call the LLM and return the response text.

    `system_text`, when provided, is sent as a separate role=system message
    before the user message. Tasks like FinQA require this to match the
    training-time prompt structure (otherwise Qwen3 sees no scaffold and the
    baseline is artificially low).
    """
    messages = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": prompt_text})
    for attempt in range(max_retries):
        try:
            create_kwargs = dict(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=8192,
            )
            _eb = _build_extra_body()
            if _eb is not None:
                create_kwargs["extra_body"] = _eb
            response = client.chat.completions.create(**create_kwargs)
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"    LLM call failed: {e}")
                if _failure_tracker:
                    _failure_tracker.record(error=str(e)[:200])
                return ""
            time.sleep(2 ** attempt)
    return ""


def _safe_unescape(s):
    """encode().decode('unicode_escape') but fall back to the raw string on failure.

    Truncated or invalid escape sequences (e.g. an incomplete \\uXXXX, common
    when an LLM truncates mid-token) raise UnicodeDecodeError otherwise, and
    that propagates up through the scoring pool and kills the run.
    """
    try:
        return s.encode().decode("unicode_escape")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return s


def _ace_final_answer(response):
    """If response is ACE-shaped JSON, return its final_answer. Else return response unchanged.

    ACE's generator emits {"reasoning": ..., "bullet_ids": [...], "final_answer": "..."}.
    This helper lets harness extractors grade ACE output and native output with the
    same logic: unwrap the JSON field when present, pass through otherwise.
    """
    if not response:
        return response
    m = re.search(r'"final_answer"\s*:\s*"((?:[^"\\]|\\.)*)"', response)
    if m:
        return _safe_unescape(m.group(1))
    m = re.search(r"'final_answer'\s*:\s*'((?:[^'\\]|\\.)*)'", response)
    if m:
        return _safe_unescape(m.group(1))
    return response


def _extract_answer(response):
    """Extract answer after 'Answer:' marker."""
    response = _ace_final_answer(response)
    match = re.search(r"(?i)answer\s*:\s*(.+)", response)
    if match:
        return match.group(1).strip()
    lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
    return lines[-1] if lines else response.strip()


def _extract_label(response):
    """Extract SUPPORTED/NOT_SUPPORTED label."""
    response = _ace_final_answer(response)
    match = re.search(r"(?i)label\s*:\s*(.+)", response)
    raw = match.group(1).strip().upper() if match else response.strip().upper()
    if "NOT" in raw and "SUPPORT" in raw:
        return "NOT_SUPPORTED"
    elif "SUPPORT" in raw:
        return "SUPPORTED"
    return raw


def _extract_mcq_answer(text):
    """Extract MCQ letter answer (A-D)."""
    text = _ace_final_answer(text)
    text = text.strip()
    for pattern in [r'^([A-Da-d])\b', r'\(([A-Da-d])\)', r'^([A-Da-d])\.']:
        m = re.search(pattern, text)
        if m:
            return m.group(1).upper()
    if text and text[0].upper() in "ABCD":
        return text[0].upper()
    return text.strip().upper()


# Dedupe formatting-failure warnings so a broken template logs once per (task, error)
_format_failures_seen = set()


def _safe_format(prompt_template, task_name, **kwargs):
    """Format prompt_template with kwargs; on failure, log loudly and return None.

    Returns the formatted string, or None if formatting failed (caller should
    score the example as 0.0). Failures are logged once per (task, error-key)
    via print + _failure_tracker so silent template breakage is visible.
    """
    try:
        return prompt_template.format(**kwargs)
    except (KeyError, ValueError, IndexError) as e:
        # Identify the offending placeholder for the dedupe key
        if isinstance(e, KeyError):
            err_key = f"KeyError:{e.args[0] if e.args else '?'}"
        else:
            err_key = f"{type(e).__name__}:{str(e)[:80]}"
        seen_key = (task_name, err_key)
        if seen_key not in _format_failures_seen:
            _format_failures_seen.add(seen_key)
            # Find leaked placeholders in the template that weren't supplied
            leaked = re.findall(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}", prompt_template)
            unmatched = [p for p in leaked if p[1:-1] not in kwargs]
            print(
                f"    [FORMAT WARN] {task_name}: {err_key}. "
                f"Unmatched placeholders in template: {unmatched[:5]}. "
                f"All examples affected by this template will score 0.0."
            )
            if _failure_tracker:
                _failure_tracker.record(error=f"format_{task_name}:{err_key}")
        return None


def _score_and_extract(ex, task_name, prompt_template, client, model, seed_mode="default"):
    """Score a single example; return (score, raw_response, extracted_answer).

    Used by both `_score_example` (float-only return, back-compat) and the
    per-example predictions logger path. Extracted answer is a short string
    representation of what the grader compared against the gold.

    `seed_mode` selects which TASK_REGISTRY system prompt to send for tasks
    that have a `system` field (finqa, sciknoweval_bio). "default" sends the
    standard system; "vanilla" sends `vanilla_system` (typically barebones)
    for the ablation experiment.
    """
    if task_name == "hotpotqa":
        from cl.evals.hotpot_evaluate_v1 import f1_score
        formatted = _safe_format(prompt_template, task_name, context=ex["context"], question=ex["question"])
        if formatted is None:
            return 0.0, "", ""
        response = _call_llm(client, model, formatted)
        answer = _extract_answer(response)
        f1, _, _ = f1_score(answer, ex["answer"])
        return f1, response, answer

    elif task_name == "ifeval":
        from cl.evals.ifeval_lib.evaluation_lib import InputExample, test_instruction_following_strict
        formatted = _safe_format(prompt_template, task_name, prompt=ex["prompt"])
        if formatted is None:
            return 0.0, "", ""
        response = _call_llm(client, model, formatted)
        inp = InputExample(
            key=ex["key"],
            instruction_id_list=ex["instruction_id_list"],
            prompt=ex["prompt"],
            kwargs=ex["kwargs"],
        )
        output = test_instruction_following_strict(inp, {ex["prompt"]: response})
        n_total = len(output.follow_instruction_list)
        n_followed = sum(output.follow_instruction_list)
        score = n_followed / n_total if n_total > 0 else 0.0
        return score, response, f"{n_followed}/{n_total} constraints followed"

    elif task_name == "hover":
        formatted = _safe_format(prompt_template, task_name, claim=ex["claim"])
        if formatted is None:
            return 0.0, "", ""
        response = _call_llm(client, model, formatted)
        pred = _extract_label(response)
        return (1.0 if pred == ex["label"] else 0.0), response, pred

    elif task_name == "sciknoweval":
        formatted = _safe_format(prompt_template, task_name, question=ex["question"])
        if formatted is None:
            return 0.0, "", ""
        response = _call_llm(client, model, formatted)
        pred = _extract_answer(response)
        gold = ex["answer"].strip()
        task_type = ex["task_type"]
        if task_type.startswith("mcq"):
            pred = _extract_mcq_answer(pred)
            return (1.0 if pred == gold.upper() else 0.0), response, pred
        else:
            return (1.0 if gold in pred else 0.0), response, pred

    elif task_name == "gsm8k":
        from cl.evals.gsm8k import _extract_predicted_number, _normalize_number
        formatted = _safe_format(prompt_template, task_name, question=ex["question"])
        if formatted is None:
            return 0.0, "", ""
        response = _call_llm(client, model, formatted)
        pred = _normalize_number(_extract_predicted_number(response))
        gold = _normalize_number(ex["answer"])
        return (1.0 if pred == gold else 0.0), response, str(pred)

    elif task_name == "livebench_math":
        from cl.evals.livebench_math import _extract_three_digit_answer, _extract_mcq_answer as _extract_mcq_lb
        formatted = _safe_format(prompt_template, task_name, question=ex["question"])
        if formatted is None:
            return 0.0, "", ""
        response = _call_llm(client, model, formatted)
        gold = ex["answer"].strip()
        if ex.get("answer_type") == "mcq":
            pred = _extract_mcq_lb(response)
            return (1.0 if pred == gold.upper() else 0.0), response, pred
        else:
            pred = _extract_three_digit_answer(response)
            return (1.0 if pred == gold.zfill(3) else 0.0), response, pred

    elif task_name == "sciknoweval_bio":
        from cl.evals.sciknoweval_bio import _extract_mcq_answer as _extract_mcq_bio
        formatted = _safe_format(prompt_template, task_name, question=ex["question"])
        if formatted is None:
            return 0.0, "", ""
        system_text = get_system_prompt(task_name, mode=seed_mode)
        response = _call_llm(client, model, formatted, system_text=system_text)
        # Pass the raw response — _extract_mcq_bio handles training XML
        # (<answer>X</answer>), legacy `Answer: X` lines, and bare-letter
        # outputs uniformly.
        pred = _extract_mcq_bio(response)
        gold = ex["answer"].strip().upper()
        return (1.0 if pred == gold else 0.0), response, pred

    elif task_name == "tooluse":
        from cl.evals.toolalpaca import _parse_actions, _score_actions
        formatted = _safe_format(prompt_template, task_name, question=ex["question"])
        if formatted is None:
            return 0.0, "", ""
        response = _call_llm(client, model, formatted)
        pred_actions = _parse_actions(response)
        score, _ = _score_actions(pred_actions, ex["golden_steps"])
        return score, response, str(pred_actions)[:400]

    elif task_name == "finqa":
        from cl.evals.finqa import _extract_predicted_number, _numbers_match
        formatted = _safe_format(prompt_template, task_name, question=ex["question"])
        if formatted is None:
            return 0.0, "", ""
        system_text = get_system_prompt(task_name, mode=seed_mode)
        response = _call_llm(client, model, formatted, system_text=system_text)
        pred = _extract_predicted_number(response)
        return (1.0 if _numbers_match(pred, ex["answer"]) else 0.0), response, str(pred)

    elif task_name == "toolalpaca":
        from cl.evals.toolalpaca import _parse_actions, _score_actions
        formatted = _safe_format(prompt_template, task_name, question=ex["question"])
        if formatted is None:
            return 0.0, "", ""
        response = _call_llm(client, model, formatted)
        pred_actions = _parse_actions(response)
        score, _ = _score_actions(pred_actions, ex["golden_steps"])
        return score, response, str(pred_actions)[:400]

    elif task_name.startswith("finance_yr_"):
        # Per-year SDPO finance bundle: 10-K excerpt -> up/down. The template
        # registered in tasks.py ("{filing_text}\n\nReturn one token...")
        # already carries the format hint, so we just substitute filing_text.
        # Unwrap ACE-shaped JSON final_answer if present so ACE responses and
        # plain-text responses route through the same regex extractor.
        from cl.evals.finance_yr import _parse_label
        formatted = _safe_format(prompt_template, task_name, filing_text=ex["filing_text"])
        if formatted is None:
            return 0.0, "", ""
        response = _call_llm(client, model, formatted)
        candidate = _ace_final_answer(response) or response
        pred = _parse_label(candidate)
        gold = (ex["answer"] or "").strip().lower()
        ok = pred == gold and pred in ("up", "down")
        return (1.0 if ok else 0.0), response, str(pred)

    elif task_name.startswith("temporalwiki_"):
        # Drift slices (temporalwiki_drift_s<i>) and the eval-only stable
        # probe (temporalwiki_stable). All use the same F1>=0.5 metric the
        # SDPO bundle reports as val-core/<source>/acc/mean@N. Template is
        # bare "{question}".
        from cl.evals.temporalwiki import _extract_answer as _tw_extract, _f1, ACC_THRESHOLD
        formatted = _safe_format(prompt_template, task_name, question=ex["question"])
        if formatted is None:
            return 0.0, "", ""
        response = _call_llm(client, model, formatted)
        candidate = _ace_final_answer(response) or response
        pred = _tw_extract(candidate)
        f1 = _f1(pred, ex["answer"] or "")
        return (1.0 if f1 >= ACC_THRESHOLD else 0.0), response, pred

    return 0.0, "", ""


def _score_example(ex, task_name, prompt_template, client, model,
                   predictions_log_path=None, stage=None, index=None,
                   general_instructions=None, seed_mode="default"):
    """Score a single example. Returns a float score (0-1 scale).

    When `predictions_log_path` is set, also appends a per-example record
    via cl.utils.predictions_logger.
    """
    score, response, extracted = _score_and_extract(
        ex, task_name, prompt_template, client, model, seed_mode=seed_mode
    )
    if predictions_log_path:
        from cl.utils.predictions_logger import log_prediction, render_question, render_gold
        log_prediction(
            path=predictions_log_path,
            stage=stage,
            task=task_name,
            question=render_question(task_name, ex),
            llm_response=response,
            extracted=extracted,
            gold=render_gold(task_name, ex),
            metric=score,
            index=index,
            general_instructions=general_instructions,
        )
    return score


def score_on_task(general_instructions, task_name, examples, client, model, num_threads=8,
                  predictions_log_path=None, stage=None, seed_mode="default"):
    """Score general instructions on a specific task's examples in parallel.

    Returns score as a percentage (0-100). When `predictions_log_path` is set
    (or `logging.predictions_log: true` is in the run config), each example is
    also logged via cl.utils.predictions_logger, tagged with `stage`.

    `seed_mode` controls which TASK_REGISTRY system prompt is sent for tasks
    with a `system` field: "default" (current behavior) or "vanilla"
    (barebones ablation; see cl.tasks.get_system_prompt).
    """
    if not examples:
        return 0.0

    if predictions_log_path is None:
        predictions_log_path = _predictions_log_path

    prompt_template = build_prompt(general_instructions, task_name)
    total = 0.0

    n_failed = 0
    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = [
            pool.submit(
                _score_example, ex, task_name, prompt_template, client, model,
                predictions_log_path=predictions_log_path, stage=stage, index=i,
                general_instructions=general_instructions, seed_mode=seed_mode,
            )
            for i, ex in enumerate(examples)
        ]
        for future in as_completed(futures):
            try:
                score = future.result()
            except Exception as e:
                # An unhandled exception in _score_example would otherwise
                # propagate up through the runner and kill the whole run.
                # Score the example as 0 and continue. The first such failure
                # in this batch logs a traceback so the cause is visible.
                if n_failed == 0:
                    import traceback
                    print(f"    [{task_name}] _score_example raised "
                          f"{type(e).__name__}: {e}")
                    traceback.print_exc()
                n_failed += 1
                score = 0.0
            total += score
            if _wandb_logger:
                _wandb_logger.record_example(task_name, score * 100, phase="eval")

    if n_failed:
        print(f"    [{task_name}] {n_failed}/{len(examples)} examples crashed during scoring (counted as 0)")

    return total / len(examples) * 100


def eval_all_tasks(general_instructions, task_data, client, model, num_threads=8, stage=None,
                   predictions_log_path=None):
    """Evaluate general instructions on all tasks in parallel."""
    def _eval_one(task_name, td):
        score = score_on_task(
            general_instructions, task_name, td.get("eval_set", td.get("val_set")),
            client, model, num_threads=num_threads,
            predictions_log_path=predictions_log_path, stage=stage,
        )
        print(f"    {task_name}: {score:.2f}")
        if _wandb_logger and stage:
            _wandb_logger.flush_task(task_name, phase="eval")
            _wandb_logger.log_task_score(task_name, score, stage)
        return task_name, score

    scores = {}
    with ThreadPoolExecutor(max_workers=len(task_data)) as pool:
        futures = {pool.submit(_eval_one, name, td): name for name, td in task_data.items()}
        for future in as_completed(futures):
            failed_name = futures[future]
            try:
                task_name, score = future.result()
                scores[task_name] = score
            except Exception as e:
                import traceback
                print(f"    eval_all_tasks failed on {failed_name}: {type(e).__name__}: {e}")
                traceback.print_exc()
                scores[failed_name] = 0.0

    if _wandb_logger and stage:
        _wandb_logger.log_stage_scores(stage, scores)

    return scores


def write_task_config(task_cfg, model_cfg, output_path, eval_num_threads=8):
    """Write a temporary task-specific config for an OpenEvolve evaluator."""
    config = {
        "dataset": task_cfg["dataset"],
        "model": model_cfg,
        "eval_num_threads": eval_num_threads,
    }
    with open(output_path, "w") as f:
        yaml.dump(config, f)
    return str(output_path)


def load_all_datasets_raw(cfg, tasks_info, drop_val=False):
    """Load datasets for all tasks using raw loaders. Returns task_data dict.

    When `drop_val=True` (the OpenEvolve/OpenEvolve-v2/v2-meta/v2-meta-single
    path), val_n is forced to 0 and a train/eval-only split is produced. The
    same `eval_set` is used for baseline, optimized, and cross-task evaluation
    — so "baseline X" and "after_X" are scored on identical examples, making
    every row of the cross-task matrix directly comparable. eval_n defaults
    to 100 if not specified in the task config.

    ACE / GEPA keep the three-way split (drop_val=False) — they actually use
    val_set during optimization (ACE picks best playbook on val; GEPA uses
    val as its fitness signal).
    """
    task_data = {}
    for task_cfg in cfg["tasks"]:
        name = task_cfg["name"]
        ds = task_cfg["dataset"]
        loader_raw = tasks_info[name]["loader_raw"]
        print(f"Loading {name} dataset from {ds['path']}...")
        # Optional pass-throughs for loaders that scope each task entry by
        # year/slice or truncate context. Mirrors gepa_runner.load_all_datasets.
        # year_filter:        sentiment10k_<year>, finance_yr_<year>, sealqa_<year>
        # slice_filter:       temporalwiki_drift_<s>, temporalwiki_stable
        # max_context_chars:  finance_yr (50k cap), sentiment10k
        extra_kwargs = {}
        for opt_key in ("year_filter", "max_context_chars", "slice_filter"):
            if opt_key in ds:
                extra_kwargs[opt_key] = ds[opt_key]
        if drop_val:
            eval_n = ds.get("eval_n", 100) or 100
            splits = loader_raw(
                path=ds["path"],
                train_n=ds["train_n"],
                val_n=0,
                seed=ds["seed"],
                eval_n=eval_n,
                **extra_kwargs,
            )
            # splits = (train_set, empty_val, eval_set). Alias val_set → eval_set
            # so any legacy accessor that still reads .["val_set"] gets the
            # correct data instead of an empty list.
            train_set, _, eval_set = splits
            td = {"train_set": train_set, "eval_set": eval_set, "val_set": eval_set}
            print(f"  {name}: {len(train_set)} train, {len(eval_set)} eval (no val)")
        else:
            eval_n = ds.get("eval_n", 0)
            splits = loader_raw(
                path=ds["path"],
                train_n=ds["train_n"],
                val_n=ds["val_n"],
                seed=ds["seed"],
                eval_n=eval_n,
                **extra_kwargs,
            )
            train_set, val_set = splits[0], splits[1]
            td = {"train_set": train_set, "val_set": val_set}
            size_str = f"  {name}: {len(train_set)} train, {len(val_set)} val"
            if len(splits) == 3:
                td["eval_set"] = splits[2]
                size_str += f", {len(splits[2])} eval"
            print(size_str)
        task_data[name] = td
    return task_data


def setup_run(cfg):
    """Common setup. Returns (client, model, tracker, failure_tracker, wlog, output_dir, metrics_log_path, eval_num_threads)."""
    global _failure_tracker, _wandb_logger
    from cl.config import ensure_unique_output_dir
    ensure_unique_output_dir(cfg)

    api_key_env = cfg["model"].get("api_key_env", "PORTKEY_API_KEY")
    # SageMaker routes use AWS auth via boto3, not an API key.
    # Portkey routes use PORTKEY_API_KEY via the Portkey SDK directly, so the
    # existing api_key_env check still applies (default api_key_env is
    # PORTKEY_API_KEY anyway).
    is_sagemaker = str(cfg["model"]["task_lm"]).startswith("sagemaker:")
    is_portkey = str(cfg["model"]["task_lm"]).startswith("portkey:")
    is_bedrock = str(cfg["model"]["task_lm"]).startswith("bedrock:")
    if is_sagemaker or is_bedrock:
        api_key = None  # unused; the shim authenticates via boto3 / bearer token
    else:
        if api_key_env not in os.environ:
            print(f"Error: {api_key_env} environment variable not set")
            sys.exit(1)
        api_key = os.environ[api_key_env]
        os.environ["OPENAI_API_KEY"] = api_key

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_log_enabled = cfg.get("logging", {}).get("detailed_metrics_log", False)
    metrics_log_path = str(output_dir / "metrics_log.jsonl") if metrics_log_enabled else None
    if metrics_log_enabled:
        print(f"  Metrics logging enabled: {metrics_log_path}")

    predictions_log_enabled = cfg.get("logging", {}).get("predictions_log", True)
    global _predictions_log_path
    _predictions_log_path = str(output_dir / "predictions.jsonl") if predictions_log_enabled else None
    if predictions_log_enabled:
        print(f"  Predictions logging enabled: {_predictions_log_path}")

    _failure_tracker = FailureTracker()
    _wandb_logger = WandbLogger.from_config(cfg)

    global _thinking_enabled, _api_base, _extra_body
    _thinking_enabled = cfg["model"].get(
        "task_thinking", cfg["model"].get("thinking", True)
    )
    _api_base = cfg["model"].get("api_base", "")
    _extra_body = cfg["model"].get("extra_body")

    if is_sagemaker:
        from methods.prompt_based.sagemaker_lm import SageMakerOpenAIShim
        endpoint = cfg["model"]["task_lm"].split(":", 1)[1]
        region = cfg["model"].get(
            "aws_region_name", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        )
        client = SageMakerOpenAIShim(
            endpoint_name=endpoint,
            region_name=region,
            no_think=(_thinking_enabled is False),
        )
    elif is_portkey:
        from methods.prompt_based.portkey_lm import PortkeyOpenAIShim
        route = cfg["model"]["task_lm"].split(":", 1)[1]
        client = PortkeyOpenAIShim(
            route=route,
            no_think=(_thinking_enabled is False),
        )
    elif is_bedrock:
        from methods.prompt_based.bedrock_lm import BedrockOpenAIShim
        model_id = cfg["model"]["task_lm"].split(":", 1)[1]
        region = cfg["model"].get(
            "aws_region_name", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        )
        client = BedrockOpenAIShim(
            model_id=model_id,
            region_name=region,
            no_think=(_thinking_enabled is False),
        )
    else:
        client = OpenAI(base_url=cfg["model"]["api_base"], api_key=api_key)
    model = cfg["model"]["task_lm"]
    eval_num_threads = cfg.get("eval_num_threads", 8)

    tracker = TokenTracker()

    return client, model, tracker, _failure_tracker, _wandb_logger, output_dir, metrics_log_path, eval_num_threads


def print_results_table(stages, all_scores, task_names):
    """Print a formatted results table."""
    header = f"{'Stage':<20}" + "".join(f" {n:>12}" for n in task_names)
    print(f"\n{header}")
    print("-" * (20 + 13 * len(task_names)))
    for i, stage in enumerate(stages):
        row = f"{stage:<20}"
        for name in task_names:
            row += f" {all_scores[name][i]:>12.2f}"
        print(row)


# ---------------------------------------------------------------------------
#  Single-task runner
# ---------------------------------------------------------------------------

def run_single(cfg, task_name=None):
    """Run OpenEvolve on a single task."""
    from cl.config import ensure_unique_output_dir

    start_time = time.time()

    if task_name is None:
        task_name = cfg.get("task_name", Path(cfg["output_dir"]).name.replace("openevolve_", ""))

    ensure_unique_output_dir(cfg)

    tasks_info = get_openevolve_tasks([task_name])
    task_info = tasks_info[task_name]

    client, model, tracker, failure_tracker, wlog, output_dir, metrics_log_path, eval_num_threads = setup_run(cfg)

    ds = cfg["dataset"]
    # OpenEvolve uses a 2-split layout: train drives evolution, eval is used
    # for BOTH the baseline read and the optimized-prompt score, so the two
    # numbers are directly comparable. val_n is forced to 0 regardless of
    # what the task config says.
    eval_n = ds.get("eval_n", 100) or 100
    print(f"Loading dataset from {ds['path']}...")
    splits = task_info["loader_raw"](
        path=ds["path"],
        train_n=ds["train_n"],
        val_n=0,
        seed=ds["seed"],
        eval_n=eval_n,
    )
    eval_set = splits[2]
    val_set = eval_set  # alias — see load_all_datasets_raw docstring

    task_config_path = write_task_config(
        {"dataset": ds},
        cfg["model"],
        output_dir / f"config_{task_name}.yaml",
        eval_num_threads=eval_num_threads,
    )
    os.environ["BENCHMARK_CONFIG"] = task_config_path

    with open(task_info["initial_prompt"]) as f:
        baseline_prompt = f.read().strip()

    skip_baseline = cfg.get("skip_baseline", False)

    with tracker.track_to_file():
        baseline_score = None
        if not skip_baseline:
            print("\n--- Baseline evaluation (unoptimized) ---")
            default_instruction = task_info["default_instruction"]
            baseline_score = score_on_task(default_instruction, task_name, eval_set, client, model, eval_num_threads,
                                           stage="baseline")
            print(f"Baseline eval Score: {baseline_score:.2f}")
        else:
            print("\nSkipping baseline evaluation")

        oe_config = openevolve.Config.from_dict(apply_thinking_to_oe_dict(cfg["openevolve"], cfg))

        initial_program_path = cfg.get("_initial_prompt_override") or task_info["initial_prompt"]

        print("\n--- Running OpenEvolve optimization ---")
        result = openevolve.run_evolution(
            initial_program=initial_program_path,
            evaluator=task_info["evaluator"],
            config=oe_config,
            output_dir=str(output_dir),
        )
        print(f"Evolution complete. Best train score: {result.best_score:.4f}")

        best_prompt = result.best_code
        print("\n--- Optimized evaluation ---")
        # For single-task, evaluate the raw evolved prompt directly
        optimized_score = _evaluate_raw_prompt_on_val(best_prompt, task_name, eval_set, client, model, eval_num_threads,
                                                      stage="optimized")
        print(f"Optimized eval Score: {optimized_score:.2f}")

    token_usage = tracker.get_usage()

    print(f"\n{'='*40}")
    if baseline_score is not None:
        print(f"Baseline eval Score:  {baseline_score:.2f}")
    print(f"Optimized eval Score: {optimized_score:.2f}")
    if baseline_score is not None:
        print(f"Improvement:         {optimized_score - baseline_score:+.2f}")
    print(f"{'='*40}")

    elapsed = time.time() - start_time
    print(f"\nTotal runtime: {elapsed / 60:.1f} minutes ({elapsed:.0f}s)")

    results = {
        "runtime_seconds": round(elapsed, 1),
        "runtime_hours": round(elapsed / 3600, 2),
        "baseline_score": baseline_score,
        "optimized_score": optimized_score,
        "best_train_score": result.best_score,
        "best_prompt": best_prompt,
        "usage": token_usage,
        "config": cfg,
    }
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    prompt_path = output_dir / "best_prompt.txt"
    with open(prompt_path, "w") as f:
        f.write(best_prompt)
    print(f"Best prompt saved to {prompt_path}")


def _evaluate_raw_prompt_on_val(prompt_template, task_name, examples, client, model, num_threads,
                                stage=None):
    """Evaluate a raw prompt template (with placeholders) on val set. Returns percentage."""
    if not examples:
        return 0.0

    predictions_log_path = _predictions_log_path
    total = 0.0
    n_failed = 0
    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = [
            pool.submit(
                _score_example, ex, task_name, prompt_template, client, model,
                predictions_log_path=predictions_log_path, stage=stage, index=i,
            )
            for i, ex in enumerate(examples)
        ]
        for future in as_completed(futures):
            try:
                total += future.result()
            except Exception as e:
                if n_failed == 0:
                    import traceback
                    print(f"    [{task_name}] _score_example raised "
                          f"{type(e).__name__}: {e}")
                    traceback.print_exc()
                n_failed += 1

    if n_failed:
        print(f"    [{task_name}] {n_failed}/{len(examples)} examples crashed during scoring (counted as 0)")

    return total / len(examples) * 100


# ---------------------------------------------------------------------------
#  Sequential runner (replace + append)
# ---------------------------------------------------------------------------

def _run_sequential_ordering(ordering, task_data, task_cfgs, client, model, cfg,
                             output_dir, metrics_log_path, tracker, strategy="replace",
                             eval_num_threads=8, order_label=None):
    """Run one sequential ordering. Returns results dict.

    order_label: e.g. "order_1" — used to namespace files so orderings don't collide.
    """
    tasks_info = get_openevolve_tasks(list(task_data.keys()))
    task_names = list(task_data.keys())

    # Create per-ordering subdirectory to avoid file collisions
    if order_label:
        ordering_dir = output_dir / f"{order_label}_{'_'.join(ordering)}"
    else:
        ordering_dir = output_dir
    ordering_dir.mkdir(parents=True, exist_ok=True)
    skip_baseline = cfg.get("skip_baseline", False)

    stages = [] if skip_baseline else ["baseline"]
    stages += [f"after_{name}" for name in ordering]
    all_scores = {name: [] for name in task_names}
    all_instructions = {}
    all_prompts = {}

    _last_usage = tracker.get_usage()

    # --- BASELINE ---
    if not skip_baseline:
        print(f"\n  {'─'*50}")
        print(f"  Baseline (default instructions)")
        print(f"  {'─'*50}")

        baseline_scores = {}
        for tn in task_names:
            default_instruction = TASK_REGISTRY[tn].get("default_instruction", "")
            print(f"    Evaluating {tn} with its default instructions...")
            score = score_on_task(default_instruction, tn, task_data[tn]["eval_set"],
                                 client, model, num_threads=eval_num_threads,
                                 stage="baseline")
            baseline_scores[tn] = score
            print(f"    {tn}: {score:.2f}")

        for name in task_names:
            all_scores[name].append(baseline_scores[name])
        all_instructions["baseline"] = "(task-specific defaults)"
        _last_usage = tracker.get_usage()
    else:
        print("  Skipping baseline evaluation")

    # --- SEQUENTIAL OPTIMIZATION ---
    current_general_instructions = None
    previous_cleaned = None

    for idx, task_name in enumerate(ordering):
        if _failure_tracker:
            _failure_tracker.set_stage(f"after_{task_name}")

        print(f"\n  {'─'*50}")
        print(f"  Optimize on {task_name}")
        print(f"  {'─'*50}")

        task_default = TASK_REGISTRY[task_name].get("default_instruction", "")

        if strategy == "append":
            if idx == 0:
                general = task_default
            else:
                general = (
                    f"{task_default}\n\n"
                    f"Previous task optimized instructions:\n{strip_placeholders(previous_cleaned)}"
                )
        else:  # replace
            if current_general_instructions is None:
                general = task_default
            else:
                general = current_general_instructions

        initial_prompt_text = build_prompt(general, task_name)

        prompt_file = ordering_dir / f"initial_prompt_{task_name}.txt"
        with open(prompt_file, "w") as f:
            f.write(initial_prompt_text)

        # Find the right task_cfg for this task
        task_cfg = next(tc for tc in cfg["tasks"] if tc["name"] == task_name)
        task_config_path = write_task_config(task_cfg, cfg["model"],
                                             ordering_dir / f"config_{task_name}.yaml",
                                             eval_num_threads=eval_num_threads)
        os.environ["BENCHMARK_CONFIG"] = task_config_path

        oe_config = openevolve.Config.from_dict(apply_thinking_to_oe_dict(cfg["openevolve"], cfg))

        print(f"  Running OpenEvolve on {task_name}...")
        stage_output_dir = str(ordering_dir / f"evolution_{task_name}")
        result = openevolve.run_evolution(
            initial_program=str(prompt_file),
            evaluator=tasks_info[task_name]["evaluator"],
            config=oe_config,
            output_dir=stage_output_dir,
        )
        print(f"  Evolution complete. Best train score: {result.best_score:.4f}")

        evolved_text = result.best_code
        all_prompts[f"after_{task_name}"] = evolved_text
        current_general_instructions = clean_evolved_text(evolved_text)
        previous_cleaned = current_general_instructions
        all_instructions[f"after_{task_name}"] = current_general_instructions
        print(f"  Cleaned instructions ({len(current_general_instructions)} chars)")

        _pre_eval_usage = tracker.get_usage()

        print(f"\n  Evaluating on all tasks after {task_name} optimization:")
        stage_scores = eval_all_tasks(current_general_instructions, task_data, client, model,
                                      num_threads=eval_num_threads, stage=f"after_{task_name}")
        for name in task_names:
            all_scores[name].append(stage_scores[name])

        if metrics_log_path:
            _post_eval_usage = tracker.get_usage()
            log_stage(
                metrics_log_path, f"after_{task_name}", "openevolve",
                stage_scores, _post_eval_usage,
                optimization_usage=usage_diff(_pre_eval_usage, _last_usage),
                eval_usage=usage_diff(_post_eval_usage, _pre_eval_usage),
            )
            _last_usage = _post_eval_usage

    return {
        "ordering": list(ordering),
        "stages": stages,
        "scores": all_scores,
        "instructions": all_instructions,
        "evolved_prompts": all_prompts,
    }


def run_sequential(cfg, strategy="replace"):
    """Run sequential optimization."""
    start_time = time.time()
    task_names = [t["name"] for t in cfg["tasks"]]

    client, model, tracker, failure_tracker, wlog, output_dir, metrics_log_path, eval_num_threads = setup_run(cfg)
    task_data = load_all_datasets_raw(cfg, get_openevolve_tasks(task_names), drop_val=True)
    task_cfgs = {tc["name"]: tc for tc in cfg["tasks"]}

    with tracker.track_to_file():
        result = _run_sequential_ordering(
            ordering=task_names,
            task_data=task_data,
            task_cfgs=task_cfgs,
            client=client,
            model=model,
            cfg=cfg,
            output_dir=output_dir,
            metrics_log_path=metrics_log_path,
            tracker=tracker,
            strategy=strategy,
            eval_num_threads=eval_num_threads,
        )

    elapsed = time.time() - start_time
    print_results_table(result["stages"], result["scores"], task_names)

    token_usage = tracker.get_usage()
    failure_summary = failure_tracker.summary() if failure_tracker else {"total": 0}

    print(f"\nToken usage: {json.dumps(token_usage, indent=2)}")
    if failure_summary["total"] > 0:
        print(f"LM failures: {json.dumps(failure_summary, indent=2)}")
    print(f"\nTotal runtime: {elapsed / 60:.1f} minutes ({elapsed:.0f}s)")

    results = {
        "runtime_seconds": round(elapsed, 1),
        "runtime_hours": round(elapsed / 3600, 2),
        "stages": result["stages"],
        "scores": result["scores"],
        "instructions": result["instructions"],
        "evolved_prompts": result.get("evolved_prompts", {}),
        "usage": token_usage,
        "lm_failures": failure_summary,
        "config": cfg,
    }
    if metrics_log_path:
        results["metrics_log"] = read_metrics_log(metrics_log_path)

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    plot_path = output_dir / "sequential_performance.png"
    plot_sequential(results, str(plot_path))

    if wlog:
        wlog.log_usage(token_usage)
        wlog.log_failures(failure_summary)
        wlog.finish()


def run_allorders(cfg, strategy="replace", ordering_indices=None):
    """Run sequential optimization across all task orderings.

    Generates all permutations of the task list from the config.
    """
    start_time = time.time()
    task_names = [t["name"] for t in cfg["tasks"]]

    all_orderings = list(itertools.permutations(task_names))
    n_orderings = len(all_orderings)

    if ordering_indices is None:
        ordering_indices = list(range(1, n_orderings + 1))
    else:
        for idx in ordering_indices:
            if idx < 1 or idx > n_orderings:
                print(f"Error: ordering index {idx} out of range (1-{n_orderings})")
                sys.exit(1)

    client, model, tracker, failure_tracker, wlog, output_dir, metrics_log_path, eval_num_threads = setup_run(cfg)
    task_data = load_all_datasets_raw(cfg, get_openevolve_tasks(task_names), drop_val=True)
    task_cfgs = {tc["name"]: tc for tc in cfg["tasks"]}

    orderings_to_run = [(i, all_orderings[i - 1]) for i in ordering_indices]
    print(f"\nRunning {len(orderings_to_run)}/{n_orderings} ordering(s): {[i for i, _ in orderings_to_run]}")

    all_order_results = {}

    with tracker.track_to_file():
        for i, ordering in orderings_to_run:
            order_label = f"order_{i}"
            order_str = " → ".join(ordering)
            print(f"\n{'='*60}")
            print(f"ORDER {i}/{n_orderings}: {order_str}")
            print(f"{'='*60}")

            result = _run_sequential_ordering(
                ordering=ordering,
                task_data=task_data,
                task_cfgs=task_cfgs,
                client=client,
                model=model,
                cfg=cfg,
                output_dir=output_dir,
                metrics_log_path=metrics_log_path,
                tracker=tracker,
                strategy=strategy,
                eval_num_threads=eval_num_threads,
                order_label=order_label,
            )
            all_order_results[order_label] = result

            # Save per-ordering results (files are already in ordering_dir)
            order_dir = output_dir / f"{order_label}_{'_'.join(ordering)}"
            with open(order_dir / "results.json", "w") as f:
                json.dump(result, f, indent=2)
            print(f"  Order {i} results saved to {order_dir}/results.json")

    elapsed = time.time() - start_time
    token_usage = tracker.get_usage()
    failure_summary = failure_tracker.summary() if failure_tracker else {"total": 0}

    # Print combined results
    print(f"\n{'='*60}")
    print("COMBINED RESULTS")
    print(f"{'='*60}")
    for i, ordering in orderings_to_run:
        result = all_order_results[f"order_{i}"]
        print(f"\nOrder {i}: {' → '.join(ordering)}")
        print_results_table(result["stages"], result["scores"], task_names)

    print(f"\nToken usage: {json.dumps(token_usage, indent=2)}")
    print(f"\nTotal runtime: {elapsed / 60:.1f} minutes ({elapsed:.0f}s)")

    results = {
        "runtime_seconds": round(elapsed, 1),
        "runtime_hours": round(elapsed / 3600, 2),
        "orderings": all_order_results,
        "usage": token_usage,
        "lm_failures": failure_summary,
        "config": cfg,
    }
    if metrics_log_path:
        results["metrics_log"] = read_metrics_log(metrics_log_path)

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    if len(orderings_to_run) == n_orderings:
        plot_path = output_dir / "allorders_performance.png"
        plot_allorders(results["orderings"], str(plot_path))
    else:
        print(f"\nRan {len(orderings_to_run)}/{n_orderings} orderings. Use scripts/merge_allorders.py to combine and plot.")

    if wlog:
        wlog.log_usage(token_usage)
        wlog.log_failures(failure_summary)
        wlog.finish()


# ---------------------------------------------------------------------------
#  Mixed (round-robin) runner
# ---------------------------------------------------------------------------

def _split_into_minibatches(data, n_batches):
    """Split a list into n roughly equal mini-batches."""
    batch_size = len(data) // n_batches
    remainder = len(data) % n_batches
    batches = []
    start = 0
    for i in range(n_batches):
        end = start + batch_size + (1 if i < remainder else 0)
        batches.append(data[start:end])
        start = end
    return batches


def run_mixed(cfg):
    """Run mixed (round-robin) task interleaving optimization."""
    start_time = time.time()
    task_names = [t["name"] for t in cfg["tasks"]]

    client, model, tracker, failure_tracker, wlog, output_dir, metrics_log_path, eval_num_threads = setup_run(cfg)
    tasks_info = get_openevolve_tasks(task_names)
    task_data = load_all_datasets_raw(cfg, tasks_info, drop_val=True)

    mixed_cfg = cfg["mixed"]
    rounds_per_task = mixed_cfg["rounds_per_task"]
    iterations_per_round = mixed_cfg.get("iterations_per_round", 1)
    seed = mixed_cfg.get("seed", 42)

    task_batches = {}
    for name in task_names:
        task_batches[name] = _split_into_minibatches(
            task_data[name]["train_set"], rounds_per_task
        )

    rng = random.Random(seed)
    task_order = list(task_names)
    rng.shuffle(task_order)
    total_rounds = rounds_per_task * len(task_names)
    schedule = task_order * rounds_per_task
    checkpoint_interval = len(task_names)

    print(f"\nSchedule ({total_rounds} rounds, eval every {checkpoint_interval}):")
    print(f"  Task order: {' → '.join(task_order)}")

    task_round_idx = {name: 0 for name in task_names}

    skip_baseline = cfg.get("skip_baseline", False)
    all_scores = {name: [] for name in task_names}
    all_instructions = {}
    stages = []

    current_general_instructions = None

    with tracker.track_to_file():
        _last_usage = tracker.get_usage()

        if not skip_baseline:
            print(f"\n{'='*60}")
            print("Baseline (default instructions)")
            print(f"{'='*60}")
            stages.append("baseline")
            if failure_tracker:
                failure_tracker.set_stage("baseline")
            baseline_scores = {}
            for tn in task_names:
                default_instruction = TASK_REGISTRY[tn].get("default_instruction", "")
                score = score_on_task(default_instruction, tn, task_data[tn]["eval_set"],
                                     client, model, num_threads=eval_num_threads,
                                     stage="baseline")
                baseline_scores[tn] = score
                print(f"    {tn}: {score:.2f}")
            for name in task_names:
                all_scores[name].append(baseline_scores[name])
            _last_usage = tracker.get_usage()

        for round_num in range(total_rounds):
            task_name = schedule[round_num]
            cycle = round_num // len(task_names) + 1
            if failure_tracker:
                failure_tracker.set_stage(f"cycle_{cycle}/{task_name}")
            batch_idx = task_round_idx[task_name]
            task_round_idx[task_name] += 1

            print(f"\n{'─'*50}")
            print(f"  Round {round_num + 1}/{total_rounds} "
                  f"(cycle {cycle}, {task_name}, batch {batch_idx + 1}/{rounds_per_task})")
            print(f"{'─'*50}")

            task_default = TASK_REGISTRY[task_name].get("default_instruction", "")
            if current_general_instructions is None:
                general = task_default
            else:
                general = current_general_instructions

            initial_prompt_text = build_prompt(general, task_name)
            prompt_file = output_dir / f"round_{round_num}_{task_name}.txt"
            with open(prompt_file, "w") as f:
                f.write(initial_prompt_text)

            task_cfg = next(tc for tc in cfg["tasks"] if tc["name"] == task_name)
            task_config_path = write_task_config(task_cfg, cfg["model"],
                                                 output_dir / f"config_{task_name}.yaml",
                                                 eval_num_threads=eval_num_threads)
            os.environ["BENCHMARK_CONFIG"] = task_config_path

            oe_cfg = cfg["openevolve"].copy()
            oe_cfg["max_iterations"] = iterations_per_round
            oe_config = openevolve.Config.from_dict(apply_thinking_to_oe_dict(oe_cfg, cfg))

            stage_output_dir = str(output_dir / f"round_{round_num}_{task_name}")
            result = openevolve.run_evolution(
                initial_program=str(prompt_file),
                evaluator=tasks_info[task_name]["evaluator"],
                config=oe_config,
                output_dir=stage_output_dir,
            )

            current_general_instructions = clean_evolved_text(result.best_code)
            print(f"  Instructions updated ({len(current_general_instructions)} chars)")

            if (round_num + 1) % checkpoint_interval == 0:
                checkpoint_label = f"cycle_{cycle}"
                stages.append(checkpoint_label)
                all_instructions[checkpoint_label] = current_general_instructions

                _pre_eval_usage = tracker.get_usage()

                print(f"\n  *** Checkpoint after cycle {cycle} ***")
                checkpoint_scores = eval_all_tasks(
                    current_general_instructions, task_data, client, model,
                    num_threads=eval_num_threads, stage=checkpoint_label,
                )
                for name in task_names:
                    all_scores[name].append(checkpoint_scores[name])

                if metrics_log_path:
                    _post_eval_usage = tracker.get_usage()
                    log_stage(
                        metrics_log_path, checkpoint_label, "openevolve_mixed",
                        checkpoint_scores, _post_eval_usage,
                        optimization_usage=usage_diff(_pre_eval_usage, _last_usage),
                        eval_usage=usage_diff(_post_eval_usage, _pre_eval_usage),
                    )
                    _last_usage = _post_eval_usage

    elapsed = time.time() - start_time
    print_results_table(stages, all_scores, task_names)

    token_usage = tracker.get_usage()
    failure_summary = failure_tracker.summary() if failure_tracker else {"total": 0}

    print(f"\nToken usage: {json.dumps(token_usage, indent=2)}")
    print(f"\nTotal runtime: {elapsed / 60:.1f} minutes ({elapsed:.0f}s)")

    results = {
        "runtime_seconds": round(elapsed, 1),
        "runtime_hours": round(elapsed / 3600, 2),
        "stages": stages,
        "scores": all_scores,
        "instructions": all_instructions,
        "schedule": schedule,
        "task_order": task_order,
        "rounds_per_task": rounds_per_task,
        "usage": token_usage,
        "lm_failures": failure_summary,
        "config": cfg,
    }
    if metrics_log_path:
        results["metrics_log"] = read_metrics_log(metrics_log_path)

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    plot_path = output_dir / "mixed_performance.png"
    plot_sequential(results, str(plot_path))

    if wlog:
        wlog.log_usage(token_usage)
        wlog.log_failures(failure_summary)
        wlog.finish()
