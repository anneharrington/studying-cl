import json
import math
import os
import random
import re

import dspy


def _format_question(item):
    """Return the RL-repo's pre-formatted prompt verbatim.

    The the verl training pipeline pipeline (data/format/finqa.py) builds prompts
    from ChanceFocus/flare-finqa with `{Context: ..., Question: ...}` already
    flattened, and a system prompt instructing the model to box its answer.
    We pass that prompt through unchanged so train-time and eval-time prompts
    match.
    """
    return item["prompt"]


def _get_gold(item):
    """Gold answer is RL repo's normalized numeric string (e.g. "3.8")."""
    return item["answer"]


def _to_example(item):
    return {
        "question": _format_question(item),
        "answer": _get_gold(item),
        "task_type": "numeric",
    }


def load_finqa_raw(path: str, train_n: int, val_n: int, seed: int = 42, eval_n: int = 0):
    """Load FinQA examples and split into train/val(/eval).

    Three modes, picked by `path`:

    1. **Parquet directory** (preferred — what the GEPA sequential pipeline
       runs on now): path holds `train.parquet` and `val.parquet` written by
       the verl training pipeline. The training parquet is seeded-shuffled and
       split 50/50 into train+val; the val parquet is used verbatim as eval —
       so we evaluate on the exact rows training reports baselines on.
    2. **JSONL** with sidecar `<path>.meta.json` containing
       {rl_train_count, rl_test_count}: eval is pulled from the RL test
       partition unshuffled (legacy behavior, kept for back-compat).
    3. **Plain JSONL**: single seeded shuffle across all examples.
    """
    from cl.evals.parquet_io import is_parquet_dir, load_parquet_dir
    if is_parquet_dir(path):
        def _to_ex(it):
            return {"question": it["question"], "answer": it["answer"], "task_type": "numeric"}
        return load_parquet_dir(path, train_n, val_n, seed, eval_n, _to_ex)

    with open(path) as f:
        data = [json.loads(line) for line in f if line.strip()]

    meta_path = os.path.splitext(path)[0] + ".meta.json"
    meta = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    if meta and "rl_train_count" in meta:
        n_train_raw = int(meta["rl_train_count"])
        train_examples = [_to_example(it) for it in data[:n_train_raw]]
        test_examples = [_to_example(it) for it in data[n_train_raw:]]
        print(f"  Loaded {len(train_examples)} RL-train + {len(test_examples)} RL-test FinQA examples from {path}")
        random.Random(seed).shuffle(train_examples)
        train_set = train_examples[:train_n]
        val_set = train_examples[train_n:] if val_n < 0 else train_examples[train_n : train_n + val_n]
        if eval_n == 0:
            return train_set, val_set
        eval_set = test_examples if eval_n < 0 else test_examples[:eval_n]
        return train_set, val_set, eval_set

    examples = [_to_example(it) for it in data]
    print(f"  Loaded {len(examples)} FinQA examples from {path}")

    random.Random(seed).shuffle(examples)
    train_set = examples[:train_n]
    val_set = examples[train_n:] if val_n < 0 else examples[train_n : train_n + val_n]

    if eval_n == 0:
        return train_set, val_set

    eval_start = train_n + (len(examples) - train_n if val_n < 0 else val_n)
    eval_set = examples[eval_start:] if eval_n < 0 else examples[eval_start : eval_start + eval_n]
    return train_set, val_set, eval_set


def load_finqa(path: str, train_n: int, val_n: int, seed: int = 42, eval_n: int = 0):
    """Load FinQA test set as dspy.Examples for GEPA."""
    splits = load_finqa_raw(path, train_n, val_n, seed, eval_n)

    def to_dspy(items):
        return [
            dspy.Example(
                question=item["question"],
                answer=str(item["answer"]),
                task_type=item["task_type"],
            ).with_inputs("question")
            for item in items
        ]

    return tuple(to_dspy(s) for s in splits)


# ---------------------------------------------------------------------------
# Scoring — mirrors the verl training pipeline's
# verl/utils/reward_score/feedback/math.py::_finqa_verify exactly:
#   * 3-tier extraction: ####, Answer:, last-number
#   * 1% relative OR 1e-3 absolute tolerance
#   * NO percent↔decimal flip and NO scale-word grace, so eval scores match
#     the RL training pipeline's reported numbers.
# ---------------------------------------------------------------------------

_HASHES_RE = re.compile(r"####\s*(.+)")
_ANSWER_RE = re.compile(r"(?i)answer\s*:\s*\$?([-\d,]+(?:\.\d+)?%?)")
_NUM_RE = re.compile(r"-?[\d,]+(?:\.\d+)?%?")


def _parse_number(s):
    """Parse a number string to a float.

    Strips commas, $, % (treating "25%" and "25" as the same number 25.0),
    accounting parens. Returns None if unparseable. Scale words are NOT
    recognized — RL repo's _finqa_normalize_number does not handle them.
    """
    if s is None:
        return None
    text = str(s).strip().lower()
    paren = re.match(r"^\(([-\d.,\s$%]+)\)$", text)
    if paren:
        text = "-" + paren.group(1)
    text = text.replace(",", "").replace("$", "").replace("%", "").strip()
    try:
        val = float(text)
        return val if math.isfinite(val) else None
    except (ValueError, OverflowError):
        return None


def _normalize_number(s):
    """Canonical form for both gold and pred — matches RL repo's _finqa_normalize_number."""
    val = _parse_number(s)
    if val is None:
        return str(s).strip()
    if val == int(val):
        return str(int(val))
    return f"{val:.4f}".rstrip("0").rstrip(".")


def _extract_number(text):
    """3-tier extraction (####, Answer:, last-number), matching the RL repo's
    _finqa_extract_number. Returns the raw token (caller normalizes).
    """
    text = (text or "").strip()

    # ACE-shaped JSON: unwrap final_answer first so ACE and native outputs
    # are parsed identically.
    jm = re.search(r'"final_answer"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if jm:
        try:
            text = jm.group(1).encode().decode("unicode_escape").strip()
        except (UnicodeDecodeError, UnicodeEncodeError):
            text = jm.group(1).strip()

    m = _HASHES_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _ANSWER_RE.search(text)
    if m:
        return m.group(1).strip()
    nums = _NUM_RE.findall(text)
    if nums:
        return nums[-1].strip()
    return text


# Back-compat alias — callers in ace_processors.py / openevolve_runner.py
# import `_extract_predicted_number` from this module.
_extract_predicted_number = _extract_number


def _numbers_match(pred, gold, rel_tol=0.01, abs_tol=1e-3):
    """Numeric match — mirrors RL repo's _finqa_verify exactly: 1% relative OR
    1e-3 absolute tolerance, NO percent↔decimal flip, NO scale-word grace.
    """
    p_val = _parse_number(pred)
    g_val = _parse_number(gold)
    if p_val is None or g_val is None:
        return False
    if abs(p_val - g_val) < abs_tol:
        return True
    if g_val != 0 and abs((p_val - g_val) / g_val) < rel_tol:
        return True
    return False


def finqa_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """Numeric match metric for FinQA — RL-repo-compatible.

    Mirrors verl/utils/reward_score/feedback/math.py::_finqa_verify in
    the verl training pipeline:
        * 3-tier extraction (####, Answer:, last-number) via _extract_number
        * 1% relative OR 1e-3 absolute tolerance
        * NO percent↔decimal flip, NO scale-word handling
    """
    gold = _normalize_number(example.answer)
    pred_raw = _extract_number(prediction.answer)
    pred = _normalize_number(pred_raw)

    gold_val = _parse_number(example.answer)
    pred_val = _parse_number(pred_raw)

    if gold_val is not None and pred_val is not None:
        if abs(pred_val - gold_val) < 1e-3:
            correct = True
        elif gold_val != 0 and abs((pred_val - gold_val) / gold_val) < 0.01:
            correct = True
        else:
            correct = False
    else:
        correct = pred == gold

    score = 1.0 if correct else 0.0

    if correct:
        feedback = f"Correct. Answer is {gold}."
    else:
        feedback = (
            f"Incorrect. Predicted '{pred}' (from '{pred_raw}'), "
            f"gold answer is '{gold}'. "
            "Show your calculations step by step using the financial data provided."
        )

    return dspy.Prediction(score=score, feedback=feedback)
