"""Loader + metric for the SDPO-curated tooluse dataset.

Each JSONL line has a pre-formatted `prompt` (API docs + user request) and an
`answer` field that is a JSON-encoded list of {"Action", "Action_Input"} dicts.
"""

import json
import os
import random

import dspy

from cl.evals.toolalpaca import _parse_actions, _score_actions


def _to_example(item):
    golden_steps = _parse_golden(item["answer"])
    if not golden_steps:
        return None
    return {
        "question": _format_question(item),
        "answer": _format_golden(golden_steps),
        "golden_steps": golden_steps,
    }


def _format_question(item):
    """Return SDPO's pre-formatted prompt verbatim."""
    return item["prompt"]


def _parse_golden(answer_str):
    """Parse the JSON-encoded list of action steps."""
    try:
        steps = json.loads(answer_str)
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(steps, list):
        return []
    # Ensure Action_Input is a string (SDPO stores it as a JSON-encoded string already).
    out = []
    for s in steps:
        if not isinstance(s, dict) or "Action" not in s or "Action_Input" not in s:
            continue
        inp = s["Action_Input"]
        if not isinstance(inp, str):
            inp = json.dumps(inp)
        out.append({"Action": s["Action"], "Action_Input": inp})
    return out


def _format_golden(golden_steps):
    """Human-readable gold answer (same format as the model is asked to emit)."""
    parts = [
        f"Action: {step['Action']}\nAction_Input: {step['Action_Input']}"
        for step in golden_steps
    ]
    return "\n\n".join(parts)


def load_tooluse_raw(path: str, train_n: int, val_n: int, seed: int = 42, eval_n: int = 0):
    """Load tooluse examples and split into train/val(/eval) as plain dicts.

    Three modes, picked by `path`:

    1. **Parquet directory**: path holds `train.parquet` and `val.parquet`
       written by the verl training pipeline. Train parquet → seeded-shuffled
       50/50 into train+val; val parquet → eval. The parquet
       `reward_model.ground_truth` is the same JSON-encoded action list the
       JSONL `answer` field carries, so `_parse_golden` works unchanged.
    2. **JSONL with `<path>.meta.json`** containing {sdpo_train_count,
       sdpo_test_count}: legacy unshuffled SDPO test partition as eval.
    3. **Plain JSONL**: legacy shuffle-then-slice.
    """
    from cl.evals.parquet_io import is_parquet_dir, load_parquet_dir
    if is_parquet_dir(path):
        def _to_ex(it):
            golden_steps = _parse_golden(it["answer"])
            if not golden_steps:
                return None
            return {
                "question": it["question"],
                "answer": _format_golden(golden_steps),
                "golden_steps": golden_steps,
            }
        train, val, eval_set = load_parquet_dir(path, train_n, val_n, seed, eval_n, _to_ex)
        # Strip None rows the original loader filters out.
        return [e for e in train if e], [e for e in val if e], [e for e in eval_set if e]

    with open(path) as f:
        data = [json.loads(line) for line in f if line.strip()]

    meta_path = os.path.splitext(path)[0] + ".meta.json"
    meta = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    if meta and "sdpo_train_count" in meta:
        n_train_raw = int(meta["sdpo_train_count"])
        train_examples = [e for e in (_to_example(it) for it in data[:n_train_raw]) if e is not None]
        test_examples = [e for e in (_to_example(it) for it in data[n_train_raw:]) if e is not None]
        print(f"  Loaded {len(train_examples)} SDPO-train + {len(test_examples)} SDPO-test tooluse examples from {path}")
        random.Random(seed).shuffle(train_examples)
        train_set = train_examples[:train_n]
        val_set = train_examples[train_n:] if val_n < 0 else train_examples[train_n : train_n + val_n]
        if eval_n == 0:
            return train_set, val_set
        eval_set = test_examples if eval_n < 0 else test_examples[:eval_n]
        return train_set, val_set, eval_set

    # Legacy path: no partition metadata available.
    examples = [e for e in (_to_example(it) for it in data) if e is not None]
    print(f"  Loaded {len(examples)} tooluse examples from {path}")

    random.Random(seed).shuffle(examples)
    train_set = examples[:train_n]
    val_set = examples[train_n:] if val_n < 0 else examples[train_n : train_n + val_n]

    if eval_n == 0:
        return train_set, val_set

    eval_start = train_n + (len(examples) - train_n if val_n < 0 else val_n)
    eval_set = examples[eval_start:] if eval_n < 0 else examples[eval_start : eval_start + eval_n]
    return train_set, val_set, eval_set


def load_tooluse(path: str, train_n: int, val_n: int, seed: int = 42, eval_n: int = 0):
    """Load tooluse as dspy.Examples for GEPA."""
    splits = load_tooluse_raw(path, train_n, val_n, seed, eval_n)

    def to_dspy(items):
        return [
            dspy.Example(
                question=item["question"],
                answer=item["answer"],
            ).with_inputs("question")
            for item in items
        ]

    return tuple(to_dspy(s) for s in splits)


def tooluse_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """Score predicted tool calls against the golden step list."""
    pred_actions = _parse_actions(prediction.answer)
    gold_actions = _parse_actions(example.answer)
    gold_steps = [{"Action": name, "Action_Input": inp} for name, inp in gold_actions]

    score, feedback = _score_actions(pred_actions, gold_steps)

    if not pred_actions:
        feedback = (
            f"No Action/Action_Input pairs found in response. "
            f"Expected format: 'Action: <name>\\nAction_Input: <json>'. "
            f"Gold answer: {example.answer[:200]}"
        )
        score = 0.0

    return dspy.Prediction(score=score, feedback=feedback)
