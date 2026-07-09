"""Loader + metric for sciknoweval-bio (Biology L3 subset curated by SDPO).

The data file is the concatenation of SDPO's train/test JSONL splits built by
scripts/build_sdpo_datasets.sh. Each line already contains a fully-formatted
prompt (question + choices) and a single-letter gold answer (A-D).
"""

import json
import os
import random
import re

import dspy


def _format_question(item):
    """Return SDPO's pre-formatted prompt string verbatim."""
    return item["prompt"]


def _get_gold(item):
    """Gold answer is an uppercase letter A-D."""
    return item["answer"].strip().upper()


def _to_example(item):
    return {
        "question": _format_question(item),
        "answer": _get_gold(item),
        "task_type": "mcq-4-choices",
    }


def load_sciknoweval_bio_raw(path: str, train_n: int, val_n: int, seed: int = 42, eval_n: int = 0):
    """Load sciknoweval-bio and split into train/val(/eval) as plain dicts.

    Three modes, picked by `path`:

    1. **Parquet directory**: path holds `train.parquet` and `val.parquet`
       from the verl training pipeline. Training parquet → seeded-shuffled 50/50
       split into train+val; val parquet → eval. Matches the training pipeline's
       eval set exactly (val_bio.parquet is the same 50 rows training scores
       33.75% on).
    2. **JSONL with `<path>.meta.json`** containing {sdpo_train_count,
       sdpo_test_count}: eval pulled from the SDPO test partition unshuffled.
    3. **Plain JSONL**: legacy shuffle-then-slice.
    """
    from cl.evals.parquet_io import is_parquet_dir, load_parquet_dir
    if is_parquet_dir(path):
        def _to_ex(it):
            return {
                "question": it["question"],
                "answer": str(it["answer"]).strip().upper(),
                "task_type": "mcq-4-choices",
            }
        return load_parquet_dir(path, train_n, val_n, seed, eval_n, _to_ex)

    with open(path) as f:
        data = [json.loads(line) for line in f if line.strip()]

    meta_path = os.path.splitext(path)[0] + ".meta.json"
    meta = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    if meta and "sdpo_train_count" in meta:
        n_train_raw = int(meta["sdpo_train_count"])
        train_examples = [_to_example(it) for it in data[:n_train_raw]]
        test_examples = [_to_example(it) for it in data[n_train_raw:]]
        print(f"  Loaded {len(train_examples)} SDPO-train + {len(test_examples)} SDPO-test sciknoweval-bio examples from {path}")
        random.Random(seed).shuffle(train_examples)
        train_set = train_examples[:train_n]
        val_set = train_examples[train_n:] if val_n < 0 else train_examples[train_n : train_n + val_n]
        if eval_n == 0:
            return train_set, val_set
        eval_set = test_examples if eval_n < 0 else test_examples[:eval_n]
        return train_set, val_set, eval_set

    # Legacy path
    examples = [_to_example(it) for it in data]
    print(f"  Loaded {len(examples)} sciknoweval-bio examples from {path}")

    random.Random(seed).shuffle(examples)
    train_set = examples[:train_n]
    val_set = examples[train_n:] if val_n < 0 else examples[train_n : train_n + val_n]

    if eval_n == 0:
        return train_set, val_set

    eval_start = train_n + (len(examples) - train_n if val_n < 0 else val_n)
    eval_set = examples[eval_start:] if eval_n < 0 else examples[eval_start : eval_start + eval_n]
    return train_set, val_set, eval_set


def load_sciknoweval_bio(path: str, train_n: int, val_n: int, seed: int = 42, eval_n: int = 0):
    """Load sciknoweval-bio as dspy.Examples for GEPA."""
    splits = load_sciknoweval_bio_raw(path, train_n, val_n, seed, eval_n)

    def to_dspy(items):
        return [
            dspy.Example(
                question=item["question"],
                answer=item["answer"],
                task_type=item["task_type"],
            ).with_inputs("question")
            for item in items
        ]

    return tuple(to_dspy(s) for s in splits)


def _extract_mcq_answer(text):
    """Extract a single letter (A-D) from model output.

    Handles three output styles in priority order:
      1. ACE-shaped JSON `final_answer` field — unwrap first, then re-extract.
      2. Training XML format `<answer>X</answer>` — what the verl training pipeline
         tells the model to produce via the system prompt.
      3. Legacy `Answer: X` line + bare-letter fallback for older evolved
         prompts that ask for `Answer:`-style output.
    """
    text = (text or "").strip()

    # 1. ACE-shaped JSON
    jm = re.search(r'"final_answer"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if jm:
        try:
            text = jm.group(1).encode().decode("unicode_escape").strip()
        except (UnicodeDecodeError, UnicodeEncodeError):
            text = jm.group(1).strip()

    # 2. Training XML format: prefer the LAST <answer>...</answer> block so
    # exemplars in the system prompt (e.g. "<answer>A</answer>") don't shadow
    # the model's own answer when echoed back.
    xml_matches = re.findall(r"<answer>\s*([A-Da-d])\s*</answer>", text,
                             re.IGNORECASE | re.DOTALL)
    if xml_matches:
        return xml_matches[-1].upper()

    # 3a. Explicit "Answer: X" line.
    am = re.search(r"(?im)^\s*answer\s*:\s*([A-Da-d])\b", text)
    if am:
        return am.group(1).upper()

    # 3b. Bare-letter patterns at line start / parens.
    for pattern in [r'^([A-Da-d])\b', r'\(([A-Da-d])\)', r'^([A-Da-d])\.']:
        m = re.search(pattern, text)
        if m:
            return m.group(1).upper()
    if text and text[0].upper() in "ABCD":
        return text[0].upper()
    return text.strip().upper()


def sciknoweval_bio_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """Exact-letter match on A-D."""
    gold = example.answer.strip().upper()
    pred = _extract_mcq_answer(prediction.answer)
    correct = pred == gold
    score = 1.0 if correct else 0.0

    if correct:
        feedback = f"Correct. Predicted '{pred}' matches gold '{gold}'."
    else:
        feedback = (
            f"Incorrect. Predicted '{pred}', gold '{gold}'. "
            f"Read the question carefully and select the single correct letter."
        )

    return dspy.Prediction(score=score, feedback=feedback)
