import json
import random

import dspy


def load_hover_raw(path: str, train_n: int, val_n: int, seed: int = 42, eval_n: int = 0):
    """Load HoVer dataset and split into train/val/eval as plain dicts.

    Returns two or three lists of dicts with keys:
        claim, label, uid
    Only returns eval split when eval_n != 0. Use eval_n=-1 for all remaining.
    """
    with open(path) as f:
        data = json.load(f)

    examples = [
        {"claim": item["claim"], "label": item["label"], "uid": item["uid"]}
        for item in data
    ]

    random.Random(seed).shuffle(examples)
    train_set = examples[:train_n]
    val_set = examples[train_n:] if val_n < 0 else examples[train_n : train_n + val_n]

    if eval_n == 0:
        return train_set, val_set

    eval_start = train_n + (len(examples) - train_n if val_n < 0 else val_n)
    eval_set = examples[eval_start:] if eval_n < 0 else examples[eval_start : eval_start + eval_n]
    return train_set, val_set, eval_set


def load_hover(path: str, train_n: int, val_n: int, seed: int = 42, eval_n: int = 0):
    """Load HoVer dataset and split into train/val(/eval) as dspy.Examples for GEPA."""
    splits = load_hover_raw(path, train_n, val_n, seed, eval_n)

    def to_dspy(items):
        return [
            dspy.Example(
                claim=item["claim"],
                label=item["label"],
            ).with_inputs("claim")
            for item in items
        ]

    return tuple(to_dspy(s) for s in splits)


def hover_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """Accuracy metric for HoVer claim verification.

    Compares predicted label against gold label (SUPPORTED / NOT_SUPPORTED).
    Returns score=1.0 for exact match, 0.0 otherwise.
    """
    gold = example.label.strip().upper()
    pred = prediction.label.strip().upper()

    # Normalize common variations
    if "NOT" in pred:
        pred_normalized = "NOT_SUPPORTED"
    elif "SUPPORT" in pred:
        pred_normalized = "SUPPORTED"
    else:
        pred_normalized = pred

    correct = pred_normalized == gold
    score = 1.0 if correct else 0.0

    if correct:
        feedback = f"Correct. Predicted '{pred_normalized}' matches gold '{gold}'."
    else:
        feedback = (
            f"Incorrect. Predicted '{pred_normalized}', gold '{gold}'. "
            f"Carefully evaluate whether the claim is factually supported."
        )

    return dspy.Prediction(score=score, feedback=feedback)
