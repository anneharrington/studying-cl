import json
import random
import re

import dspy


def _extract_gold_number(answer_str):
    """Extract the final numeric answer after #### from GSM8K answer string."""
    match = re.search(r"####\s*(.+)", answer_str)
    if match:
        return match.group(1).strip().replace(",", "")
    return answer_str.strip()


def load_gsm8k_raw(path: str, train_n: int, val_n: int, seed: int = 42, eval_n: int = 0):
    """Load GSM8K test set and split into train/val/eval as plain dicts.

    Each dict has: question (str), answer (str — just the final number).
    """
    with open(path) as f:
        data = [json.loads(line) for line in f]

    examples = [
        {
            "question": item["question"],
            "answer": _extract_gold_number(item["answer"]),
        }
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


def load_gsm8k(path: str, train_n: int, val_n: int, seed: int = 42, eval_n: int = 0):
    """Load GSM8K test set as dspy.Examples for GEPA."""
    splits = load_gsm8k_raw(path, train_n, val_n, seed, eval_n)

    def to_dspy(items):
        return [
            dspy.Example(
                question=item["question"],
                answer=item["answer"],
            ).with_inputs("question")
            for item in items
        ]

    return tuple(to_dspy(s) for s in splits)


def _extract_predicted_number(text):
    """Extract the final numeric answer from model output.

    Tries several patterns:
    1. #### <number>  (GSM8K format)
    2. "the answer is <number>"
    3. Last number in the response
    """
    text = text.strip()

    # Pattern 1: #### marker
    match = re.search(r"####\s*(.+)", text)
    if match:
        return match.group(1).strip().replace(",", "")

    # Pattern 2: "the answer is X" or "Answer: X"
    match = re.search(r"(?i)(?:the answer is|answer\s*:)\s*\$?([\d,]+(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")

    # Pattern 3: Last number in the text (including negative and decimal)
    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return text


def _normalize_number(s):
    """Normalize a number string for comparison (strip trailing .0, commas)."""
    s = s.replace(",", "").strip()
    try:
        val = float(s)
        if val == int(val):
            return str(int(val))
        return str(val)
    except ValueError:
        return s


def gsm8k_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """Exact-match metric on final numeric answer.

    Extracts the number from the prediction, compares to gold answer.
    Returns dspy.Prediction with score (0 or 1) and feedback.
    """
    gold = _normalize_number(example.answer)
    pred_raw = _extract_predicted_number(prediction.answer)
    pred = _normalize_number(pred_raw)

    correct = pred == gold
    score = 1.0 if correct else 0.0

    if correct:
        feedback = f"Correct. Answer is {gold}."
    else:
        feedback = (
            f"Incorrect. Predicted '{pred}' (extracted from '{pred_raw}'), "
            f"gold answer is '{gold}'. "
            f"Show your work step by step and put the final answer after ####."
        )

    return dspy.Prediction(score=score, feedback=feedback)
