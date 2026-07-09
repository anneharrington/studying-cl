import json
import random
import re

import dspy


def _is_mcq(answer):
    """Check if the ground truth is a single letter (MCQ answer)."""
    return len(answer.strip()) == 1 and answer.strip().upper() in "ABCDE"


def load_livebench_math_raw(path: str, train_n: int, val_n: int, seed: int = 42, eval_n: int = 0):
    """Load LiveBench math_comp subset and split into train/val/eval as plain dicts.

    Includes both AIME (3-digit integer answers) and AMC/SMC (MCQ letter answers).
    Each dict has: question (str), answer (str), answer_type ("mcq" or "integer").
    """
    with open(path) as f:
        data = [json.loads(line) for line in f]

    # Filter to math_comp only
    examples = []
    for item in data:
        if item["task"] != "math_comp":
            continue
        answer = item["ground_truth"].strip()
        examples.append({
            "question": item["turns"][0],
            "answer": answer,
            "answer_type": "mcq" if _is_mcq(answer) else "integer",
            "subtask": item.get("subtask", ""),
        })

    n_mcq = sum(1 for e in examples if e["answer_type"] == "mcq")
    n_int = sum(1 for e in examples if e["answer_type"] == "integer")
    print(f"  Loaded {len(examples)} LiveBench math_comp examples ({n_mcq} MCQ, {n_int} integer)")

    random.Random(seed).shuffle(examples)
    train_set = examples[:train_n]
    val_set = examples[train_n:] if val_n < 0 else examples[train_n : train_n + val_n]

    if eval_n == 0:
        return train_set, val_set

    eval_start = train_n + (len(examples) - train_n if val_n < 0 else val_n)
    eval_set = examples[eval_start:] if eval_n < 0 else examples[eval_start : eval_start + eval_n]
    return train_set, val_set, eval_set


def load_livebench_math(path: str, train_n: int, val_n: int, seed: int = 42, eval_n: int = 0):
    """Load LiveBench math_comp as dspy.Examples for GEPA."""
    splits = load_livebench_math_raw(path, train_n, val_n, seed, eval_n)

    def to_dspy(items):
        return [
            dspy.Example(
                question=item["question"],
                answer=item["answer"],
                answer_type=item["answer_type"],
            ).with_inputs("question")
            for item in items
        ]

    return tuple(to_dspy(s) for s in splits)


def _extract_three_digit_answer(text):
    """Extract a 3-digit integer answer from model response.

    Tries:
    1. Last standalone 3-digit number in the response
    2. Any number after "answer" keyword, padded to 3 digits
    3. Last number in text, padded
    """
    text = text.strip()

    three_digit = re.findall(r'(?<!\d)(\d{3})(?!\d)', text)
    if three_digit:
        return three_digit[-1]

    match = re.search(r'(?i)answer[:\s]*(\d+)', text)
    if match:
        return match.group(1).zfill(3)[-3:]

    numbers = re.findall(r'\d+', text)
    if numbers:
        return numbers[-1].zfill(3)[-3:]

    return ""


def _extract_mcq_answer(text):
    """Extract a single letter (A-E) from model response."""
    text = text.strip()

    # Look for common patterns
    for pattern in [r'(?i)answer\s*(?:is\s*)?[:\s]*\(?([A-Ea-e])\)?',
                    r'\b([A-Ea-e])\)?\s*$',
                    r'^([A-Ea-e])\b']:
        m = re.search(pattern, text)
        if m:
            return m.group(1).upper()

    # Fallback: last standalone letter A-E
    letters = re.findall(r'(?<![a-zA-Z])([A-Ea-e])(?![a-zA-Z])', text)
    if letters:
        return letters[-1].upper()

    return ""


def livebench_math_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """Exact match metric that handles both MCQ and integer answers.

    Returns dspy.Prediction with score (0 or 1) and feedback.
    """
    gold = example.answer.strip()
    answer_type = example.answer_type

    if answer_type == "mcq":
        pred = _extract_mcq_answer(prediction.answer)
        correct = pred == gold.upper()
    else:
        pred = _extract_three_digit_answer(prediction.answer)
        gold = gold.zfill(3)
        correct = pred == gold

    score = 1.0 if correct else 0.0

    if correct:
        feedback = f"Correct. Answer is {gold}."
    else:
        kind = "letter (A-E)" if answer_type == "mcq" else "3-digit integer (000-999)"
        feedback = (
            f"Incorrect. Predicted '{pred}', gold answer is '{gold}'. "
            f"The answer should be a {kind}."
        )

    return dspy.Prediction(score=score, feedback=feedback)
