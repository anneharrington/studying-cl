"""SciKnowEval SDPO-format MCQ tasks (biology/chemistry/physics/material).

Data format from lasgroup/SDPO repo: prompt has question+options inline,
answer is single letter A/B/C/D. System prompt requests
<reasoning>...</reasoning><answer>X</answer> format.
"""

import json
import random
import re

import dspy


def load_sciknow_raw(path, train_n, val_n, seed=42, eval_n=0):
    with open(path) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    # Build normalized records: loader expects 'question' + 'answer'.
    # Accept both raw (prompt) and relabeled-rollout (question) formats.
    examples = []
    for r in rows:
        if "question" in r and "answer" in r:
            ex = {
                "question": r["question"],
                "answer": str(r["answer"]).strip().upper(),
                "system": r.get("system", ""),
            }
            if "model_response" in r:
                ex["model_response"] = r["model_response"]
            examples.append(ex)
            continue
        examples.append({
            "question": r["prompt"],
            "answer": r["answer"].strip().upper(),
            "system": r.get("system", ""),
        })
    random.Random(seed).shuffle(examples)
    if train_n < 0:
        train_n = len(examples)
    train_set = examples[:train_n]
    val_set = examples[train_n:] if val_n < 0 else examples[train_n:train_n + val_n]
    if eval_n == 0:
        return train_set, val_set
    eval_start = train_n + (len(examples) - train_n if val_n < 0 else val_n)
    eval_set = examples[eval_start:] if eval_n < 0 else examples[eval_start:eval_start + eval_n]
    return train_set, val_set, eval_set


def load_sciknow(path, train_n, val_n, seed=42, eval_n=0):
    splits = load_sciknow_raw(path, train_n, val_n, seed, eval_n)
    def to_dspy(items):
        return [
            dspy.Example(question=x["question"], answer=x["answer"])
                .with_inputs("question")
            for x in items
        ]
    return tuple(to_dspy(s) for s in splits)


def _extract_letter(text):
    """Pull A/B/C/D from SDPO-style response."""
    # Primary: <answer>X</answer>
    m = re.search(r"<answer>\s*([A-D])\s*</answer>", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Fallback: "Answer: X"
    m = re.search(r"(?i)answer\s*:\s*([A-D])\b", text)
    if m:
        return m.group(1).upper()
    # Last A-D in last 200 chars
    matches = re.findall(r"\b([A-D])\b", text[-200:])
    return matches[-1].upper() if matches else ""


def sciknow_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    gold = example.answer.strip().upper()
    pred = _extract_letter(prediction.answer)
    correct = pred == gold
    score = 1.0 if correct else 0.0
    feedback = (f"Correct. Answer is {gold}." if correct
                else f"Incorrect. Predicted '{pred}', gold '{gold}'.")
    return dspy.Prediction(score=score, feedback=feedback)
