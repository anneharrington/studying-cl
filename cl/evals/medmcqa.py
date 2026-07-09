"""MedMCQA dataset loader and metric.

Dataset: MedMCQA (Indian medical exam MCQs, 4-way multiple choice).
Fields: question, options (pre-formatted "A. ...\\nB. ..." string), answer (letter).
Metric: Exact letter match (A/B/C/D).
"""

import json
import random

import dspy

from cl.evals.medqa import _extract_letter


def load_medmcqa_raw(path: str, train_n: int, val_n: int,
                      seed: int = 42, eval_n: int = 0):
    """Load MedMCQA as plain dicts.

    Each dict has: question, options (formatted string), answer (letter).
    Accepts both the raw JSONL format and relabel outputs (with model_response).
    """
    with open(path) as f:
        data = [json.loads(line) for line in f if line.strip()]

    examples = []
    for item in data:
        # HF multiple-choice format (medmcqa_val.jsonl): sent1/sent2/ending0-3/label
        if "sent1" in item and "label" in item:
            question = item["sent1"]
            if item.get("sent2"):
                question = f"{question} {item['sent2']}".strip()
            endings = [item[f"ending{i}"] for i in range(4) if f"ending{i}" in item]
            options = "\n".join(f"{chr(ord('A') + i)}. {e}" for i, e in enumerate(endings))
            letter = chr(ord("A") + int(item["label"]))
            examples.append({
                "question": question,
                "options": options,
                "answer": letter,
            })
            continue

        # Normalized format (medmcqa_train.jsonl, relabeled): question/options/answer
        q = item.get("question", "")
        opts = item.get("options", "")
        ans = item.get("answer", "")
        if not (q and opts and ans):
            continue
        ex = {
            "question": q,
            "options": opts if isinstance(opts, str) else str(opts),
            "answer": str(ans).strip().upper(),
        }
        if "model_response" in item:
            ex["model_response"] = item["model_response"]
        examples.append(ex)

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


def load_medmcqa(path: str, train_n: int, val_n: int,
                  seed: int = 42, eval_n: int = 0):
    """Load MedMCQA as dspy.Examples."""
    splits = load_medmcqa_raw(path, train_n, val_n, seed, eval_n)

    def to_dspy(items):
        return [
            dspy.Example(
                question=item["question"],
                options=item["options"],
                answer=item["answer"],
            ).with_inputs("question", "options")
            for item in items
        ]

    return tuple(to_dspy(s) for s in splits)


def medmcqa_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """Exact letter match metric."""
    gold = example.answer.strip().upper()
    pred = _extract_letter(prediction.answer)

    correct = gold == pred
    score = 1.0 if correct else 0.0

    if correct:
        feedback = f"Correct. Answer is {gold}."
    else:
        feedback = (
            f"Incorrect. Predicted '{pred}', gold answer is '{gold}'. "
            'Give your final answer as "Answer: X" where X is A, B, C, or D.'
        )

    return dspy.Prediction(score=score, feedback=feedback)
