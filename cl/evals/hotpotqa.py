import json
import random

import dspy

from cl.evals.hotpot_evaluate_v1 import f1_score


def load_hotpotqa_raw(path: str, train_n: int, val_n: int, seed: int = 42, eval_n: int = 0):
    """Load HotpotQA dev distractor set and split into train/val/eval as plain dicts.

    Returns two or three lists of {"question": str, "context": str, "answer": str} dicts.
    - train: first train_n examples (used for optimization)
    - val: next val_n examples (used for candidate scoring during optimization)
    - eval: remaining examples after train+val (used for post-optimization evaluation)
      Only returned when eval_n != 0. Use eval_n=-1 for all remaining.
    """
    with open(path) as f:
        data = json.load(f)

    examples = []
    for item in data:
        context = "\n\n".join(
            f"{title}: {' '.join(sentences)}"
            for title, sentences in item["context"]
        )
        examples.append({
            "question": item["question"],
            "context": context,
            "answer": item["answer"],
        })

    random.Random(seed).shuffle(examples)
    train_set = examples[:train_n]
    val_set = examples[train_n:] if val_n < 0 else examples[train_n : train_n + val_n]

    if eval_n == 0:
        return train_set, val_set

    eval_start = train_n + (len(examples) - train_n if val_n < 0 else val_n)
    eval_set = examples[eval_start:] if eval_n < 0 else examples[eval_start : eval_start + eval_n]
    return train_set, val_set, eval_set


def load_hotpotqa(path: str, train_n: int, val_n: int, seed: int = 42, eval_n: int = 0):
    """Load HotpotQA dev distractor set and split into train/val(/eval) as dspy.Examples for GEPA."""
    splits = load_hotpotqa_raw(path, train_n, val_n, seed, eval_n)

    def to_dspy(items):
        return [
            dspy.Example(
                question=item["question"],
                context=item["context"],
                answer=item["answer"],
            ).with_inputs("question", "context")
            for item in items
        ]

    return tuple(to_dspy(s) for s in splits)


def hotpotqa_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """F1-based metric that returns feedback for GEPA's reflection loop."""
    pred_answer = prediction.answer
    gold_answer = example.answer

    f1, prec, recall = f1_score(pred_answer, gold_answer)

    if f1 == 1.0:
        feedback = f"Correct. Predicted '{pred_answer}' matches gold '{gold_answer}'."
    elif f1 > 0:
        feedback = (
            f"Partially correct (F1={f1:.2f}). "
            f"Predicted '{pred_answer}', gold '{gold_answer}'. "
            f"Try to match the gold answer more precisely."
        )
    else:
        feedback = (
            f"Incorrect (F1=0). "
            f"Predicted '{pred_answer}', gold '{gold_answer}'. "
            f"Re-read the context carefully and identify the relevant information."
        )

    return dspy.Prediction(score=f1, feedback=feedback)
