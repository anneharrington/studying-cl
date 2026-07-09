import json
import random
import re
from pathlib import Path

import dspy


# Chemistry L3 subtasks (all MCQ-4-choices except balancing_chemical_equation which is filling)
CHEM_L3_FILES = [
    "molar_weight_calculation.jsonl",
    "molecular_property_calculation.jsonl",
    "molecule_structure_prediction.jsonl",
    "reaction_prediction.jsonl",
    "retrosynthesis.jsonl",
    "balancing_chemical_equation.jsonl",
]


def _format_question(item):
    """Build a self-contained question string from a SciKnowEval item."""
    prompt_text = item["prompt"]["default"]
    question = item["question"]

    if item["type"] == "mcq-4-choices":
        choices = item["choices"]
        choice_lines = "\n".join(
            f"{label}. {text}"
            for label, text in zip(choices["label"], choices["text"])
        )
        return f"{prompt_text}\n\n{question}\n\n{choice_lines}"
    else:
        # filling, true_or_false, etc.
        return f"{prompt_text}\n\n{question}"


def _get_gold_answer(item):
    """Extract the gold answer from a SciKnowEval item."""
    if item["type"].startswith("mcq"):
        return item["answerKey"]
    else:
        return item["answer"]


def load_sciknoweval_raw(path: str, train_n: int, val_n: int, seed: int = 42, eval_n: int = 0):
    """Load SciKnowEval Chemistry L3 subset and split into train/val/eval as plain dicts.

    Args:
        path: Directory containing raw_data/Chemistry/L3/*.jsonl files
              (either the cloned SciKnowEval repo or data/sciknoweval).

    Returns two or three lists of {"question": str, "answer": str, "task_type": str} dicts.
    """
    base = Path(path)

    # Support both data/sciknoweval (HF clone) and SciKnowEval repo layouts
    chem_l3 = base / "raw_data" / "Chemistry" / "L3"
    if not chem_l3.exists():
        chem_l3 = base / "Chemistry" / "L3"
    if not chem_l3.exists():
        raise FileNotFoundError(f"Cannot find Chemistry/L3 data under {base}")

    examples = []
    for fname in CHEM_L3_FILES:
        fpath = chem_l3 / fname
        if not fpath.exists():
            print(f"  Warning: {fpath} not found, skipping")
            continue
        with open(fpath) as f:
            for line in f:
                item = json.loads(line)
                examples.append({
                    "question": _format_question(item),
                    "answer": _get_gold_answer(item),
                    "task_type": item["type"],
                    "subtask": item["details"]["task"],
                })

    print(f"  Loaded {len(examples)} Chemistry-L3 examples from {len(CHEM_L3_FILES)} files")

    random.Random(seed).shuffle(examples)
    train_set = examples[:train_n]
    val_set = examples[train_n:] if val_n < 0 else examples[train_n : train_n + val_n]

    if eval_n == 0:
        return train_set, val_set

    eval_start = train_n + (len(examples) - train_n if val_n < 0 else val_n)
    eval_set = examples[eval_start:] if eval_n < 0 else examples[eval_start : eval_start + eval_n]
    return train_set, val_set, eval_set


def load_sciknoweval(path: str, train_n: int, val_n: int, seed: int = 42, eval_n: int = 0):
    """Load SciKnowEval Chemistry L3 and split into train/val(/eval) as dspy.Examples for GEPA."""
    splits = load_sciknoweval_raw(path, train_n, val_n, seed, eval_n)

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
    """Extract a single letter (A-D) from model output."""
    text = text.strip()
    # Try common patterns: standalone letter, parenthesized, with period
    for pattern in [r'^([A-Da-d])\b', r'\(([A-Da-d])\)', r'^([A-Da-d])\.']:
        m = re.search(pattern, text)
        if m:
            return m.group(1).upper()
    # Fallback: first character if it's A-D
    if text and text[0].upper() in "ABCD":
        return text[0].upper()
    return text.strip().upper()


def _extract_filling_answer(text):
    """Extract filling answer (e.g. balanced equation) from model output."""
    return text.strip()


def sciknoweval_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """Accuracy metric for SciKnowEval Chemistry L3.

    MCQ: exact letter match. Filling: substring containment (gold in pred).
    Returns score=1.0 for correct, 0.0 otherwise.
    """
    task_type = example.task_type
    gold = example.answer.strip()

    if task_type.startswith("mcq"):
        pred = _extract_mcq_answer(prediction.answer)
        correct = pred == gold.upper()
    else:
        # filling (balancing_chemical_equation): check if gold answer appears in response
        pred = _extract_filling_answer(prediction.answer)
        correct = gold in pred

    score = 1.0 if correct else 0.0

    if correct:
        feedback = f"Correct. Predicted '{pred}' matches gold '{gold}'."
    else:
        feedback = (
            f"Incorrect. Predicted '{pred}', gold '{gold}'. "
            f"Read the question carefully and select the correct answer."
        )

    return dspy.Prediction(score=score, feedback=feedback)
