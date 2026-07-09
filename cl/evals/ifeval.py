import json
import random

import dspy

from cl.evals.ifeval_lib.evaluation_lib import (
    InputExample,
    test_instruction_following_strict,
)


def load_ifeval_raw(path: str, train_n: int, val_n: int, seed: int = 42, eval_n: int = 0):
    """Load IFEval dataset and split into train/val/eval as plain dicts.

    Returns two or three lists of dicts with keys:
        prompt, instruction_id_list, kwargs, key
    Only returns eval split when eval_n != 0. Use eval_n=-1 for all remaining.
    """
    with open(path) as f:
        examples = [json.loads(line) for line in f]

    random.Random(seed).shuffle(examples)
    train_set = examples[:train_n]
    val_set = examples[train_n:] if val_n < 0 else examples[train_n : train_n + val_n]

    if eval_n == 0:
        return train_set, val_set

    eval_start = train_n + (len(examples) - train_n if val_n < 0 else val_n)
    eval_set = examples[eval_start:] if eval_n < 0 else examples[eval_start : eval_start + eval_n]
    return train_set, val_set, eval_set


def load_ifeval(path: str, train_n: int, val_n: int, seed: int = 42, eval_n: int = 0):
    """Load IFEval dataset and split into train/val(/eval) as dspy.Examples for GEPA."""
    splits = load_ifeval_raw(path, train_n, val_n, seed, eval_n)

    def to_dspy(items):
        return [
            dspy.Example(
                prompt=item["prompt"],
                instruction_id_list=item["instruction_id_list"],
                kwargs=item["kwargs"],
                key=item["key"],
            ).with_inputs("prompt")
            for item in items
        ]

    return tuple(to_dspy(s) for s in splits)


def ifeval_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """Constraint-based metric using Google's IFEval checkers.

    Returns a score = fraction of instructions followed (0.0 to 1.0),
    with feedback describing which constraints passed or failed.
    """
    inp = InputExample(
        key=example.key,
        instruction_id_list=example.instruction_id_list,
        prompt=example.prompt,
        kwargs=example.kwargs,
    )
    prompt_to_response = {example.prompt: prediction.response}

    output = test_instruction_following_strict(inp, prompt_to_response)

    n_total = len(output.follow_instruction_list)
    n_followed = sum(output.follow_instruction_list)
    score = n_followed / n_total if n_total > 0 else 0.0

    # Build feedback listing which instructions passed/failed
    details = []
    for inst_id, followed in zip(
        output.instruction_id_list, output.follow_instruction_list
    ):
        status = "PASS" if followed else "FAIL"
        details.append(f"  {status}: {inst_id}")

    if score == 1.0:
        feedback = f"All {n_total} instructions followed.\n" + "\n".join(details)
    elif score > 0:
        feedback = (
            f"Partially correct ({n_followed}/{n_total} instructions followed).\n"
            + "\n".join(details)
            + "\nRe-read the failed constraints and adjust the response format."
        )
    else:
        feedback = (
            f"No instructions followed (0/{n_total}).\n"
            + "\n".join(details)
            + "\nCarefully follow each formatting and content constraint."
        )

    return dspy.Prediction(score=score, feedback=feedback)
