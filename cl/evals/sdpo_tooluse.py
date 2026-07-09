"""SDPO-style tooluse dataset loader + metric.

Data format: {prompt, answer=[{Action,Action_Input}...], system}.
Metric: reuses toolalpaca's Action/Action_Input parser + scorer.
"""

import json
import random

import dspy

from cl.evals.toolalpaca import _parse_actions, _score_actions


def _serialize_answer(answer_list):
    """Convert list of {Action,Action_Input} dicts to text the parser understands.

    The source field is stored as a JSON-encoded string, so parse first if needed.
    """
    if isinstance(answer_list, str):
        try:
            answer_list = json.loads(answer_list)
        except json.JSONDecodeError:
            return answer_list
    if not isinstance(answer_list, list):
        return str(answer_list)
    parts = []
    for a in answer_list:
        if isinstance(a, dict):
            parts.append(f"Action: {a.get('Action','')}\nAction_Input: {a.get('Action_Input','')}")
        else:
            parts.append(str(a))
    return "\n".join(parts)


def load_sdpo_tool_raw(path, train_n, val_n, seed=42, eval_n=0):
    with open(path) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    examples = []
    for r in rows:
        # Relabeled rollout format already has `question` (already serialized answer)
        # and optional `model_response`. Raw SDPO-tool format has `prompt` + answer list.
        if "question" in r and "answer" in r:
            ex = {"question": r["question"], "answer": str(r["answer"])}
            if "model_response" in r:
                ex["model_response"] = r["model_response"]
            examples.append(ex)
            continue
        ans_list = r.get("answer", [])
        examples.append({
            "question": r["prompt"],
            "answer": _serialize_answer(ans_list),
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


def load_sdpo_tool(path, train_n, val_n, seed=42, eval_n=0):
    splits = load_sdpo_tool_raw(path, train_n, val_n, seed, eval_n)
    def to_dspy(items):
        return [
            dspy.Example(question=x["question"], answer=x["answer"]).with_inputs("question")
            for x in items
        ]
    return tuple(to_dspy(s) for s in splits)


def sdpo_tool_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    pred_text = prediction.answer
    gold_text = example.answer
    pred_actions = _parse_actions(pred_text)
    gold_actions = _parse_actions(gold_text)
    gold_steps = [{"Action": n, "Action_Input": inp} for n, inp in gold_actions]
    if not pred_actions:
        return dspy.Prediction(score=0.0, feedback="no Action/Action_Input found")
    score, fb = _score_actions(pred_actions, gold_steps)
    return dspy.Prediction(score=float(score), feedback=fb)
