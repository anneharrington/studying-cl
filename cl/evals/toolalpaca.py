import json
import os
import random
import re
from collections import Counter
from pathlib import Path

import dspy


# Strict scoring is the DEFAULT (mirrors the verl training pipeline's
# verl/utils/reward_score/feedback/tooluse.py::compute_score). To opt back into
# the legacy lenient partial-credit scorer set env var TOOLUSE_STRICT=0.
def _strict_scoring_enabled():
    val = os.environ.get("TOOLUSE_STRICT", "1").lower()
    return val in ("1", "true", "yes")


def _score_actions_strict(pred_actions, gold_steps):
    """Strict (RL-repo) tooluse scoring: 1.0 iff Counter(actions) match AND
    merged-Action_Input dicts are exactly equal, else 0.0.

    Mirrors verl/utils/reward_score/feedback/tooluse.py::compute_score.
    """
    if not gold_steps:
        return 1.0, "No golden steps to match."

    # Gold side
    gt_actions = [step["Action"] for step in gold_steps]
    gt_inputs_list = []
    for step in gold_steps:
        raw = step.get("Action_Input", "")
        if isinstance(raw, str):
            try:
                gt_inputs_list.append(json.loads(raw))
            except json.JSONDecodeError:
                gt_inputs_list.append({})
        elif isinstance(raw, dict):
            gt_inputs_list.append(raw)
        else:
            gt_inputs_list.append({})
    gt_merged = {}
    for d in gt_inputs_list:
        if d: gt_merged.update(d)

    # Pred side — pred_actions is list of (name, input_str) tuples per our regex
    pred_action_names = [a[0] for a in pred_actions]
    pred_merged = {}
    for _, inp in pred_actions:
        try:
            d = json.loads(inp)
            if isinstance(d, dict):
                pred_merged.update(d)
        except (json.JSONDecodeError, TypeError):
            continue

    actions_ok = Counter(pred_action_names) == Counter(gt_actions)
    inputs_ok = pred_merged == gt_merged

    correct = actions_ok and inputs_ok
    if correct:
        return 1.0, "Correct (strict): actions and merged inputs match exactly."
    fb = []
    if not actions_ok:
        fb.append(f"Action multiset mismatch: pred={pred_action_names} gold={gt_actions}")
    if not inputs_ok:
        fb.append(f"Merged input mismatch: pred={pred_merged} gold={gt_merged}")
    return 0.0, "; ".join(fb)


def _format_question(api_data, instruction):
    """Build a self-contained prompt from API docs + user instruction."""
    nl_docs = api_data["NLDocumentation"]
    api_name = api_data["Name"]
    api_desc = api_data["Description"]

    # List available function names
    func_names = list(api_data["Function_Projection"].keys())
    func_list = ", ".join(func_names)

    return (
        f"You have access to the following API: {api_name} - {api_desc}\n"
        f"Available functions: {func_list}\n\n"
        f"API Documentation:\n{nl_docs}\n\n"
        f"User request: {instruction}\n\n"
        f"Respond with the sequence of function calls needed, one per line, in this exact format:\n"
        f"Action: <function_name>\n"
        f"Action_Input: <json_parameters>\n\n"
        f"If multiple calls are needed, separate them with a blank line."
    )


def _format_golden(golden_answer):
    """Format golden answer as the expected output string."""
    parts = []
    for step in golden_answer:
        parts.append(f"Action: {step['Action']}\nAction_Input: {step['Action_Input']}")
    return "\n\n".join(parts)


def load_toolalpaca_raw(path: str, train_n: int, val_n: int, seed: int = 42, eval_n: int = 0):
    """Load ToolAlpaca eval_simulated and split into train/val/eval as plain dicts.

    Args:
        path: Path to toolalpaca_eval_simulated.json file, or a directory containing it.

    Returns two or three lists of {"question": str, "answer": str, "golden_steps": list} dicts.
    """
    base = Path(path)
    if base.is_file():
        data_file = base
    else:
        data_file = base / "data" / "toolalpaca_eval_simulated.json"
        if not data_file.exists():
            data_file = base / "toolalpaca_eval_simulated.json"
    if not data_file.exists():
        raise FileNotFoundError(f"Cannot find toolalpaca_eval_simulated.json at {path}")

    with open(data_file) as f:
        all_apis = json.load(f)

    examples = []
    for api_data in all_apis:
        for q_idx, instruction in enumerate(api_data["Instructions"]):
            golden = api_data["Golden_Answers"][q_idx]
            examples.append({
                "question": _format_question(api_data, instruction),
                "answer": _format_golden(golden),
                "golden_steps": golden,
                "api_name": api_data["Name"],
            })

    print(f"  Loaded {len(examples)} ToolAlpaca eval_simulated examples from {len(all_apis)} APIs")

    random.Random(seed).shuffle(examples)
    train_set = examples[:train_n]
    val_set = examples[train_n:] if val_n < 0 else examples[train_n : train_n + val_n]

    if eval_n == 0:
        return train_set, val_set

    eval_start = train_n + (len(examples) - train_n if val_n < 0 else val_n)
    eval_set = examples[eval_start:] if eval_n < 0 else examples[eval_start : eval_start + eval_n]
    return train_set, val_set, eval_set


def load_toolalpaca(path: str, train_n: int, val_n: int, seed: int = 42, eval_n: int = 0):
    """Load ToolAlpaca eval_simulated as dspy.Examples for GEPA."""
    splits = load_toolalpaca_raw(path, train_n, val_n, seed, eval_n)

    def to_dspy(items):
        return [
            dspy.Example(
                question=item["question"],
                answer=item["answer"],
            ).with_inputs("question")
            for item in items
        ]

    return tuple(to_dspy(s) for s in splits)


def _parse_actions(text):
    """Parse Action/Action_Input pairs from model output.

    Returns list of (action_name, action_input_str) tuples. If the text is
    ACE-shaped JSON, unwrap the final_answer field first so both ACE and
    native outputs are parsed identically.

    For multi-line JSON Action_Input values (which Qwen3-32B pretty-prints
    across newlines), uses brace-matching to capture the full JSON object.
    The legacy regex truncated at the first newline, returning just "{" and
    causing strict scoring to fail.
    """
    text = text or ""
    jm = re.search(r'"final_answer"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if jm:
        try:
            text = jm.group(1).encode().decode("unicode_escape")
        except (UnicodeDecodeError, UnicodeEncodeError):
            text = jm.group(1)
    actions = []
    # Find each "Action: <name>\nAction_Input: " header. The Action_Input
    # value is then extracted via _extract_action_input_value, which handles
    # both single-line and balanced multi-line JSON.
    header_pat = re.compile(
        r"Action\s*:\s*([^\n\r]+)[\n\r]+[ \t]*Action[_ ]?Input\s*:\s*",
        re.IGNORECASE,
    )
    for m in header_pat.finditer(text):
        name = m.group(1).strip()
        inp = _extract_action_input_value(text, m.end())
        actions.append((name, inp))
    return actions


def _extract_action_input_value(text, start):
    """Extract the Action_Input value beginning at `start`.

    If the first non-whitespace character is `{` or `[`, scans forward with
    brace/bracket counting (respecting JSON string literals + escapes) and
    returns the full balanced substring. Otherwise returns up to the next
    newline. Falls back to a single-line capture if the JSON is unbalanced.
    """
    # Skip leading spaces/tabs but not newlines (a newline before content
    # means the model wrote `Action_Input:\n<value>`).
    i = start
    while i < len(text) and text[i] in " \t":
        i += 1
    if i >= len(text):
        return ""

    first = text[i]
    if first in "{[":
        open_c, close_c = first, ("}" if first == "{" else "]")
        depth = 0
        in_string = False
        escape_next = False
        j = i
        while j < len(text):
            c = text[j]
            if escape_next:
                escape_next = False
            elif in_string:
                if c == "\\":
                    escape_next = True
                elif c == '"':
                    in_string = False
            else:
                if c == '"':
                    in_string = True
                elif c == open_c:
                    depth += 1
                elif c == close_c:
                    depth -= 1
                    if depth == 0:
                        return text[i:j + 1].strip()
            j += 1
        # Unbalanced — fall through to single-line capture as a safe fallback.

    nl = text.find("\n", i)
    return (text[i:nl] if nl >= 0 else text[i:]).strip()


def _parse_json_safe(s):
    """Try to parse JSON, return empty dict on failure."""
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return None


def _score_actions(pred_actions, gold_steps):
    """Score predicted actions against golden steps.

    Returns (score, feedback_str).
    Default: lenient partial-credit (fraction of golden steps matched by
    function name + key params).
    When TOOLUSE_STRICT env var is set: strict all-or-nothing matching the
    the verl training pipeline scoring exactly (Counter(actions) equality +
    merged-Action_Input dict equality).
    """
    if _strict_scoring_enabled():
        return _score_actions_strict(pred_actions, gold_steps)

    if not gold_steps:
        return 1.0, "No golden steps to match."

    matched = 0
    feedback_parts = []

    for i, gold_step in enumerate(gold_steps):
        gold_name = gold_step["Action"]
        gold_input = gold_step["Action_Input"]

        # Check if any predicted action matches this golden step
        step_matched = False
        for pred_name, pred_input in pred_actions:
            if pred_name != gold_name:
                continue

            # Function name matches — check params
            gold_params = _parse_json_safe(gold_input)
            pred_params = _parse_json_safe(pred_input)

            if gold_params is None or "${" in gold_input:
                # Template variable or unparseable — name match is sufficient
                step_matched = True
                break

            if pred_params is None:
                # Pred not parseable but gold is — partial credit for name match
                step_matched = True
                break

            # Both parseable — check if gold keys are present with correct values
            keys_match = True
            for k, v in gold_params.items():
                if k not in pred_params:
                    keys_match = False
                    break
                if str(pred_params[k]) != str(v):
                    keys_match = False
                    break

            if keys_match:
                step_matched = True
                break

        if step_matched:
            matched += 1
        else:
            feedback_parts.append(
                f"Step {i+1} not matched: expected Action={gold_name}, "
                f"Action_Input={gold_input}"
            )

    score = matched / len(gold_steps)
    if score == 1.0:
        feedback = "All tool calls matched correctly."
    else:
        feedback = (
            f"Matched {matched}/{len(gold_steps)} steps. "
            + " ".join(feedback_parts)
        )

    return score, feedback


def toolalpaca_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """Score predicted tool call sequence against golden answer.

    Parses Action/Action_Input pairs from prediction, compares against golden steps.
    Returns dspy.Prediction with score (0-1) and feedback.
    """
    pred_text = prediction.answer
    gold_text = example.answer

    pred_actions = _parse_actions(pred_text)
    gold_actions = _parse_actions(gold_text)

    # Reconstruct golden steps from parsed gold text
    gold_steps = [{"Action": name, "Action_Input": inp} for name, inp in gold_actions]

    score, feedback = _score_actions(pred_actions, gold_steps)

    if not pred_actions:
        feedback = (
            f"No Action/Action_Input pairs found in response. "
            f"Expected format: 'Action: <name>\\nAction_Input: <json>'. "
            f"Gold answer: {gold_text[:200]}"
        )
        score = 0.0

    return dspy.Prediction(score=score, feedback=feedback)
