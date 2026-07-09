# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py

import math as _std_math
import re
import signal
from typing import Optional
from math_verify import parse as mv_parse, verify as mv_verify

FORMAT_PENALTY = False


def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last LaTeX boxed expression from a string.

    Args:
        string: Input string containing LaTeX code

    Returns:
        The last boxed expression or None if not found
    """
    idx = string.rfind(r"\boxed{")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else ""#None


def remove_boxed(s: str) -> str:
    r"""Remove the LaTeX boxed command from a string.

    Args:
        s: String with format "\boxed{content}"

    Returns:
        The content inside the boxed command
    """
    left = r"\boxed{"
    #assert s[: len(left)] == left, f"box error: {s}"
    #assert s[-1] == "}", f"box error: {s}"
    if s[: len(left)] == left and  s[-1] == "}":
        return s[len(left) : -1]
    else:
        return ""


class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def is_correct_strict_box(
    pred: str, gt: str, pause_tokens_index: Optional[list[int]] = None
) -> tuple[int, Optional[str]]:
    """Check if the prediction is correct using strict boxed answer criteria.

    Args:
        pred: The prediction string
        gt: The ground truth answer
        pause_tokens_index: Indices of pause tokens

    Returns:
        Tuple of (score, extracted_prediction)
    """
    # Extract the relevant part of the prediction
    if pause_tokens_index is not None:
        assert len(pause_tokens_index) == 4
        pred = pred[pause_tokens_index[-1] - 100 :]
    else:
        pred = pred[-100:]

    # Extract and check the boxed answer
    boxed_pred = last_boxed_only_string(pred)
    extracted_pred = remove_boxed(boxed_pred) if boxed_pred is not None else None

    return extracted_pred == gt, extracted_pred


# ---------------------------------------------------------------------------
# FinQA verifier — ported verbatim from
# anneharrington/adaptive_intelligence_benchmark @ lora:src/evals/finqa.py.
# Used only when `numeric_tolerance=True` (data_source=="finqa"); math/gsm8k/
# dapo_math keep the strict-box + math-verify path above unchanged.
# ---------------------------------------------------------------------------

_FINQA_HASHES_RE = re.compile(r"####\s*(.+)")
_FINQA_ANSWER_RE = re.compile(r"(?i)answer\s*:\s*\$?([-\d,]+(?:\.\d+)?%?)")
_FINQA_NUM_RE = re.compile(r"-?[\d,]+(?:\.\d+)?%?")


def _finqa_normalize_number(s: str) -> str:
    s = str(s).replace(",", "").replace("%", "").replace("$", "").strip()
    try:
        val = float(s)
        if not _std_math.isfinite(val):
            return s
        if val == int(val):
            return str(int(val))
        return f"{val:.4f}".rstrip("0").rstrip(".")
    except (ValueError, OverflowError):
        return s


def _finqa_extract_number(text: str) -> tuple[str, bool]:
    """3-tier extraction (####, Answer:, last-number). Returns (pred_raw, matched)."""
    text = text.strip()
    m = _FINQA_HASHES_RE.search(text)
    if m:
        return m.group(1).strip(), True
    m = _FINQA_ANSWER_RE.search(text)
    if m:
        return m.group(1), True
    nums = _FINQA_NUM_RE.findall(text)
    if nums:
        return nums[-1], True
    return text, False


def _finqa_verify(solution_str: str, answer: str) -> tuple[bool, str, bool]:
    """Returns (correct, pred_normalized, format_matched)."""
    gold = _finqa_normalize_number(answer)
    pred_raw, matched = _finqa_extract_number(solution_str)
    pred = _finqa_normalize_number(pred_raw)
    try:
        gold_val = float(gold)
        pred_val = float(pred)
        correct = abs(gold_val - pred_val) < 1e-3 or (
            gold_val != 0 and abs((gold_val - pred_val) / gold_val) < 0.01
        )
    except (ValueError, ZeroDivisionError):
        correct = pred == gold
    return correct, pred, matched


def verify(
    solution_str: str, answer: str, pause_tokens_index: Optional[list[int]] = None,
    numeric_tolerance: bool = False,
) -> bool:
    """Verify if the solution is correct.

    Args:
        solution_str: The solution string to verify
        answer: The ground truth answer
        pause_tokens_index: Indices of pause tokens
        numeric_tolerance: if True, route to the FinQA verifier (3-tier extraction +
            1% rel / 1e-3 abs tolerance, ported from Anne Harrington's benchmark).
            Enabled only for data_source=="finqa"; math/gsm8k/etc. stay on the strict
            boxed + math-verify path.

    Returns:
        True if the solution is correct, False otherwise
    """
    if numeric_tolerance:
        correct, pred, _matched = _finqa_verify(solution_str, answer)
        return correct, pred

    correct, pred = is_correct_strict_box(solution_str, answer, pause_tokens_index)
    if pred is None:
        pred = ""

    # try Math-Verify equivalence check
    if not correct and pred != "":
        try:
            with timeout(seconds=5):
                gold_expr = mv_parse(answer)
                pred_expr = mv_parse(pred)
                correct = mv_verify(gold_expr, pred_expr)
        except Exception:  # ignore any parsing/verification errors
            pass

    return correct, pred


def compute_score(
    solution_str: str,
    ground_truth: str,
    extra_info = None,
    pause_tokens_index: Optional[list[int]] = None,
    format_feedback: bool = True,
    correctness_feedback: bool = False,
    numeric_tolerance: bool = False,
) -> float:
    """Compute the reward score for a solution.

    Args:
        solution_str: The solution string
        ground_truth: The ground truth answer
        config: Configuration object containing reward model settings
        pause_tokens_index: Indices of pause tokens
        numeric_tolerance: routed through to verify(); enabled for data_source=="finqa".

    Returns:
        Reward score (1.0 for correct, 0 for incorrect)
    """
    split = extra_info.get("split", "test")
    was_truncated = extra_info.get("truncated", False)

    # Verify the solution. FinQA routes to Anne's 3-tier extractor, which also returns a
    # `matched` flag (False iff no numeric pattern was found at all) — used for the
    # incorrect_format signal so logging stays meaningful under the looser verifier.
    if numeric_tolerance:
        correct, pred, matched = _finqa_verify(solution_str, ground_truth)
        incorrect_format = not matched
    else:
        correct, pred = verify(solution_str, ground_truth, pause_tokens_index,
                               numeric_tolerance=numeric_tolerance)
        incorrect_format = pred is None or pred == ""

    reward = 1.0 if correct else 0.0
    score = reward
    was_truncated = extra_info.get("truncated", False)
    if FORMAT_PENALTY and split == "train" and incorrect_format and (not was_truncated):
        score -= 0.5

    # Generate explicit feedback for format errors (analogous to code feedback)
    feedback = ""
    if incorrect_format and not was_truncated and format_feedback:
        if numeric_tolerance:
            feedback = ("Your answer had the wrong format. Please provide the final "
                        "numerical answer, e.g. `\\boxed{...}`, `#### ...`, or "
                        "`Answer: ...`.")
        else:
            feedback = "Your answer had the wrong format. The solution must be given in the format: \\boxed{your_answer}."
    elif was_truncated and format_feedback:
        feedback = "Your response was truncated because it exceeded the maximum length."
    elif not correct and correctness_feedback:
        feedback = f"Your answer is incorrect. The correct answer is {ground_truth}."

    return {
        "score": score,
        "acc": reward,
        "pred": pred,
        "incorrect_format": 1 if incorrect_format else 0,
        "truncated": 1 if was_truncated else 0,
        "truncated_and_missing_answer": 1 if incorrect_format and was_truncated else 0,
        "feedback": feedback,
    }
