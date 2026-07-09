"""Reward scorer for TemporalWiki drift Q&A.

Continual-learning setup: model is asked `"{subject} {relation}"`, expected to
emit the object string (paper Appendix E format). Ground truth changes across
monthly slices for the same (subject, relation) — the temporal drift signal.

Metric: normalized F1 on whitespace tokens of the predicted vs. gold object.
Normalization mirrors the SQuAD / TemporalWiki convention:
    lowercase  ->  strip articles ('a','an','the')  ->  strip punctuation  ->
    collapse whitespace  ->  word-level F1.

Robust extraction:
    - drop <think>...</think> if present (Qwen3 thinking mode)
    - cut at <|im_end|> / <|endoftext|> / <|im_start|> so trailing chatter
      doesn't pollute the F1 score
    - prefer "Answer: <X>" if the model self-tags it; otherwise first non-empty
      line.

Returns dict matching the tooluse / mcq scorers' shape so the dispatcher and
downstream loggers don't care which task it is:
    {score, acc, pred, incorrect_format, truncated, truncated_and_missing_answer, feedback}

acc here = (F1 >= ACC_THRESHOLD), so GRPO sees clean binary advantages and the
val-core/<source>/acc/mean@N keys our analyze.py / plotter consume already-mean
behave correctly. SDPO/SDFT use the continuous F1 from `score`.
"""
from __future__ import annotations

import re
import string
from collections import Counter

# F1 >= this counts as "correct" for the binary acc field.
# 0.5 lets paraphrase-equivalent answers through (e.g. "University of Iowa" vs
# "Iowa University" → F1 ~0.67) while rejecting actually-wrong answers.
ACC_THRESHOLD = 0.5

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_WS = re.compile(r"\s+")


def _normalize(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.lower()
    s = _ARTICLES.sub(" ", s)
    s = s.translate(_PUNCT_TABLE)
    s = _WS.sub(" ", s).strip()
    return s


def _f1(pred: str, gold: str) -> float:
    p_toks = _normalize(pred).split()
    g_toks = _normalize(gold).split()
    if not p_toks or not g_toks:
        return float(p_toks == g_toks)
    common = Counter(p_toks) & Counter(g_toks)
    n = sum(common.values())
    if n == 0:
        return 0.0
    precision = n / len(p_toks)
    recall = n / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def _extract_answer(solution: str) -> str:
    if not isinstance(solution, str):
        return ""
    solution = re.sub(r"<think>.*?</think>", "", solution, flags=re.DOTALL)
    for term in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
        idx = solution.find(term)
        if idx != -1:
            solution = solution[:idx]
    m = re.search(r"(?im)^\s*answer\s*[:\-]\s*(.+)$", solution)
    if m:
        return m.group(1).strip()
    for line in solution.splitlines():
        line = line.strip()
        if line:
            return line
    return solution.strip()


def compute_score(solution: str, ground_truth, extra_info: dict | None = None) -> dict:
    if isinstance(ground_truth, (list, tuple)) and ground_truth:
        ground_truth = ground_truth[0]
    if isinstance(ground_truth, bytes):
        ground_truth = ground_truth.decode("utf-8", errors="replace")
    gold = str(ground_truth) if ground_truth is not None else ""

    pred = _extract_answer(solution)
    f1 = _f1(pred, gold)
    # twiki-easy parquets opt into a dense (continuous-F1) acc by setting
    # extra_info.continuous_reward = True. Original cl_drift_data parquets never
    # set this field → falls through to the bit-identical binary path.
    if extra_info and extra_info.get("continuous_reward"):
        acc = float(f1)
    else:
        acc = 1.0 if f1 >= ACC_THRESHOLD else 0.0

    feedback = "" if acc >= ACC_THRESHOLD else f"Object mismatch: predicted {pred!r}, expected {gold!r} (F1={f1:.2f})"

    return {
        "score": f1,
        "acc": acc,
        "pred": pred,
        "incorrect_format": 0,
        "truncated": 0,
        "truncated_and_missing_answer": 0,
        "feedback": feedback,
    }
