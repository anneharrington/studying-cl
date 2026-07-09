"""Reward scorer for 10-K forward sentiment (JanosAudran/financial-reports-sec).

Continual-learning setup: per-year filing excerpts; the model predicts the
forward 30-day stock direction. Gold is "up" or "down" (mapped from the
dataset's positive/negative pre-computed labels at the 30d horizon).

Metric: exact-match on {up, down} after light extraction.

Robust extraction (same shape as temporalwiki.py):
    - drop <think>...</think> if present (Qwen3 thinking mode)
    - cut at <|im_end|> / <|endoftext|> / <|im_start|> so trailing chatter
      doesn't pollute extraction
    - match `\\bup\\b` first, `\\bdown\\b` second.

Returns dict matching the tooluse / temporalwiki scorers' shape:
    {score, acc, pred, incorrect_format, truncated, truncated_and_missing_answer, feedback}

acc == score (binary) so GRPO/SDPO/SDFT all see the same signal and the
val-core/<source>/acc/mean@N keys downstream tooling consumes are produced
in the same way as for tooluse.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

_UP = re.compile(r"\bup\b", re.IGNORECASE)
_DOWN = re.compile(r"\bdown\b", re.IGNORECASE)

# When set, every call to compute_score appends one JSONL line per example to
# $FINANCE_DUMP_DIR/<data_source>.jsonl. Each verl rollout (n=8) calls this
# scorer once, so we record one row per (prompt, rollout). Used by the
# offline prediction-dump pass; unset for normal training runs.
_DUMP_DIR_ENV = "FINANCE_DUMP_DIR"


def _strip_chat_tail(s: str) -> str:
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL)
    for term in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
        idx = s.find(term)
        if idx != -1:
            s = s[:idx]
    return s


def _parse_sentiment_label(text: str) -> str:
    """Returns 'up', 'down', or '' if neither is found."""
    if not isinstance(text, str):
        return ""
    t = _strip_chat_tail(text)
    if _UP.search(t):
        return "up"
    if _DOWN.search(t):
        return "down"
    return ""


def compute_score(solution: str, ground_truth, extra_info: dict | None = None,
                  data_source: str = "") -> dict:
    if isinstance(ground_truth, (list, tuple)) and ground_truth:
        ground_truth = ground_truth[0]
    if isinstance(ground_truth, bytes):
        ground_truth = ground_truth.decode("utf-8", errors="replace")
    gold = (str(ground_truth) if ground_truth is not None else "").strip().lower()

    pred = _parse_sentiment_label(solution if isinstance(solution, str) else "")

    if pred == gold and pred in ("up", "down"):
        score = 1.0
        feedback = ""
        incorrect_format = 0
    elif not pred:
        score = 0.0
        feedback = "Return one label: up or down."
        incorrect_format = 1
    else:
        score = 0.0
        feedback = f"Incorrect label. Expected {gold}, got {pred}."
        incorrect_format = 0

    # Optional per-example dump for the offline prediction-export pass.
    dump_dir = os.environ.get(_DUMP_DIR_ENV)
    if dump_dir:
        try:
            d = Path(dump_dir); d.mkdir(parents=True, exist_ok=True)
            ds = data_source or (extra_info or {}).get("data_source", "unknown")
            row = {
                "data_source": ds,
                "gold": gold,
                "pred": pred,
                "score": score,
                "raw_solution": solution if isinstance(solution, str) else str(solution),
                "extra_info": extra_info or {},
            }
            # one JSONL per data_source so concurrent val sources don't fight
            with open(d / f"{ds}.jsonl", "a") as f:
                f.write(json.dumps(row, default=str) + "\n")
        except Exception:
            # never break training/eval if dumping fails
            pass

    return {
        "score": score,
        "acc": score,
        "pred": pred,
        "incorrect_format": incorrect_format,
        "truncated": 0,
        "truncated_and_missing_answer": 0,
        "feedback": feedback,
    }
