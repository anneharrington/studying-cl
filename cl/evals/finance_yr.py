"""finance_yr eval — yearly 10-K forward-sentiment continual-learning task.

Reads the SDPO finance bundle's per-year parquets directly so GEPA scores on
the same 50 val rows the RL runs (SFT/SDFT/GRPO/SDPO) evaluated on. Apples-
to-apples: same prompts, same gold labels, same regex extraction (mirrors
verl/utils/reward_score/feedback/finance.py).

Layout expected at `path` (the directory):
    task<i>_y<YYYY>_train.parquet   — 500 rows
    val_y<YYYY>.parquet             — 50 rows

`path` resolves relative to the repository root if not absolute.
"""

import os
import random
import re
from pathlib import Path

import dspy
import pyarrow.parquet as pq


_LABEL_RE = re.compile(r"\b(up|down)\b", re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _user_text(prompt) -> str:
    """Pull the user-message content out of a verl PPO-format `prompt` column.

    The parquets store `prompt` as a list of {role, content} dicts (system
    first, user second). Numpy/Arrow may surface it as ndarray-of-dicts.
    """
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    user_chunks = [m.get("content", "") for m in prompt
                   if isinstance(m, dict) and m.get("role") == "user"]
    return "\n\n".join(user_chunks).strip()


def _resolve_train_path(dir_path: Path, year: int) -> Path:
    """The bundle uses task<i>_y<YYYY>_train.parquet where i is 1..6 for 2015..2020.

    We don't want to hard-code i; just glob for the year.
    """
    matches = sorted(dir_path.glob(f"task*_y{year}_train.parquet"))
    if not matches:
        raise FileNotFoundError(
            f"No train parquet found for year {year} in {dir_path} "
            f"(expected task<i>_y{year}_train.parquet)"
        )
    if len(matches) > 1:
        print(f"  Warning: multiple train parquets for year {year}: {matches}; using {matches[0]}")
    return matches[0]


def _read_rows(parquet_path: Path, max_context_chars: int):
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    out = []
    for _, row in df.iterrows():
        filing_text = _user_text(row["prompt"])
        if max_context_chars and len(filing_text) > max_context_chars:
            filing_text = filing_text[:max_context_chars]
        rm = row["reward_model"]
        gold = rm.get("ground_truth", "") if isinstance(rm, dict) else ""
        ei = row["extra_info"] if "extra_info" in row else {}
        if not isinstance(ei, dict):
            ei = dict(ei) if ei is not None else {}
        out.append({
            "filing_text": filing_text,
            "answer": str(gold).strip().lower(),
            "year": int(ei.get("year", 0)) if ei.get("year") else None,
            "ticker": ei.get("cik", "") or "",
            "company": ei.get("company", "") or "",
            "filing_date": ei.get("filing_date", "") or "",
            "horizon": ei.get("horizon", "") or "",
        })
    return out


def load_finance_yr_raw(path, train_n, val_n, seed=42, eval_n=0,
                        year_filter=None, max_context_chars=50000):
    """Load finance bundle parquets for one year. Returns (train, val) or (train, val, eval).

    `train_n + val_n` rows are sampled from `task<i>_y<Y>_train.parquet` (500 rows).
    `eval_n` rows are taken from `val_y<Y>.parquet` (50 rows total). eval_n=0
    skips the eval split and the runner falls back to val_set.
    """
    if year_filter is None:
        raise ValueError("finance_yr loader requires `year_filter` (one of 2015..2020).")
    year = int(year_filter)

    dir_path = Path(path)
    if not dir_path.is_absolute():
        dir_path = Path(__file__).resolve().parents[2] / dir_path
    if not dir_path.exists():
        raise FileNotFoundError(f"finance_yr data dir not found: {dir_path}")

    train_path = _resolve_train_path(dir_path, year)
    val_path = dir_path / f"val_y{year}.parquet"
    if not val_path.exists():
        raise FileNotFoundError(f"val parquet not found: {val_path}")

    train_rows = _read_rows(train_path, max_context_chars)
    held_eval_rows = _read_rows(val_path, max_context_chars)

    rng = random.Random(seed)
    rng.shuffle(train_rows)

    train_set = train_rows[:train_n]
    val_set = (train_rows[train_n:] if val_n < 0
               else train_rows[train_n : train_n + val_n])

    print(f"  Loaded finance_yr year={year}: "
          f"{len(train_set)} train, {len(val_set)} val (from {train_path.name}); "
          f"{len(held_eval_rows)} eval pool (from {val_path.name})")

    if eval_n == 0:
        return train_set, val_set

    if eval_n < 0 or eval_n >= len(held_eval_rows):
        eval_set = held_eval_rows
    else:
        eval_rng = random.Random(seed + 1)
        eval_rng.shuffle(held_eval_rows)
        eval_set = held_eval_rows[:eval_n]
    return train_set, val_set, eval_set


def _to_dspy(items):
    return [
        dspy.Example(
            filing_text=item["filing_text"],
            answer=str(item["answer"]),
            year=item["year"],
            ticker=item["ticker"],
            company=item["company"],
            filing_date=item["filing_date"],
            horizon=item["horizon"],
        ).with_inputs("filing_text")
        for item in items
    ]


def load_finance_yr(path, train_n, val_n, seed=42, eval_n=0,
                    year_filter=None, max_context_chars=50000):
    """DSPy-Example wrapper around load_finance_yr_raw."""
    splits = load_finance_yr_raw(
        path, train_n, val_n, seed=seed, eval_n=eval_n,
        year_filter=year_filter, max_context_chars=max_context_chars,
    )
    return tuple(_to_dspy(s) for s in splits)


def _strip_chat_tail(s: str) -> str:
    s = _THINK_RE.sub("", s)
    for term in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
        idx = s.find(term)
        if idx != -1:
            s = s[:idx]
    return s


def _parse_label(text):
    """Match verl finance.py: drop <think>/chat tail, then first up|down hit."""
    if not text:
        return None
    t = _strip_chat_tail(text)
    m = _LABEL_RE.search(t)
    return m.group(1).lower() if m else None


def finance_yr_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """1.0 if predicted label matches gold (up/down), else 0.0.

    Mirrors verl/utils/reward_score/feedback/finance.py:_parse_sentiment_label
    so a GEPA score and an RL `mean@N` score reduce to identical 0/1 outcomes
    on the same example.
    """
    gold = (example.answer or "").strip().lower()
    pred_text = getattr(prediction, "answer", "") or ""
    pred = _parse_label(pred_text)

    if pred is None:
        return dspy.Prediction(
            score=0.0,
            feedback=(
                f"No label found in output. Return one token: 'up' or 'down'. "
                f"Gold answer: {gold}. Output was: {pred_text[:200]!r}"
            ),
        )

    correct = (pred == gold)
    score = 1.0 if correct else 0.0
    feedback = (
        f"Correct. Gold answer: {gold}."
        if correct
        else f"Incorrect. Predicted '{pred}', gold answer is '{gold}'."
    )
    return dspy.Prediction(score=score, feedback=feedback)
