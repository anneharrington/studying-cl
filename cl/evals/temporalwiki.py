"""temporalwiki eval — TemporalWiki drift Q&A continual-learning task.

Reads the SDPO `cl_drift_data/` parquet bundle directly so GEPA scores on the
same 50 val rows per slice the RL runs (SFT/SDFT/GRPO/SDPO) evaluated on.
The metric mirrors verl/utils/reward_score/feedback/temporalwiki.py — SQuAD-
style normalized whitespace-token F1, with `acc = (F1 >= 0.5)` as the binary
headline number. The continuous F1 is included in the metric's feedback string
so GEPA's reflection LM has a graded signal during prompt evolution.

Layout expected at `path` (the directory):
    train_s{1,2,3}.parquet     — 500 rows each, drift training slices
    val_s{1,2,3}.parquet       — 50 rows each, drift eval slices
    val_stable.parquet         — 50 rows, stable knowledge probe (no drift,
                                  no train shard; loader returns empty
                                  train/val and only an eval split)
    manifest.json              — slice metadata (system prompt, chronology)

`path` resolves relative to the repository root if not absolute.

Slices selected via `slice_filter`:
    "s1" | "s2" | "s3"   — drift training+eval slice
    "stable"             — stable knowledge eval-only slice
"""
from __future__ import annotations

import os
import random
import re
import string
from collections import Counter
from pathlib import Path

import dspy
import pyarrow.parquet as pq


_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_WS = re.compile(r"\s+")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# F1 >= this counts as "correct" for the binary acc field. Mirrors verl
# temporalwiki.py:ACC_THRESHOLD; lets paraphrase-equivalent answers through
# (e.g. "University of Iowa" vs "Iowa University", F1 ~= 0.67) while
# rejecting actually-wrong answers.
ACC_THRESHOLD = 0.5


def _user_text(prompt) -> str:
    """Pull the user-message content out of a verl PPO-format `prompt` column."""
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    user_chunks = [m.get("content", "") for m in prompt
                   if isinstance(m, dict) and m.get("role") == "user"]
    return "\n\n".join(user_chunks).strip()


def _read_rows(parquet_path: Path):
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    out = []
    for _, row in df.iterrows():
        question = _user_text(row["prompt"])
        rm = row["reward_model"]
        gold = rm.get("ground_truth", "") if isinstance(rm, dict) else ""
        ei = row["extra_info"] if "extra_info" in row else {}
        if not isinstance(ei, dict):
            ei = dict(ei) if ei is not None else {}
        out.append({
            "question": question,
            "answer": str(gold).strip(),
            "subject": ei.get("subject", "") or "",
            "relation": ei.get("relation", "") or "",
            "slice_old": ei.get("slice_old", "") or "",
            "slice_new": ei.get("slice_new", "") or "",
            "slice_label_short": ei.get("slice_label_short", "") or "",
            "slice_tag": ei.get("slice_tag", "") or "",
        })
    return out


def _resolve_paths(dir_path: Path, slice_filter: str):
    """Return (train_path | None, val_path) for `slice_filter`.

    Raises FileNotFoundError if the expected files are missing.
    """
    sf = slice_filter.lower()
    if sf in ("s1", "s2", "s3"):
        train = dir_path / f"train_{sf}.parquet"
        val = dir_path / f"val_{sf}.parquet"
        for p in (train, val):
            if not p.exists():
                raise FileNotFoundError(f"temporalwiki parquet not found: {p}")
        return train, val
    if sf == "stable":
        val = dir_path / "val_stable.parquet"
        if not val.exists():
            raise FileNotFoundError(f"temporalwiki parquet not found: {val}")
        return None, val
    raise ValueError(
        f"slice_filter must be one of s1/s2/s3/stable, got {slice_filter!r}"
    )


def load_temporalwiki_raw(path, train_n, val_n, seed=42, eval_n=0,
                          slice_filter=None, max_context_chars=None):
    """Load TemporalWiki bundle parquets for one slice. Returns (train, val) or
    (train, val, eval).

    For drift slices (s1/s2/s3): `train_n + val_n` rows are sampled (seeded)
    from `train_s<i>.parquet` (500 rows). `eval_n` rows are taken from
    `val_s<i>.parquet` (50 rows).

    For "stable": no train shard exists. Returns ([], [], eval_set), with
    `eval_set` = first `eval_n` rows of `val_stable.parquet` (or all 50 if
    eval_n >= 50). `train_n` and `val_n` are ignored.

    `max_context_chars` is accepted for interface symmetry with finance_yr
    but unused here — TemporalWiki questions are short ("subject relation").
    """
    if slice_filter is None:
        raise ValueError("temporalwiki loader requires `slice_filter` (s1|s2|s3|stable).")
    sf = str(slice_filter).lower()
    del max_context_chars  # unused; kept in signature for cfg pass-through

    dir_path = Path(path)
    if not dir_path.is_absolute():
        dir_path = Path(__file__).resolve().parents[2] / dir_path
    if not dir_path.exists():
        raise FileNotFoundError(f"temporalwiki data dir not found: {dir_path}")

    train_path, val_path = _resolve_paths(dir_path, sf)

    if sf == "stable":
        eval_pool = _read_rows(val_path)
        if eval_n == 0:
            print(f"  Loaded temporalwiki slice=stable: 0 train, 0 val (eval-only); "
                  f"{len(eval_pool)} eval pool (from {val_path.name})")
            return [], []
        if eval_n < 0 or eval_n >= len(eval_pool):
            eval_set = eval_pool
        else:
            rng = random.Random(seed + 1)
            rng.shuffle(eval_pool)
            eval_set = eval_pool[:eval_n]
        print(f"  Loaded temporalwiki slice=stable: 0 train, 0 val; "
              f"{len(eval_set)} eval (from {val_path.name})")
        return [], [], eval_set

    train_rows = _read_rows(train_path)
    held_eval_rows = _read_rows(val_path)

    rng = random.Random(seed)
    rng.shuffle(train_rows)

    train_set = train_rows[:train_n]
    val_set = (train_rows[train_n:] if val_n < 0
               else train_rows[train_n : train_n + val_n])

    print(f"  Loaded temporalwiki slice={sf}: "
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
            question=item["question"],
            answer=str(item["answer"]),
            subject=item["subject"],
            relation=item["relation"],
            slice_old=item["slice_old"],
            slice_new=item["slice_new"],
            slice_label_short=item["slice_label_short"],
            slice_tag=item["slice_tag"],
        ).with_inputs("question")
        for item in items
    ]


def load_temporalwiki(path, train_n, val_n, seed=42, eval_n=0,
                      slice_filter=None, max_context_chars=None):
    """DSPy-Example wrapper around load_temporalwiki_raw."""
    splits = load_temporalwiki_raw(
        path, train_n, val_n, seed=seed, eval_n=eval_n,
        slice_filter=slice_filter, max_context_chars=max_context_chars,
    )
    return tuple(_to_dspy(s) for s in splits)


# ---------------------------------------------------------------------------
# Metric — mirrors verl/utils/reward_score/feedback/temporalwiki.py exactly so
# GEPA acc and RL `mean@N` reduce to identical 0/1 outcomes per row.
# ---------------------------------------------------------------------------

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
    solution = _THINK_RE.sub("", solution)
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


def temporalwiki_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """Binary acc (F1 >= 0.5) on the predicted object string vs. gold.

    Score is binary so it matches the headline `val-core/<source>/acc/mean@N`
    metric the SDPO bundle reports. The continuous F1 is surfaced in the
    feedback string so GEPA's reflection LM sees a graded signal even on
    "wrong" attempts (e.g. F1=0.4 → "close but rejected", F1=0.0 → "totally
    off").
    """
    gold = (example.answer or "").strip()
    pred_text = getattr(prediction, "answer", "") or ""
    pred = _extract_answer(pred_text)
    f1 = _f1(pred, gold)
    acc = 1.0 if f1 >= ACC_THRESHOLD else 0.0

    if acc == 1.0:
        feedback = f"Correct (F1={f1:.2f}). Gold answer: {gold}."
    elif not pred:
        feedback = (
            f"No answer extracted. Output ONLY the object value as a short "
            f"plain-text string. Gold: {gold!r}."
        )
    else:
        feedback = (
            f"Object mismatch: predicted {pred!r}, expected {gold!r} "
            f"(F1={f1:.2f}, threshold {ACC_THRESHOLD}). Output ONLY the "
            f"object value as a short plain-text string."
        )

    return dspy.Prediction(score=acc, feedback=feedback)
