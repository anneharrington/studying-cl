"""Sentiment10K eval — temporal-drift dataset built from FinanceBench 10-Ks.

Each example is a 10-K filing's text plus an `up`/`down` label derived from
the company's actual stock movement over the configured horizon. Examples are
indexed by filing year, so the runner can iterate stages year-by-year and
plot past-vs-future generalization.

Data is built by `cl/finance/build_data/build_sentiment10k_data.py`, which
writes `data/sentiment10k/sentiment10k.json` — a list of records with
{id, doc_name, year, company_name, ticker, filing_date, direction, horizon,
 question, text_path, context_char_limit, answer}. Filing text is stored
separately at `text_path`; the loader reads and truncates it on demand.
"""

import json
import os
import random
import re

import dspy


_LABEL_RE = re.compile(r"\b(up|down)\b", re.IGNORECASE)


def _read_filing_text(text_path: str, max_chars: int) -> str:
    if not os.path.exists(text_path):
        return ""
    with open(text_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    if max_chars and len(text) > max_chars:
        text = text[:max_chars]
    return text


def _to_example(item, max_context_chars):
    filing_text = _read_filing_text(item["text_path"], max_context_chars)
    return {
        "doc_name": item["doc_name"],
        "year": int(item["year"]) if item.get("year") is not None else None,
        "ticker": item.get("ticker"),
        "filing_date": item.get("filing_date"),
        "direction": item.get("direction"),
        "horizon": item.get("horizon"),
        "filing_text": filing_text,
        "answer": item["answer"],
    }


def load_sentiment10k_raw(path, train_n, val_n, seed=42, eval_n=0,
                          year_filter=None, max_context_chars=50000):
    """Load sentiment10k.json, filter by year, seeded shuffle, split."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if year_filter is not None:
        year_int = int(year_filter)
        data = [r for r in data if int(r.get("year", -1)) == year_int]

    examples = [_to_example(r, max_context_chars) for r in data]
    examples = [ex for ex in examples if ex["filing_text"]]
    print(f"  Loaded {len(examples)} sentiment10k examples"
          f"{' (year=' + str(year_filter) + ')' if year_filter is not None else ''} from {path}")

    random.Random(seed).shuffle(examples)
    train_set = examples[:train_n]
    val_set = examples[train_n:] if val_n < 0 else examples[train_n : train_n + val_n]

    if eval_n == 0:
        return train_set, val_set

    eval_start = train_n + (len(examples) - train_n if val_n < 0 else val_n)
    eval_set = examples[eval_start:] if eval_n < 0 else examples[eval_start : eval_start + eval_n]
    return train_set, val_set, eval_set


def load_sentiment10k(path, train_n, val_n, seed=42, eval_n=0,
                      year_filter=None, max_context_chars=50000):
    """Load sentiment10k as dspy.Example splits keyed on `filing_text`."""
    splits = load_sentiment10k_raw(
        path, train_n, val_n, seed, eval_n,
        year_filter=year_filter, max_context_chars=max_context_chars,
    )

    def to_dspy(items):
        return [
            dspy.Example(
                doc_name=item["doc_name"],
                year=item["year"],
                ticker=item["ticker"] or "",
                filing_date=item["filing_date"] or "",
                direction=item["direction"] or "",
                horizon=item["horizon"] or "",
                filing_text=item["filing_text"],
                answer=str(item["answer"]),
            ).with_inputs("filing_text")
            for item in items
        ]

    return tuple(to_dspy(s) for s in splits)


def _parse_label(text):
    if not text:
        return None
    m = _LABEL_RE.search(text)
    return m.group(1).lower() if m else None


def sentiment10k_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """1.0 if predicted label matches gold (up/down), else 0.0.

    Accepts predictions like 'up', 'Down.', 'I think the stock will go up',
    by extracting the first 'up' or 'down' token. Returns 0 with a corrective
    feedback string when no label is found, so GEPA's reflection LM has
    something to learn from.
    """
    gold = (example.answer or "").strip().lower()
    pred_text = getattr(prediction, "answer", "") or ""
    pred = _parse_label(pred_text)

    if pred is None:
        return dspy.Prediction(
            score=0.0,
            feedback=(
                f"No label found in output. Expected exactly one token: 'up' or 'down'. "
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
