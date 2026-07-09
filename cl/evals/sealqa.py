"""SealQA loader + metric.

SealQA is a year-stamped open-ended QA dataset (vtllms/sealqa on HF). Each
example pairs a `question` with an `answer` and a set of context documents
(`golds`, `12_docs`, `30_docs` lists). The dataset includes a `freshness`
field; the colleague's setup uses only `freshness == "fast-changing"` rows
since those are the ones whose correct answer depends on `effective_year`.

Loader semantics mirror the other per-year tasks (sentiment10k):
  * dataset.path is the directory of parquet files
  * dataset.year_filter (optional) restricts to a single effective_year
  * dataset.{max_docs,max_doc_chars,max_context_chars} bound context size

Scoring is contains-style: correct iff normalized gold is a substring of
normalized prediction. Token-F1 and judge modes are available in the
colleague's standalone driver if a stricter signal is needed later.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import dspy

_NORM_RE = re.compile(r"\s+")
_NORM_KEEP = re.compile(r"[^a-z0-9$%\.\- ]+")
_YEAR_RE = re.compile(r"(19|20)\d{2}")


def _norm_text(s):
    if not s:
        return ""
    s = str(s).lower().strip()
    s = _NORM_RE.sub(" ", s)
    s = _NORM_KEEP.sub("", s)
    return s


def _parse_year(raw):
    if raw is None:
        return None
    m = _YEAR_RE.search(str(raw).strip())
    return int(m.group(0)) if m else None


def _format_question(row, *, max_docs, max_doc_chars, max_context_chars):
    """Build the colleague's task input: year + question + concatenated docs."""
    year = _parse_year(row.get("effective_year"))
    question = str(row.get("question", "")).strip()

    docs = []
    for k in ("golds", "12_docs", "30_docs"):
        for d in (row.get(k) or []):
            if isinstance(d, dict):
                docs.append(d)

    parts = []
    total = 0
    for i, d in enumerate(docs[:max_docs], start=1):
        title = str(d.get("title", "")).strip()
        date = str(d.get("date", "")).strip()
        text = str(d.get("text", "")).strip()
        if max_doc_chars > 0:
            text = text[:max_doc_chars]
        block = f"[DOC {i}]\nDate: {date}\nTitle: {title}\nContent:\n{text}\n"
        if max_context_chars > 0 and total + len(block) > max_context_chars:
            break
        parts.append(block)
        total += len(block)

    header = f"The current year is {year}.\n" if year is not None else ""
    return (
        f"{header}"
        f"Question: {question}\n\n"
        "Context documents:\n"
        + "\n".join(parts)
    )


def _to_example(row, *, max_docs, max_doc_chars, max_context_chars):
    if str(row.get("freshness", "")).lower() != "fast-changing":
        return None
    question = str(row.get("question", "")).strip()
    answer = str(row.get("answer", "")).strip()
    if not question or not answer:
        return None
    formatted = _format_question(
        row,
        max_docs=max_docs,
        max_doc_chars=max_doc_chars,
        max_context_chars=max_context_chars,
    )
    if not formatted.strip():
        return None
    return {
        "question": formatted,
        "answer": answer,
        "task_type": "qa",
        "effective_year": _parse_year(row.get("effective_year")),
    }


def _read_parquet_rows(path):
    import pyarrow.parquet as pq

    files = sorted(Path(path).rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {path}")
    rows = []
    for p in files:
        rows.extend(pq.read_table(p).to_pylist())
    return rows


def load_sealqa_raw(path, train_n, val_n, seed=42, eval_n=0,
                    year_filter=None,
                    max_docs=12, max_doc_chars=4000, max_context_chars=50000):
    """Load SealQA parquet files and split into train/val(/eval) as plain dicts.

    Filters to `freshness == "fast-changing"` and optionally to a single
    `effective_year`. Same shuffle-then-slice semantics as sentiment10k.
    """
    rows = _read_parquet_rows(path)

    if year_filter is not None:
        year_int = int(year_filter)
        rows = [r for r in rows if _parse_year(r.get("effective_year")) == year_int]

    examples = [
        e for e in (
            _to_example(
                r,
                max_docs=max_docs,
                max_doc_chars=max_doc_chars,
                max_context_chars=max_context_chars,
            ) for r in rows
        ) if e is not None
    ]
    print(f"  Loaded {len(examples)} sealqa examples"
          f"{' (year=' + str(year_filter) + ')' if year_filter is not None else ''} from {path}")

    random.Random(seed).shuffle(examples)
    train_set = examples[:train_n]
    val_set = examples[train_n:] if val_n < 0 else examples[train_n : train_n + val_n]

    if eval_n == 0:
        return train_set, val_set

    eval_start = train_n + (len(examples) - train_n if val_n < 0 else val_n)
    eval_set = examples[eval_start:] if eval_n < 0 else examples[eval_start : eval_start + eval_n]
    return train_set, val_set, eval_set


def load_sealqa(path, train_n, val_n, seed=42, eval_n=0,
                year_filter=None,
                max_docs=12, max_doc_chars=4000, max_context_chars=50000):
    """Load SealQA as dspy.Example splits keyed on `question`."""
    splits = load_sealqa_raw(
        path, train_n, val_n, seed, eval_n,
        year_filter=year_filter,
        max_docs=max_docs,
        max_doc_chars=max_doc_chars,
        max_context_chars=max_context_chars,
    )

    def to_dspy(items):
        return [
            dspy.Example(
                question=item["question"],
                answer=str(item["answer"]),
            ).with_inputs("question")
            for item in items
        ]

    return tuple(to_dspy(s) for s in splits)


def _extract_answer_text(text):
    """Pull the part after 'Answer:' if present, else return full text.

    Contains-style scoring is forgiving — keeping the full response avoids
    over-restricting; the 'Answer:' grab is just to make feedback strings
    more readable when the model uses that convention.
    """
    if not text:
        return ""
    m = re.search(r"(?im)^\s*answer\s*:\s*(.+)$", text)
    if m:
        return m.group(1).strip()
    return text.strip()


def sealqa_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """1.0 iff normalized gold is a substring of normalized prediction.

    Mirrors the contains-style scorer in cl/finance/gepa_sequential_sealqa.py.
    Returns a dspy.Prediction with score + corrective feedback so GEPA's
    reflection LM has something to work with on incorrect cases.
    """
    gold = (example.answer or "").strip()
    pred_text = getattr(prediction, "answer", "") or ""
    if not pred_text and prediction is not None:
        pred_text = str(prediction)

    if not gold or not pred_text:
        return dspy.Prediction(
            score=0.0,
            feedback=(
                f"Empty {'gold' if not gold else 'prediction'}. "
                f"Gold answer: {gold!r}."
            ),
        )

    correct = _norm_text(gold) in _norm_text(pred_text)
    score = 1.0 if correct else 0.0
    feedback = (
        f"Correct. Gold answer: {gold}."
        if correct
        else f"Incorrect. Gold answer: {gold}. "
             f"Predicted (first 200 chars): {pred_text[:200]!r}"
    )
    return dspy.Prediction(score=score, feedback=feedback)
