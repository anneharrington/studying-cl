#!/usr/bin/env python3
"""Convert our finance ordering-F train parquets into the per-year text corpora
that Cartridges' SelfStudySynthesizer consumes.

Per phase i (year y_i in 2015..2020) we want a single text file containing the
concatenated bodies of the 500 train 10-K filings for that year. The synthesizer
chunks this into 512-1024 token chunks and asks Qwen3-8B to self-study them.

Inputs (read):
    /workspace/home/nayan/finance_data/cl_yearly/train_y{2015..2020}.parquet
    Each row has `prompt` = [{role:system, ...}, {role:user, content: <filing body>}]
    where the user message contains "[START OF FILING]\\n<body>\\n[END OF FILING]".

Outputs (write):
    <out>/y{YYYY}/corpus.txt              one text file per year (the synth target)
    <out>/y{YYYY}/manifest.json           per-year provenance (n_docs, n_chars, n_tokens)
    <out>/index.json                      top-level index across all 6 years

Usage:
    python /home/nayan/scripts/cartridges_finance/data_adapter.py
    python /home/nayan/scripts/cartridges_finance/data_adapter.py --years 2015 2016
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow.parquet as pq

DEFAULT_DATA_DIR = Path("/workspace/home/nayan/finance_data/cl_yearly")
DEFAULT_OUT_DIR = Path("/workspace/home/nayan/cartridges_finance/corpora")
DEFAULT_YEARS = [2015, 2016, 2017, 2018, 2019, 2020]

# Pull the body between [START OF FILING] and [END OF FILING] markers our prep
# script inserted (see scripts/data/prep_finance_yearly.py:139-141).
_BODY_RE = re.compile(r"\[START OF FILING\]\n(.*?)\n\[END OF FILING\]", re.DOTALL)


def _extract_body(user_content: str) -> str | None:
    m = _BODY_RE.search(user_content)
    return m.group(1).strip() if m else None


def build_year_corpus(parquet_path: Path, year: int, out_dir: Path) -> dict:
    """Concatenate all train 10-K bodies for one year into a single text file.

    Each filing is wrapped in a clear delimiter so the synthesizer's chunker
    can keep filings reasonably separable but still allow cross-filing
    questions (the temporal-drift signal is *across* filings, not per).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(parquet_path)
    rows = table.to_pylist()

    n_docs = 0
    n_dropped = 0
    total_chars = 0
    text_parts: list[str] = []
    for r in rows:
        prompt_msgs = r["prompt"]
        if hasattr(prompt_msgs, "tolist"):
            prompt_msgs = prompt_msgs.tolist()
        user_msg = next((m for m in prompt_msgs if dict(m).get("role") == "user"), None)
        if user_msg is None:
            n_dropped += 1; continue
        body = _extract_body(dict(user_msg)["content"])
        if not body:
            n_dropped += 1; continue
        ei = dict(r["extra_info"])
        company = str(ei.get("company", ""))
        date = str(ei.get("filing_date", ""))
        gold = dict(r["reward_model"]).get("ground_truth", "")
        # Doc header carries metadata so the synthesizer can ask "by date / by company"
        # questions and we keep the answer key (up/down) in scope.
        header = (f"\n=== 10-K filing  company={company}  filing_date={date}  "
                  f"forward_30d={gold} ===\n")
        text_parts.append(header)
        text_parts.append(body)
        text_parts.append("\n")
        n_docs += 1
        total_chars += len(body)

    text = "".join(text_parts)
    corpus_path = out_dir / "corpus.txt"
    corpus_path.write_text(text, encoding="utf-8")

    # Rough token estimate (~3.75 chars/token for English 10-Ks).
    est_tokens = total_chars // 4

    manifest = {
        "year": year,
        "data_source": f"finance_yr_{year}",
        "source_parquet": str(parquet_path),
        "n_docs_kept": n_docs,
        "n_docs_dropped": n_dropped,
        "total_chars": total_chars,
        "est_tokens": est_tokens,
        "corpus_path": str(corpus_path),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  y{year}: {n_docs} docs kept, {n_dropped} dropped, "
          f"{total_chars/1e6:.2f}M chars, ~{est_tokens/1e6:.2f}M tokens "
          f"-> {corpus_path}", file=sys.stderr)
    return manifest


def build_cumulative_corpus(per_year_dir: Path, years_through: list[int],
                             phase_idx: int, out_dir: Path,
                             seed: int = 42) -> dict:
    """Build the cumulative corpus for phase `phase_idx`: union of all per-year
    corpora through year y_i, with documents shuffled so KVFromText's "first
    32k tokens" init is a balanced sample across years (not biased toward y_1).

    Each per-year corpus.txt is split on the doc-header marker `=== 10-K filing`
    so we shuffle at the doc level, not character level.
    """
    import random
    rng = random.Random(seed)

    docs: list[str] = []
    n_total_chars = 0
    for y in years_through:
        cp = per_year_dir / f"y{y}" / "corpus.txt"
        if not cp.exists():
            print(f"  !! cumulative: missing {cp}", file=sys.stderr); continue
        text = cp.read_text(encoding="utf-8")
        # Per-year corpus is "\n=== 10-K filing ... ===\n<body>\n\n=== ... ===\n<body>\n…"
        # Split on the header line, restoring it at the front of each chunk.
        parts = text.split("\n=== 10-K filing ")
        # Skip empty leading split element; prepend marker back to non-empty parts.
        for part in parts:
            part = part.strip()
            if not part: continue
            doc = "\n=== 10-K filing " + part if not part.startswith("===") else "\n" + part
            docs.append(doc)
            n_total_chars += len(doc)
    rng.shuffle(docs)

    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = out_dir / "corpus.txt"
    corpus_path.write_text("\n".join(docs), encoding="utf-8")
    manifest = {
        "phase_idx": phase_idx,
        "years_through": years_through,
        "n_docs": len(docs),
        "total_chars": n_total_chars,
        "est_tokens": n_total_chars // 4,
        "shuffled_seed": seed,
        "corpus_path": str(corpus_path),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  cumulative phase {phase_idx} (years {years_through}): "
          f"{len(docs)} docs, {n_total_chars/1e6:.2f}M chars -> {corpus_path}",
          file=sys.stderr)
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                    help="Directory containing train_y{year}.parquet files.")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                    help="Output root for per-year corpora.")
    ap.add_argument("--years", type=int, nargs="+", default=DEFAULT_YEARS,
                    help="Years to build corpora for.")
    ap.add_argument("--cumulative", action="store_true",
                    help="Also build cumulative_y{i}/corpus.txt where i is each "
                         "phase position (1..N). cumulative_y{i} = shuffled union "
                         "of y{years[0]..years[i-1]}. Used by the cumulative-cartridge CL recipe.")
    ap.add_argument("--seed", type=int, default=42, help="Shuffle seed for cumulative corpora.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    index = {"years": [], "cumulative": [], "out_dir": str(args.out_dir)}
    for y in args.years:
        p = args.data_dir / f"train_y{y}.parquet"
        if not p.exists():
            print(f"!! missing {p}", file=sys.stderr); continue
        m = build_year_corpus(p, y, args.out_dir / f"y{y}")
        index["years"].append({"year": y, "manifest": str(args.out_dir / f"y{y}/manifest.json"),
                               "corpus": m["corpus_path"], "n_docs": m["n_docs_kept"],
                               "est_tokens": m["est_tokens"]})

    if args.cumulative:
        for i, y in enumerate(args.years, 1):
            cm = build_cumulative_corpus(
                per_year_dir=args.out_dir,
                years_through=args.years[:i],
                phase_idx=i,
                out_dir=args.out_dir / f"cumulative_y{i}",
                seed=args.seed,
            )
            index["cumulative"].append({"phase_idx": i, "manifest": str(
                args.out_dir / f"cumulative_y{i}/manifest.json"), **cm})

    (args.out_dir / "index.json").write_text(json.dumps(index, indent=2))
    print(f"\nwrote index -> {args.out_dir/'index.json'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
