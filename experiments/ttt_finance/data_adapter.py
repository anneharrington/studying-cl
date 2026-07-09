#!/usr/bin/env python3
"""Convert finance year corpora into VeOmni's plaintext-iterable format.

VeOmni's recommended config uses:
    data.data_type=plaintext
    data.datasets_type=iterable
    data.text_keys=content_split

That means each shard is a JSONL where every line is a JSON object with a
`content_split` key whose value is a string (a chunk of training text). We
treat each *filing* (between `=== 10-K filing ===` markers in our existing
corpus.txt) as one JSON record so VeOmni can do per-document packing.

Reuses the per-year and cumulative corpora produced by
`scripts/cartridges_finance/data_adapter.py` (so we don't duplicate the
parquet→text extraction).

Inputs:
    /workspace/home/nayan/cartridges_finance/corpora/y{2015..2020}/corpus.txt
    /workspace/home/nayan/cartridges_finance/corpora/cumulative_y{1..6}/corpus.txt

Outputs:
    <out>/y{YYYY}/data.jsonl
    <out>/cumulative_y{i}/data.jsonl
    <out>/index.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

# cartridges_finance writes corpora INTO a run-tagged dir, not the top level.
# Default to the most recent prod run's corpora directory.
DEFAULT_SRC = Path(
    "/workspace/home/nayan/cartridges_finance/runs/"
    "finance-cl-20260504-001029-nogit_cartridges_orderF_nothink_s42/corpora"
)
DEFAULT_OUT = Path("/workspace/home/nayan/ttt_finance/data")
DEFAULT_YEARS = [2015, 2016, 2017, 2018, 2019, 2020]


def split_filings(text: str) -> Iterable[str]:
    """Split corpus.txt on the `=== 10-K filing ...` header. Yields full
    filing chunks including the header line."""
    parts = text.split("\n=== 10-K filing ")
    for i, p in enumerate(parts):
        p = p.strip()
        if not p:
            continue
        # restore the header on all but the first chunk (which already has it
        # from the data_adapter's leading newline-prefix convention)
        if i == 0 and not p.startswith("===") and "=== 10-K filing" not in p[:200]:
            yield p
        else:
            yield "=== 10-K filing " + p


def write_jsonl(corpus_path: Path, out_path: Path) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = corpus_path.read_text(encoding="utf-8")
    n_docs = 0
    n_chars = 0
    with out_path.open("w", encoding="utf-8") as f:
        for doc in split_filings(text):
            f.write(json.dumps({"content_split": doc}, ensure_ascii=False) + "\n")
            n_docs += 1
            n_chars += len(doc)
    manifest = {
        "source": str(corpus_path),
        "out": str(out_path),
        "n_docs": n_docs,
        "n_chars": n_chars,
        "est_tokens": n_chars // 4,
    }
    out_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  wrote {out_path}  n_docs={n_docs}  n_chars/1e6={n_chars/1e6:.2f}",
          file=sys.stderr)
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-dir", type=Path, default=DEFAULT_SRC,
                    help="cartridges_finance/corpora root.")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT,
                    help="ttt_finance/data root (writes JSONLs here).")
    ap.add_argument("--years", type=int, nargs="+", default=DEFAULT_YEARS)
    ap.add_argument("--cumulative", action="store_true", default=True,
                    help="Also write cumulative_y{i}/data.jsonl.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    index = {"per_year": [], "cumulative": []}

    for y in args.years:
        cp = args.src_dir / f"y{y}" / "corpus.txt"
        if not cp.exists():
            print(f"  [skip] missing {cp}", file=sys.stderr); continue
        m = write_jsonl(cp, args.out_dir / f"y{y}" / "data.jsonl")
        index["per_year"].append({"year": y, **m})

    if args.cumulative:
        for i in range(1, len(args.years) + 1):
            cp = args.src_dir / f"cumulative_y{i}" / "corpus.txt"
            if not cp.exists():
                print(f"  [skip] missing {cp}", file=sys.stderr); continue
            m = write_jsonl(cp, args.out_dir / f"cumulative_y{i}" / "data.jsonl")
            index["cumulative"].append({"phase_idx": i, **m})

    (args.out_dir / "index.json").write_text(json.dumps(index, indent=2))
    print(f"\nwrote index → {args.out_dir/'index.json'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
