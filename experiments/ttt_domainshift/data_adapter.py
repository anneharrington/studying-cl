#!/usr/bin/env python3
"""Convert the domain-shift corpora into VeOmni's plaintext-iterable format for
In-Place TTT (continual pretraining on the ToolUse -> FinQA -> SciKE-Bio sequence).

VeOmni's recommended config uses:
    data.data_type=plaintext  data.datasets_type=iterable  data.text_keys=content_split

so each shard is a JSONL where every line is {"content_split": "<text chunk>"}.
We reuse the per-task / cumulative corpora produced by
experiments/cartridges_domainshift/data_adapter.py (each example is delimited by a
"=== <task> example ===" header), emitting one JSON record per example so VeOmni
can pack per document.

Inputs:
    <src>/<task>/corpus.txt
    <src>/cumulative_p{i}/corpus.txt
Outputs:
    <out>/<task>/data.jsonl
    <out>/cumulative_p{i}/data.jsonl
    <out>/index.json

Usage:
    python data_adapter.py --src <run>/corpora --out <run>/data
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_TASKS = ["tooluse", "finqa", "sciknoweval_bio"]


def _examples(corpus_text: str) -> list[str]:
    """Split a corpus.txt back into per-example chunks (delimiter '=== ... ==='). """
    parts = [p.strip() for p in corpus_text.split("\n=== ")]
    return [("=== " + p if not p.startswith("=== ") else p) for p in parts if p]


def write_jsonl(corpus_path: Path, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    examples = _examples(corpus_path.read_text())
    out = out_dir / "data.jsonl"
    with out.open("w") as f:
        for ex in examples:
            f.write(json.dumps({"content_split": ex}) + "\n")
    print(f"  {corpus_path.parent.name}: {len(examples)} records -> {out}")
    return len(examples)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="corpora root from cartridges_domainshift/data_adapter.py")
    ap.add_argument("--out", required=True, help="VeOmni jsonl output root")
    ap.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    index = {"tasks": args.tasks, "shards": []}

    for task in args.tasks:
        cp = src / task / "corpus.txt"
        if not cp.exists():
            print(f"!! missing {cp} — run cartridges_domainshift/data_adapter.py first",
                  file=sys.stderr)
            return 2
        n = write_jsonl(cp, out / task)
        index["shards"].append({"task": task, "n": n})

    for i in range(1, len(args.tasks) + 1):
        cp = src / f"cumulative_p{i}" / "corpus.txt"
        if cp.exists():
            write_jsonl(cp, out / f"cumulative_p{i}")

    (out / "index.json").write_text(json.dumps(index, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
