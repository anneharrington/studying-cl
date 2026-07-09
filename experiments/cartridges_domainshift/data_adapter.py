#!/usr/bin/env python3
"""Convert the domain-shift task parquets into per-phase text corpora that
Cartridges' SelfStudySynthesizer consumes.

Domain shift is the sequence ToolUse -> FinQA -> SciKE-Bio. Unlike the finance /
temporalwiki settings (where a phase is a single drifting corpus), here each
phase is a *task* training set. We turn each task's train split into one text
corpus by concatenating, per example, the user-facing prompt and its gold
answer — that is the material a cartridge should compress for that phase.

Inputs (verl-format parquet, the same datasets the weight-update methods train on):
    <data-dir>/<task>/train.parquet
    each row: prompt = [{role, content}, ...], reward_model = {ground_truth: ...}

Outputs:
    <out>/<task>/corpus.txt                 one text file per phase (the synth target)
    <out>/<task>/manifest.json              per-phase provenance
    <out>/cumulative_p{i}/corpus.txt        union(task_1..task_i), shuffled (optional)
    <out>/index.json                        top-level index

Usage:
    python data_adapter.py --data-dir datasets --out-dir <run>/corpora
    python data_adapter.py --data-dir datasets --tasks tooluse finqa sciknoweval_bio --cumulative
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq

DEFAULT_TASKS = ["tooluse", "finqa", "sciknoweval_bio"]


def _last_user_content(prompt) -> str | None:
    """Extract the user-facing text from a verl `prompt` (list of message dicts)."""
    if isinstance(prompt, str):
        return prompt
    if not prompt:
        return None
    for msg in reversed(list(prompt)):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return str(msg.get("content", "")).strip()
    last = prompt[-1]
    return str(last.get("content", "")).strip() if isinstance(last, dict) else None


def _gold(reward_model) -> str:
    gt = reward_model.get("ground_truth", "") if isinstance(reward_model, dict) else reward_model
    if isinstance(gt, (list, tuple)):
        return " | ".join(str(x) for x in gt)
    return str(gt)


def build_task_corpus(parquet_path: Path, task: str, out_dir: Path) -> dict:
    """Concatenate (prompt, gold) text for every train example of one task."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = pq.read_table(parquet_path).to_pylist()
    chunks, n_docs = [], 0
    for r in rows:
        user = _last_user_content(r.get("prompt"))
        if not user:
            continue
        gold = _gold(r.get("reward_model"))
        chunks.append(f"=== {task} example ===\n{user}\n\n[answer]\n{gold}\n")
        n_docs += 1
    text = "\n".join(chunks)
    corpus_path = out_dir / "corpus.txt"
    corpus_path.write_text(text)
    manifest = {"task": task, "n_docs": n_docs, "n_chars": len(text),
                "source": str(parquet_path)}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  {task}: {n_docs} examples -> {corpus_path} ({len(text)} chars)")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="datasets",
                    help="root holding <task>/train.parquet (verl format)")
    ap.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS,
                    help="ordered phase tasks (default: %(default)s)")
    ap.add_argument("--out-dir", required=True, help="corpora output root")
    ap.add_argument("--cumulative", action="store_true",
                    help="also emit shuffled union(task_1..task_i) per phase")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    index = {"tasks": args.tasks, "phases": []}
    per_task_text: dict[str, str] = {}

    for task in args.tasks:
        pq_path = data_dir / task / "train.parquet"
        if not pq_path.exists():
            print(f"!! missing {pq_path} — build it first (see data/prep/ and docs/TASKS.md)",
                  file=sys.stderr)
            return 2
        m = build_task_corpus(pq_path, task, out_dir / task)
        per_task_text[task] = (out_dir / task / "corpus.txt").read_text()
        index["phases"].append(m)

    if args.cumulative:
        rng = random.Random(args.seed)
        for i in range(1, len(args.tasks) + 1):
            docs = []
            for t in args.tasks[:i]:
                docs.extend(d for d in per_task_text[t].split("\n=== ") if d.strip())
            rng.shuffle(docs)
            cdir = out_dir / f"cumulative_p{i}"
            cdir.mkdir(parents=True, exist_ok=True)
            (cdir / "corpus.txt").write_text("\n=== ".join(docs))
            print(f"  cumulative_p{i}: union({','.join(args.tasks[:i])}) -> {cdir}/corpus.txt")

    (out_dir / "index.json").write_text(json.dumps(index, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
