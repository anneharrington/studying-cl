#!/usr/bin/env python3
"""Pull Wikipedia article bodies from per-snapshot SQLite DBs and emit
VeOmni plaintext-iterable JSONL for In-Place TTT continual pretraining.

Mirrors cartridges_temporalwiki/data_adapter.py — same article corpus, same
subject scope (train + val drift-pool + val_stable for slice 1) — but the
output is per-line JSONL with `{"content_split": "<article body>"}` so
VeOmni's plaintext-iterable dataloader can consume it directly.

Why articles (not triples): In-Place TTT is designed for continual
*pretraining* on raw text. Triples are a degenerate corpus (~5K tokens/slice)
and don't engage TTT's strength. Article bodies (~4M tokens/slice) match the
paper's regime (long-context CPT) and keep the comparison with cartridges
apples-to-apples — both methods consume the same Wikipedia diff data, just
in their method-appropriate format.

Inputs:
    /workspace/home/nayan/temporalwiki/wp_text/{YYYYMMDD}.db
        schema: pages(title TEXT, text TEXT)
    /workspace/home/nayan/temporalwiki/runs/cl_drift_data/train_s{i}.parquet
    /workspace/home/nayan/temporalwiki/runs/cl_drift_data/val_s{i}.parquet
    /workspace/home/nayan/temporalwiki/runs/cl_drift_data/val_stable.parquet
    /workspace/home/nayan/sdpo_seq/runs/twiki-cl_sft_orderT_nothink_s42/temporal_manifest.json

Outputs:
    <out>/ts{i}/data.jsonl                 — slice-i article corpus
    <out>/cumulative_ts{i}/data.jsonl      — shuffled union of slices 1..i
    <out>/ts{i}/manifest.json
    <out>/index.json
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from pathlib import Path
from typing import List, Tuple

import pyarrow.parquet as pq

DEFAULT_OUT = Path("/workspace/home/nayan/ttt_temporalwiki/data")
DEFAULT_DRIFT = Path("/workspace/home/nayan/temporalwiki/runs/cl_drift_data")
DEFAULT_WP_TEXT = Path("/workspace/home/nayan/temporalwiki/wp_text")
DEFAULT_MANIFEST = Path(
    "/workspace/home/nayan/sdpo_seq/runs/twiki-cl_sft_orderT_nothink_s42/temporal_manifest.json")
DEFAULT_SLICES = ["ts1", "ts2", "ts3"]
# Articles below this many chars (~25 tokens) are usually redirect stubs;
# skip so TTT doesn't waste steps on near-empty contexts.
MIN_ARTICLE_CHARS = 200


def _slice_subjects(parquet_path: Path) -> List[Tuple[str, str, str]]:
    """Return [(sitelink, subject, fact_string)] for one parquet."""
    out = []
    if not parquet_path.exists():
        return out
    for r in pq.read_table(parquet_path).to_pylist():
        ei = dict(r["extra_info"])
        sl = ei.get("subject_sitelink", "")
        if not sl: continue
        subj = ei.get("subject", sl)
        rel = ei.get("relation", "")
        gold = str(dict(r["reward_model"]).get("ground_truth", "")).strip()
        out.append((sl, subj, f"{subj} {rel} -> {gold}" if rel else ""))
    return out


def write_slice_jsonl(
    *, slice_id: str, slice_new: str, drift_dir: Path, wp_text_dir: Path,
    out_path: Path, include_stable: bool,
) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    db_path = wp_text_dir / f"{slice_new.replace('-', '')}.db"
    if not db_path.exists():
        raise FileNotFoundError(f"wp_text DB missing: {db_path}")

    train_p = drift_dir / f"train_s{slice_id[-1]}.parquet"
    val_p = drift_dir / f"val_s{slice_id[-1]}.parquet"
    stable_p = drift_dir / "val_stable.parquet"

    sitelink_set: set[str] = set()
    fact_lines: dict[str, list[str]] = {}
    sources = [(train_p, "train"), (val_p, "val_drift")]
    if include_stable:
        sources.append((stable_p, "val_stable"))
    for p, _src in sources:
        for sl, _subj, fact in _slice_subjects(p):
            sitelink_set.add(sl)
            if fact: fact_lines.setdefault(sl, []).append(fact)

    con = sqlite3.connect(str(db_path))
    n_kept = 0; n_short = 0; n_missing = 0; total_chars = 0
    with out_path.open("w", encoding="utf-8") as f:
        for sl in sorted(sitelink_set):
            row = con.execute("SELECT text FROM pages WHERE title=?", (sl,)).fetchone()
            if row is None:
                n_missing += 1; continue
            body = (row[0] or "").strip()
            if len(body) < MIN_ARTICLE_CHARS:
                n_short += 1; continue
            # Header carries the article title + snapshot date so the TTT
            # objective gets explicit (subject, time) grounding. Append the
            # known facts AFTER the article body so the model sees both the
            # narrative source and the canonical Q→A form.
            content = f"=== Wikipedia article  title={sl}  as_of={slice_new} ===\n{body}\n"
            if sl in fact_lines:
                content += "\nKnown facts:\n" + "\n".join(f"  - {fact}" for fact in fact_lines[sl]) + "\n"
            f.write(json.dumps({"content_split": content}, ensure_ascii=False) + "\n")
            n_kept += 1; total_chars += len(content)
    con.close()

    manifest = {
        "slice_id": slice_id,
        "slice_new": slice_new,
        "wp_text_db": str(db_path),
        "n_articles_kept": n_kept,
        "n_articles_short_skipped": n_short,
        "n_articles_missing": n_missing,
        "n_subjects_with_train_facts": sum(1 for v in fact_lines.values() if v),
        "n_chars": total_chars,
        "est_tokens": total_chars // 4,
        "include_stable": include_stable,
        "out": str(out_path),
    }
    out_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"  {slice_id}: {n_kept} articles ({n_short} short, {n_missing} missing), "
        f"{total_chars/1e6:.2f}M chars, ~{total_chars/4000:.1f}K tokens -> {out_path}",
        file=sys.stderr,
    )
    return manifest


def write_cumulative_jsonl(
    per_slice_dir: Path, slices_through: List[str], phase_idx: int,
    out_path: Path, seed: int = 42,
) -> dict:
    """Cumulative_tsN/data.jsonl = shuffled union of per-slice JSONLs through
    slice N. Shuffles at article granularity so KVFromText-equivalent
    initialization isn't biased by slice order."""
    rng = random.Random(seed)
    docs: List[str] = []
    n_chars = 0
    for s in slices_through:
        p = per_slice_dir / s / "data.jsonl"
        if not p.exists():
            print(f"  !! cumulative: missing {p}", file=sys.stderr); continue
        for line in p.open("r", encoding="utf-8"):
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
                docs.append(obj.get("content_split", ""))
                n_chars += len(obj.get("content_split", ""))
            except json.JSONDecodeError:
                continue
    rng.shuffle(docs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for c in docs:
            f.write(json.dumps({"content_split": c}, ensure_ascii=False) + "\n")
    manifest = {
        "phase_idx": phase_idx,
        "slices_through": slices_through,
        "n_articles": len(docs),
        "n_chars": n_chars,
        "est_tokens": n_chars // 4,
        "shuffled_seed": seed,
        "out": str(out_path),
    }
    out_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"  cumulative phase {phase_idx} (slices {slices_through}): "
        f"{len(docs)} articles, {n_chars/1e6:.2f}M chars -> {out_path}",
        file=sys.stderr,
    )
    return manifest


def _load_slice_meta(manifest_path: Path, slices: List[str]) -> dict:
    meta = json.loads(manifest_path.read_text())
    by_key = {s["task_key"]: s["new"] for s in meta["slices"]}
    return {s: by_key[s] for s in slices if s in by_key}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--drift-dir", type=Path, default=DEFAULT_DRIFT)
    ap.add_argument("--wp-text-dir", type=Path, default=DEFAULT_WP_TEXT)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--slices", nargs="+", default=DEFAULT_SLICES)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-cumulative", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    slice_meta = _load_slice_meta(args.manifest, args.slices)

    index = {"slices": [], "cumulative": [], "out_dir": str(args.out_dir)}
    for i, s in enumerate(args.slices, start=1):
        slice_new = slice_meta[s]
        m = write_slice_jsonl(
            slice_id=s, slice_new=slice_new,
            drift_dir=args.drift_dir, wp_text_dir=args.wp_text_dir,
            out_path=args.out_dir / s / "data.jsonl",
            include_stable=(i == 1),
        )
        index["slices"].append({"slice_id": s, **m})

    if not args.no_cumulative:
        for i, s in enumerate(args.slices, start=1):
            cm = write_cumulative_jsonl(
                per_slice_dir=args.out_dir,
                slices_through=args.slices[:i],
                phase_idx=i,
                out_path=args.out_dir / f"cumulative_{s}" / "data.jsonl",
                seed=args.seed,
            )
            index["cumulative"].append({"phase_idx": i, **cm})

    (args.out_dir / "index.json").write_text(json.dumps(index, indent=2))
    print(f"\nwrote index -> {args.out_dir/'index.json'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
