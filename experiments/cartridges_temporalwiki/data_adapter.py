#!/usr/bin/env python3
"""Pull Wikipedia article bodies from per-snapshot SQLite DBs to build the
per-slice corpus that Cartridges' SelfStudySynthesizer consumes.

For each TemporalWiki slice ts{i}, the slice's "new" snapshot date (e.g. ts1 →
2025-12-01) is the snapshot whose article state encodes the new fact for
drift triples. We pull article bodies for the union of (train_s{i} subjects ∪
val_s{i} subjects ∪ val_stable subjects), so the cartridge has context for
every Q/A in the eval matrix.

Inputs (read):
    /workspace/home/nayan/temporalwiki/wp_text/{YYYYMMDD}.db
        schema: pages(title TEXT, text TEXT)
    /workspace/home/nayan/temporalwiki/runs/cl_drift_data/train_s{i}.parquet
    /workspace/home/nayan/temporalwiki/runs/cl_drift_data/val_s{i}.parquet
    /workspace/home/nayan/temporalwiki/runs/cl_drift_data/val_stable.parquet
        rows have extra_info.subject_sitelink → wp_text title (100% match)

Outputs (write):
    <out>/ts{i}/corpus.txt              article bodies for slice i (train+val)
    <out>/ts{i}/manifest.json
    <out>/cumulative_ts{i}/corpus.txt   shuffled union of slices 1..i
    <out>/index.json

Usage:
    python /home/nayan/scripts/cartridges_temporalwiki/data_adapter.py
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import pyarrow.parquet as pq

DEFAULT_OUT = Path("/workspace/home/nayan/cartridges_temporalwiki/corpora")
DEFAULT_DRIFT = Path("/workspace/home/nayan/temporalwiki/runs/cl_drift_data")
DEFAULT_WP_TEXT = Path("/workspace/home/nayan/temporalwiki/wp_text")
DEFAULT_MANIFEST = Path(
    "/workspace/home/nayan/sdpo_seq/runs/twiki-cl_sft_orderT_nothink_s42/temporal_manifest.json")
DEFAULT_SLICES = ["ts1", "ts2", "ts3"]
# Articles below this many chars (~25 tokens) are usually redirect stubs;
# skip so the synthesizer doesn't waste samples on empty contexts.
MIN_ARTICLE_CHARS = 200


def _slice_subjects(parquet_path: Path) -> list[tuple[str, str, str]]:
    """Return [(sitelink, subject, relation_object_pair)] for one parquet."""
    out = []
    for r in pq.read_table(parquet_path).to_pylist():
        ei = dict(r["extra_info"])
        sl = ei.get("subject_sitelink", "")
        if not sl:
            continue
        subj = ei.get("subject", sl)
        rel = ei.get("relation", "")
        gold = str(dict(r["reward_model"]).get("ground_truth", "")).strip()
        out.append((sl, subj, f"{subj} {rel} -> {gold}" if rel else ""))
    return out


def build_slice_corpus(
    slice_id: str, slice_new: str, drift_dir: Path, wp_text_dir: Path,
    out_dir: Path, include_stable: bool,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = wp_text_dir / f"{slice_new.replace('-', '')}.db"
    if not db_path.exists():
        raise FileNotFoundError(f"wp_text DB missing: {db_path}")

    train_p = drift_dir / f"train_s{slice_id[-1]}.parquet"
    val_p = drift_dir / f"val_s{slice_id[-1]}.parquet"
    stable_p = drift_dir / "val_stable.parquet"

    # Union sitelinks across train + val_drift (+ val_stable, once, at slice 1).
    sitelink_set: set[str] = set()
    fact_lines: dict[str, list[str]] = {}
    sources = [(train_p, "train"), (val_p, "val_drift")]
    if include_stable:
        sources.append((stable_p, "val_stable"))
    for p, _src in sources:
        if not p.exists():
            print(f"  !! missing {p}", file=sys.stderr); continue
        for sl, _subj, fact in _slice_subjects(p):
            sitelink_set.add(sl)
            if fact:
                fact_lines.setdefault(sl, []).append(fact)

    con = sqlite3.connect(str(db_path))
    n_kept = 0; n_short = 0; n_missing = 0; total_chars = 0
    text_parts: list[str] = []
    for sl in sorted(sitelink_set):
        row = con.execute("SELECT text FROM pages WHERE title=?", (sl,)).fetchone()
        if row is None:
            n_missing += 1; continue
        body = (row[0] or "").strip()
        if len(body) < MIN_ARTICLE_CHARS:
            n_short += 1; continue
        # Header carries the article title + snapshot date so the synthesizer
        # can ground questions to the (subject, time) tuple. Append the train
        # fact lines (where present) AFTER the article body so the cartridge
        # also sees the canonical Q→A form for that subject.
        header = f"\n=== Wikipedia article  title={sl}  as_of={slice_new} ===\n"
        text_parts.append(header)
        text_parts.append(body)
        if sl in fact_lines:
            text_parts.append("\n\nKnown facts:\n")
            text_parts.append("\n".join(f"  - {f}" for f in fact_lines[sl]))
        text_parts.append("\n")
        n_kept += 1
        total_chars += len(body)
    con.close()

    text = "".join(text_parts)
    cp = out_dir / "corpus.txt"
    cp.write_text(text, encoding="utf-8")
    manifest = {
        "slice_id": slice_id,
        "slice_new": slice_new,
        "wp_text_db": str(db_path),
        "n_articles_kept": n_kept,
        "n_articles_short_skipped": n_short,
        "n_articles_missing": n_missing,
        "n_subjects_with_train_facts": sum(1 for v in fact_lines.values() if v),
        "n_chars": len(text),
        "est_tokens": len(text) // 4,
        "corpus_path": str(cp),
        "include_stable": include_stable,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"  {slice_id}: {n_kept} articles kept ({n_short} short, {n_missing} missing), "
        f"{len(text)/1e6:.2f}M chars, ~{len(text)/4000:.1f}K tokens -> {cp}",
        file=sys.stderr,
    )
    return manifest


def build_cumulative_corpus(
    per_slice_dir: Path, slices_through: list[str], phase_idx: int,
    out_dir: Path, seed: int = 42,
) -> dict:
    """Cumulative_tsN/corpus.txt = shuffled union of per-slice corpora through
    slice N. Each per-slice corpus is split on the article header marker and
    shuffled at article granularity (not character)."""
    import random
    rng = random.Random(seed)

    docs: list[str] = []
    n_chars = 0
    for s in slices_through:
        cp = per_slice_dir / s / "corpus.txt"
        if not cp.exists():
            print(f"  !! cumulative: missing {cp}", file=sys.stderr); continue
        text = cp.read_text(encoding="utf-8")
        parts = text.split("\n=== Wikipedia article ")
        for part in parts:
            part = part.strip()
            if not part: continue
            doc = "\n=== Wikipedia article " + part if not part.startswith("===") else "\n" + part
            docs.append(doc); n_chars += len(doc)
    rng.shuffle(docs)
    out_dir.mkdir(parents=True, exist_ok=True)
    cp = out_dir / "corpus.txt"
    cp.write_text("\n".join(docs), encoding="utf-8")
    manifest = {
        "phase_idx": phase_idx,
        "slices_through": slices_through,
        "n_articles": len(docs),
        "n_chars": n_chars,
        "est_tokens": n_chars // 4,
        "shuffled_seed": seed,
        "corpus_path": str(cp),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"  cumulative phase {phase_idx} (slices {slices_through}): "
        f"{len(docs)} articles, {n_chars/1e6:.2f}M chars -> {cp}",
        file=sys.stderr,
    )
    return manifest


def _load_slice_meta(manifest_path: Path, slices: list[str]) -> dict[str, str]:
    """Map ts{i} -> slice_new (YYYY-MM-DD)."""
    meta = json.loads(manifest_path.read_text())
    by_key = {s["task_key"]: s["new"] for s in meta["slices"]}
    return {s: by_key[s] for s in slices if s in by_key}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--drift-dir", type=Path, default=DEFAULT_DRIFT,
                    help="Dir with train_s{1..3}.parquet, val_s{1..3}.parquet, val_stable.parquet")
    ap.add_argument("--wp-text-dir", type=Path, default=DEFAULT_WP_TEXT,
                    help="Dir with per-snapshot {YYYYMMDD}.db (pages: title,text)")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                    help="temporal_manifest.json (defines slice -> snapshot date)")
    ap.add_argument("--slices", nargs="+", default=DEFAULT_SLICES)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-cumulative", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    slice_meta = _load_slice_meta(args.manifest, args.slices)

    index = {"slices": [], "cumulative": [], "out_dir": str(args.out_dir)}
    for i, s in enumerate(args.slices, start=1):
        slice_new = slice_meta[s]
        m = build_slice_corpus(
            slice_id=s, slice_new=slice_new,
            drift_dir=args.drift_dir, wp_text_dir=args.wp_text_dir,
            out_dir=args.out_dir / s,
            include_stable=(i == 1),
        )
        index["slices"].append({"slice_id": s, **m})

    if not args.no_cumulative:
        for i, s in enumerate(args.slices, start=1):
            cm = build_cumulative_corpus(
                per_slice_dir=args.out_dir,
                slices_through=args.slices[:i],
                phase_idx=i,
                out_dir=args.out_dir / f"cumulative_{s}",
                seed=args.seed,
            )
            index["cumulative"].append({"phase_idx": i, **cm})

    (args.out_dir / "index.json").write_text(json.dumps(index, indent=2))
    print(f"\nwrote index -> {args.out_dir/'index.json'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
