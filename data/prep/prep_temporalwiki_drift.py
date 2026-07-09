#!/usr/bin/env python3
"""Prepare TemporalWiki temporal-drift parquets for our continual-learning sweep.

Output layout (under --out-dir):
    train_s1.parquet    train_s2.parquet    train_s3.parquet    train_s4.parquet
    val_s1.parquet      val_s2.parquet      val_s3.parquet      val_s4.parquet
    val_stable.parquet
    manifest.json

Each parquet matches the schema verl + our run_sequential.py expect:
    columns = ['prompt', 'embedding', 'system', 'data_source', 'ability',
               'reward_model', 'extra_info']

What's in each set
------------------
- TRAIN (4 files): the SAME 450 (subject, relation) keys appear in every train_sN.parquet,
  but the gold object differs per slice (slice_i's truth). This directly tests the
  fact-override CL question: phase i trains on (Hans Zimmer, spouse) -> "Suzanne Zimmer";
  phase i+1 trains on the same prompt with -> "Vicki Carolin"; etc.

- VAL drift (4 files): 50 fixed (subject, relation) keys held out from training; each
  val_sN.parquet asks the same prompts but tags them with slice N's gold. M[i,j] = mean F1
  on val_sJ after training through phase i.

- VAL stability (1 file): 50 probes from `unchanged.csv` (facts stable across all 4
  slices). Same gold across the chain — measures whether drift training is corrupting
  knowledge that didn't drift. Catches "model overfits to drifting facts and corrupts
  unrelated stable knowledge" failure mode.

Determinism: numpy seed=42 throughout. (s,r) keys in train ∩ val = ∅.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# 4-date / 3-pair chronological chain: Nov→Dec, Dec→Jan, Jan→Feb 2026.
# Trades off compute vs depth — 3 sequential phases is enough to surface forgetting
# (BWT) and forward transfer (FWT) signals while staying within ~1 hr total compute
# for all 4 methods on 2 nodes.
SLICES = (
    "pair_20251120_20251201",
    "pair_20251201_20260101",
    "pair_20260101_20260201",
)

# Strong instruction-style system prompt with format demonstration.
# Robustness: explicit "no explanation / no quotes / no extra words" + few-shot showing the
# exact desired surface form. The F1 scorer's _extract_answer is permissive (handles
# <think>...</think>, "Answer: X" tags, etc.) so this is belt-and-suspenders.
SYSTEM_PROMPT = (
    "You are answering factual knowledge questions about Wikipedia entities. "
    "Given a subject and a relation, output ONLY the object value as a short plain-text "
    "string. No explanation, no quotation marks, no markup, no extra words. "
    "If multiple values are valid, pick the most canonical one.\n"
    "Examples:\n"
    "  Marshal Yanda educated at -> University of Iowa\n"
    "  Hans Zimmer spouse -> Suzanne Zimmer\n"
    "  Ho Chi Minh City contains administrative territorial entity -> District 7"
)


def load_changed(probes_root: Path, slice_tag: str) -> Dict[Tuple[str, str], Dict[str, str]]:
    """Return {(subject, relation): {object, subject_sitelink}} for changed.csv."""
    out: Dict[Tuple[str, str], Dict[str, str]] = {}
    p = probes_root / slice_tag / "changed.csv"
    with open(p) as f:
        for r in csv.DictReader(f):
            key = (r["subject"], r["relation"])
            out[key] = {"object": r["object"], "sitelink": r.get("subject_sitelink", r["subject"])}
    return out


def load_unchanged(probes_root: Path, slice_tag: str) -> List[Dict[str, str]]:
    p = probes_root / slice_tag / "unchanged.csv"
    rows = []
    with open(p) as f:
        for r in csv.DictReader(f):
            rows.append({
                "subject": r["subject"],
                "relation": r["relation"],
                "object": r["object"],
                "sitelink": r.get("subject_sitelink", r["subject"]),
            })
    return rows


def build_drift_pool(slice_data: Dict[str, Dict[Tuple[str, str], Dict[str, str]]]) -> List[Tuple[str, str]]:
    """Return (subject, relation) keys that appear in ALL slices and have a *different*
    object value in EVERY slice — i.e., #distinct objects == #slices.

    For an N-slice chain this gives keys where every slice has its own gold value.
    Reused-across-phases training on these keys guarantees every phase is a real
    fact-override event (not a no-op reinforcement of the previous phase's value).

    Note: still includes Wikipedia-editor-churn facts (multi-valued entities where
    different valid answers get surfaced across edits). That noise applies equally
    across all methods, so cross-method ranking stays meaningful even if absolute F1
    is biased.
    """
    common = set.intersection(*[set(d.keys()) for d in slice_data.values()])
    drift = []
    for k in common:
        objs = set(slice_data[s][k]["object"] for s in slice_data)
        if len(objs) == len(slice_data):  # every slice has a unique object
            drift.append(k)
    return sorted(drift)


def make_prompt_array(user_text: str) -> np.ndarray:
    """Match the bio/finqa/tooluse parquet shape: an object array of message dicts."""
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    arr = np.empty(len(msgs), dtype=object)
    for i, m in enumerate(msgs):
        arr[i] = m
    return arr


def slice_to_label(slice_tag: str) -> Dict[str, str]:
    """Parse 'pair_YYYYMMDD_YYYYMMDD' into machine + human-readable date labels.

    Returns: {"old": "2025-12-01", "new": "2026-01-01", "short": "Dec→Jan", "month": "2026-01"}
    These are propagated into per-row extra_info AND into the prep manifest so plotters
    and analysis scripts can recover the actual time range from any artifact.
    """
    # slice_tag like "pair_20251201_20260101"
    parts = slice_tag.split("_")
    assert len(parts) == 3 and parts[0] == "pair", f"unexpected slice tag: {slice_tag}"
    old, new = parts[1], parts[2]
    old_iso = f"{old[:4]}-{old[4:6]}-{old[6:8]}"
    new_iso = f"{new[:4]}-{new[4:6]}-{new[6:8]}"
    months = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    short = f"{months[int(old[4:6])]}→{months[int(new[4:6])]}"
    return {"old": old_iso, "new": new_iso, "short": short, "month": new_iso[:7]}


def build_row(subject: str, relation: str, gold_object: str, *,
              data_source: str, slice_tag: str, sitelink: str, idx: int,
              slice_label: Dict[str, str]) -> Dict:
    user_text = f"{subject} {relation}"
    return {
        "prompt": make_prompt_array(user_text),
        "embedding": np.array([], dtype=object),
        "system": SYSTEM_PROMPT,
        "data_source": data_source,
        "ability": "temporalwiki_drift",
        "reward_model": {"ground_truth": gold_object, "style": "temporalwiki_drift"},
        "extra_info": {
            "subject": subject,
            "relation": relation,
            "subject_sitelink": sitelink,
            "slice_tag": slice_tag,
            "slice_old": slice_label["old"],
            "slice_new": slice_label["new"],
            "slice_label_short": slice_label["short"],
            "index": str(idx),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--probes-root", type=Path,
        default=Path(os.environ.get("TEMPORALWIKI_PROBES", "data/temporalwiki_probes")),
        help="Directory containing per-slice probe CSVs (changed.csv / unchanged.csv) "
             "produced by the upstream TemporalWiki probe pipeline; one subdir per slice tag.",
    )
    ap.add_argument(
        "--out-dir", type=Path,
        default=Path(os.environ.get("CL_TEMPORAL_DRIFT_DATA", "data/temporalwiki_drift")),
        help="Output dir for the per-slice parquets + manifest.json.",
    )
    ap.add_argument("--slices", nargs="+", default=list(SLICES),
                    help="N chronological slice tags (default: Nov 2025 → Feb 2026, 4 dates / 3 pairs)")
    ap.add_argument("--n-train", type=int, default=500,
                    help="Train examples per slice = n_val_drift + (n_train - n_val_drift) disjoint background. "
                         "Same (s,r) keys reused across slices, slice-specific gold per phase.")
    ap.add_argument("--n-val-drift", type=int, default=50,
                    help="Drift val probes per slice. SAME 50 keys appear in train (with phase-specific gold) "
                         "AND val (with each-slice gold) — drift override CL test.")
    ap.add_argument("--n-val-stable", type=int, default=50,
                    help="Stability val probes (held fixed across the chain). NEVER appear in train. "
                         "Pure no-leak preservation test.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if len(args.slices) < 2:
        print(f"error: need >=2 slices, got {len(args.slices)}", file=sys.stderr)
        return 2

    rng = np.random.default_rng(args.seed)

    print(f"[prep] loading changed.csv for {len(args.slices)} slices...", file=sys.stderr)
    slice_data = {s: load_changed(args.probes_root, s) for s in args.slices}
    for s, d in slice_data.items():
        print(f"  {s}: {len(d)} changed (s,r) keys", file=sys.stderr)

    drift_keys = build_drift_pool(slice_data)
    print(f"[prep] drift pool (all-{len(args.slices)}-distinct): {len(drift_keys)} keys", file=sys.stderr)
    if len(drift_keys) < args.n_train:
        print(f"error: drift pool {len(drift_keys)} < n_train = {args.n_train}", file=sys.stderr)
        return 2
    if args.n_train < args.n_val_drift:
        print(f"error: n_train ({args.n_train}) must be >= n_val_drift ({args.n_val_drift})", file=sys.stderr)
        return 2

    # OVERRIDE design:
    #   * 50 val_drift keys held out as the static eval set (same keys evaluated against
    #     each slice's gold — that's the drift signal).
    #   * Train per phase = those 50 val keys (phase-specific gold; this is intentional —
    #     diagonal cell M[i,i] is the "did update take?" signal) + (n_train - 50)
    #     disjoint background drift keys to prevent pure memorization.
    # Off-diagonal cells M[i, j!=i] have train_gold != val_gold so they cleanly measure
    # forgetting / retention, no leak.
    perm = rng.permutation(len(drift_keys))
    val_idx = perm[: args.n_val_drift]
    background_idx = perm[args.n_val_drift : args.n_train]  # n_train - n_val_drift entries
    val_keys = [drift_keys[i] for i in val_idx]
    background_keys = [drift_keys[i] for i in background_idx]
    train_keys = list(val_keys) + background_keys           # val keys are FIRST in train
    assert len(train_keys) == args.n_train, f"train size {len(train_keys)} != {args.n_train}"
    assert set(val_keys).isdisjoint(set(background_keys)), "background overlaps val"
    print(f"[prep] train per phase = {len(val_keys)} val_drift keys + {len(background_keys)} disjoint bg = {len(train_keys)}",
          file=sys.stderr)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-compute slice labels once.
    slice_labels = {s: slice_to_label(s) for s in args.slices}

    # ----- TRAIN: same keys, slice-specific gold per file -----
    for i, slice_tag in enumerate(args.slices, start=1):
        ds = f"temporalwiki_drift_s{i}"
        rows = [
            build_row(subj, rel,
                      gold_object=slice_data[slice_tag][(subj, rel)]["object"],
                      data_source=ds, slice_tag=slice_tag,
                      sitelink=slice_data[slice_tag][(subj, rel)]["sitelink"],
                      idx=k_idx,
                      slice_label=slice_labels[slice_tag])
            for k_idx, (subj, rel) in enumerate(train_keys)
        ]
        df = pd.DataFrame(rows)
        out = args.out_dir / f"train_s{i}.parquet"
        df.to_parquet(out, index=False)
        print(f"  wrote {out.name}: {len(df)} rows", file=sys.stderr)

    # ----- VAL drift: same keys, slice-specific gold per file -----
    for i, slice_tag in enumerate(args.slices, start=1):
        ds = f"temporalwiki_drift_s{i}"
        rows = [
            build_row(subj, rel,
                      gold_object=slice_data[slice_tag][(subj, rel)]["object"],
                      data_source=ds, slice_tag=slice_tag,
                      sitelink=slice_data[slice_tag][(subj, rel)]["sitelink"],
                      idx=k_idx,
                      slice_label=slice_labels[slice_tag])
            for k_idx, (subj, rel) in enumerate(val_keys)
        ]
        df = pd.DataFrame(rows)
        out = args.out_dir / f"val_s{i}.parquet"
        df.to_parquet(out, index=False)
        print(f"  wrote {out.name}: {len(df)} rows", file=sys.stderr)

    # ----- VAL stability: probes from unchanged.csv of the LATEST slice (most authoritative). -----
    # We use the same probe set + gold across all chain points; degradation here is corruption,
    # not drift. Pull from the latest slice's unchanged.csv to maximize alignment with the
    # final-state reality the model is being driven toward.
    stable_rows_all = load_unchanged(args.probes_root, args.slices[-1])
    rng_stable = np.random.default_rng(args.seed + 1)
    pick = rng_stable.permutation(len(stable_rows_all))[: args.n_val_stable]
    stable_label = {"old": "(any)", "new": slice_labels[args.slices[-1]]["new"],
                    "short": "stable", "month": slice_labels[args.slices[-1]]["month"]}
    rows = [
        build_row(stable_rows_all[k]["subject"],
                  stable_rows_all[k]["relation"],
                  gold_object=stable_rows_all[k]["object"],
                  data_source="temporalwiki_stable",
                  slice_tag="(stable)",
                  sitelink=stable_rows_all[k]["sitelink"],
                  idx=k_idx,
                  slice_label=stable_label)
        for k_idx, k in enumerate(pick)
    ]
    df = pd.DataFrame(rows)
    out = args.out_dir / "val_stable.parquet"
    df.to_parquet(out, index=False)
    print(f"  wrote {out.name}: {len(df)} rows", file=sys.stderr)

    # ----- manifest.json -----
    # Maps task position -> slice tag -> human-readable date label so plotting and
    # downstream analysis can reconstruct "S1 = Dec→Jan 2026" from any artifact.
    slices_meta = []
    for i, s in enumerate(args.slices, start=1):
        slices_meta.append({
            "position": i,
            "task_key": f"ts{i}",
            "data_source": f"temporalwiki_drift_s{i}",
            "slice_tag": s,
            "old": slice_labels[s]["old"],
            "new": slice_labels[s]["new"],
            "label_short": slice_labels[s]["short"],
            "month": slice_labels[s]["month"],
        })
    manifest = {
        "slices": slices_meta,
        "n_train": args.n_train,
        "n_val_drift": args.n_val_drift,
        "n_val_stable": args.n_val_stable,
        "seed": args.seed,
        "drift_pool_size": len(drift_keys),
        "train_keys_subset": [(s, r) for (s, r) in train_keys[:5]],
        "val_keys_subset": [(s, r) for (s, r) in val_keys[:5]],
        "data_sources": [f"temporalwiki_drift_s{i}" for i in range(1, 5)] + ["temporalwiki_stable"],
        "system_prompt": SYSTEM_PROMPT,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  wrote manifest.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
