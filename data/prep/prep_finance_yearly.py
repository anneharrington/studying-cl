#!/usr/bin/env python3
"""Build per-year train/val parquets from JanosAudran/financial-reports-sec
for sequential temporal-drift CL on 10-K forward-sentiment prediction.

Inputs:
    <HF cache>/datasets--JanosAudran--financial-reports-sec/snapshots/<hash>/
        data/large/train/shard_*.jsonl
    (run scripts/data/download_finance_data.py first)

Outputs (under --out-dir):
    train_yYYYY.parquet   for each YYYY in --years
    val_yYYYY.parquet     for each YYYY in --years
    manifest.json         (year metadata, label counts, sample counts)

Per-row schema is the verl PPO format used by run_sequential.py:
    columns = ['prompt', 'embedding', 'system', 'data_source', 'ability',
               'reward_model', 'extra_info']

Determinism: numpy seed=42. Per-year, val_keys ∩ train_keys = ∅.

Source labels are JanosAudran's pre-computed forward stock direction at the
1d/5d/30d horizon; we use 30d (closest analog to a "forward quarter" sentiment
horizon). Mapping: positive -> "up", negative -> "down".
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

# Default: 6 phases of chronological 10-K filings 2015-2020.
DEFAULT_YEARS = (2015, 2016, 2017, 2018, 2019, 2020)

# Closest analog to FinanceBench's "forward sentiment over quarter" horizon.
DEFAULT_HORIZON = "30d"

# Cap raw filing text at this many characters. At ~3.75 chars/token, 50k chars
# ≈ 13.3k body tokens; with the system + user wrapper the prompt fits inside
# data.max_prompt_length=16384 (~50% of Qwen3-8B's 32k native context).
# 10-K filings have median ~260k chars / p99 ~624k chars (full report-section
# concat), so almost every filing gets truncated — the cap is a training-
# memory dial, not a data-completeness one.
DEFAULT_MAX_CHARS = 50000

# Strong instruction-style system prompt + 2-shot format example so the
# exact-match scorer doesn't fight surface noise.
SYSTEM_PROMPT = (
    "You are a financial analyst. Given a 10-K filing excerpt, predict the "
    "forward stock direction over the next 30 days after the filing date. "
    "Return ONE word ONLY: up or down. No explanation, no quotation marks, "
    "no extra words.\n"
    "Examples:\n"
    "  ... -> up\n"
    "  ... -> down"
)

# Section ordering inside a 10-K: items 1, 1A, 1B, 2..15, with 7A/9A/9B as
# subsections. We concatenate in standard order (Business, Risk Factors,
# Management Discussion, Financials, ...) until the char cap is hit.
SECTION_ORDER = [
    "section_1", "section_1A", "section_1B",
    "section_2", "section_3", "section_4",
    "section_5", "section_6", "section_7", "section_7A", "section_7B",
    "section_8", "section_9", "section_9A", "section_9B",
    "section_10", "section_11", "section_12", "section_13", "section_14", "section_15",
]


def find_shards(cache_root: str) -> List[str]:
    """Return sorted list of shard paths under the HF cache for large/train."""
    pattern = (
        f"{cache_root}/datasets--JanosAudran--financial-reports-sec/"
        "snapshots/*/data/large/train/shard_*.jsonl"
    )
    shards = sorted(glob.glob(pattern))
    if not shards:
        raise FileNotFoundError(
            f"No JanosAudran shards found at {pattern}. "
            f"Run scripts/data/download_finance_data.py first."
        )
    return shards


def iter_filings(shards: List[str]) -> Iterable[Tuple[Dict, Dict]]:
    """Yield (company_record, filing_record) for every filing across all shards."""
    for shard in shards:
        with open(shard) as f:
            for line in f:
                if not line.strip():
                    continue
                ex = json.loads(line)
                for fl in ex.get("filings", []):
                    yield ex, fl


def filing_to_text(filing: Dict, max_chars: int) -> str:
    """Concatenate report sections in canonical order, capped at max_chars chars."""
    sections = filing.get("report", {})
    parts: List[str] = []
    total = 0
    for key in SECTION_ORDER:
        sentences = sections.get(key) or []
        for s in sentences:
            if not isinstance(s, str):
                continue
            remaining = max_chars - total
            if remaining <= 0:
                break
            piece = s if len(s) <= remaining else s[:remaining]
            parts.append(piece)
            total += len(piece)
        if total >= max_chars:
            break
    return " ".join(parts)


def build_row(
    *,
    year: int,
    company_name: str,
    cik: str,
    filing: Dict,
    horizon: str,
    text: str,
    idx: int,
) -> Dict:
    raw_label = filing["labels"][horizon]
    gold = "up" if raw_label == "positive" else "down"
    user_text = (
        f"Document: {company_name} 10-K filed {filing['filingDate']}\n"
        f"[START OF FILING]\n{text}\n[END OF FILING]\n\n"
        f"Forward stock direction over 30 days: "
    )
    prompt_arr = np.empty(2, dtype=object)
    prompt_arr[0] = {"role": "system", "content": SYSTEM_PROMPT}
    prompt_arr[1] = {"role": "user", "content": user_text}
    return {
        "prompt": prompt_arr,
        "embedding": np.array([], dtype=object),
        "system": SYSTEM_PROMPT,
        "data_source": f"finance_yr_{year}",
        "ability": "finance_sentiment",
        "reward_model": {"ground_truth": gold, "style": "finance_sentiment"},
        "extra_info": {
            "cik": str(cik),
            "company": str(company_name),
            "filing_date": str(filing["filingDate"]),
            "year": str(year),
            "form": str(filing.get("form", "")),
            "horizon": horizon,
            "raw_label": raw_label,
            "index": str(idx),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cache-root", type=str,
        default=os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface")) + "/hub",
        help="HuggingFace cache 'hub' directory (where download_finance_data.py wrote).",
    )
    ap.add_argument(
        "--out-dir", type=Path,
        default=Path(os.environ.get("CL_FINANCE_DATA", "data/finance_yearly")),
        help="Output dir for the per-year parquets + manifest.json.",
    )
    ap.add_argument("--years", type=int, nargs="+", default=list(DEFAULT_YEARS),
                    help="Filing years to use as sequential CL phases (default: 2015-2020)")
    ap.add_argument("--horizon", default=DEFAULT_HORIZON, choices=("1d", "5d", "30d"),
                    help="Forward stock direction horizon (default: 30d)")
    ap.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS,
                    help=f"Cap filing text excerpt at this many chars (default: {DEFAULT_MAX_CHARS})")
    ap.add_argument("--n-train", type=int, default=500,
                    help="Train rows per year (sampled deterministically). Default 500.")
    ap.add_argument("--n-val", type=int, default=50,
                    help="Val rows per year (held out, disjoint from train). Default 50.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if any(y < 1990 or y > 2025 for y in args.years):
        print(f"error: year out of range: {args.years}", file=sys.stderr)
        return 2

    print(f"[prep_finance] scanning shards under {args.cache_root}", file=sys.stderr)
    shards = find_shards(args.cache_root)
    print(f"[prep_finance] found {len(shards)} shards", file=sys.stderr)

    # Pass 1: collect all eligible filings (10-K, year in scope, valid 30d label).
    # Dedupe by (cik, filing_date) — the JanosAudran source contains a small number
    # of exact duplicates (same company filed twice on the same date, e.g. PETRO USA
    # 2017-10-17). Keeping the first occurrence ensures train ∩ val = 0 by construction.
    seen_keys: set = set()
    by_year: Dict[int, List[Tuple[str, str, Dict]]] = {y: [] for y in args.years}
    n_total = 0
    n_dups_dropped = 0
    for company, filing in iter_filings(shards):
        if filing.get("form") != "10-K":
            continue
        date = filing.get("filingDate", "")
        if len(date) < 4 or not date[:4].isdigit():
            continue
        year = int(date[:4])
        if year not in by_year:
            continue
        label = filing.get("labels", {}).get(args.horizon)
        if label not in ("positive", "negative"):
            continue
        cik = str(company.get("cik", ""))
        key = (cik, date)
        if key in seen_keys:
            n_dups_dropped += 1
            continue
        seen_keys.add(key)
        by_year[year].append((company.get("name", ""), cik, filing))
        n_total += 1

    print(f"[prep_finance] eligible filings (10-K, label[{args.horizon}] valid): {n_total} "
          f"({n_dups_dropped} duplicates dropped)", file=sys.stderr)
    for y in args.years:
        print(f"  {y}: {len(by_year[y])}", file=sys.stderr)

    # Sanity: each year has at least n_train + n_val candidates.
    needed = args.n_train + args.n_val
    short = [y for y in args.years if len(by_year[y]) < needed]
    if short:
        print(f"error: years with fewer than n_train+n_val={needed} candidates: {short}",
              file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Pass 2: per year, shuffle once with seed, val first then train, write parquets.
    manifest_years = []
    for y in args.years:
        items = by_year[y]
        perm = rng.permutation(len(items))
        val_idx = perm[: args.n_val]
        train_idx = perm[args.n_val : args.n_val + args.n_train]

        val_rows = [
            build_row(year=y, company_name=items[i][0], cik=items[i][1],
                      filing=items[i][2], horizon=args.horizon,
                      text=filing_to_text(items[i][2], args.max_chars), idx=k)
            for k, i in enumerate(val_idx)
        ]
        train_rows = [
            build_row(year=y, company_name=items[i][0], cik=items[i][1],
                      filing=items[i][2], horizon=args.horizon,
                      text=filing_to_text(items[i][2], args.max_chars), idx=k)
            for k, i in enumerate(train_idx)
        ]

        train_path = args.out_dir / f"train_y{y}.parquet"
        val_path = args.out_dir / f"val_y{y}.parquet"
        pd.DataFrame(train_rows).to_parquet(train_path, index=False)
        pd.DataFrame(val_rows).to_parquet(val_path, index=False)

        # Per-year stats for the manifest.
        train_pos = sum(1 for r in train_rows if r["reward_model"]["ground_truth"] == "up")
        val_pos = sum(1 for r in val_rows if r["reward_model"]["ground_truth"] == "up")
        manifest_years.append({
            "year": y,
            "data_source": f"finance_yr_{y}",
            "n_train": len(train_rows),
            "n_val": len(val_rows),
            "n_pool": len(items),
            "train_up_frac": train_pos / max(1, len(train_rows)),
            "val_up_frac": val_pos / max(1, len(val_rows)),
        })
        print(f"  wrote {train_path.name} ({len(train_rows)}) + {val_path.name} ({len(val_rows)})  "
              f"train_up={train_pos}/{len(train_rows)} val_up={val_pos}/{len(val_rows)}",
              file=sys.stderr)

    # Manifest mirrors the temporalwiki one's shape so plotting can reuse infra.
    manifest = {
        "source": "JanosAudran/financial-reports-sec, large/train",
        "horizon": args.horizon,
        "max_chars": args.max_chars,
        "n_train": args.n_train,
        "n_val": args.n_val,
        "seed": args.seed,
        "years": list(args.years),
        "data_sources": [f"finance_yr_{y}" for y in args.years],
        "system_prompt": SYSTEM_PROMPT,
        "year_stats": manifest_years,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  wrote manifest.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
