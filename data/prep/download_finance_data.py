#!/usr/bin/env python3
"""Download JanosAudran/financial-reports-sec shards from HuggingFace Hub.

Pulls the `large/train` split by default (10 shards, ~11 GB total) and caches
to the standard HF cache. Idempotent — re-runs skip already-downloaded shards.

Usage:
    python scripts/data/download_finance_data.py
    python scripts/data/download_finance_data.py --split test
    python scripts/data/download_finance_data.py --subset small --split train
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ID = "JanosAudran/financial-reports-sec"
DEFAULT_SUBSET = "large"     # "small" (~600 filings) or "large" (~6k+ filings)
DEFAULT_SPLIT = "train"      # "train" / "validate" / "test"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default=DEFAULT_SUBSET, choices=("small", "large"))
    ap.add_argument("--split", default=DEFAULT_SPLIT, choices=("train", "validate", "test"))
    args = ap.parse_args()

    # Avoid hf_transfer if not installed (silent default would otherwise crash).
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    all_files = api.list_repo_files(REPO_ID, repo_type="dataset")
    prefix = f"data/{args.subset}/{args.split}/"
    shards = sorted(f for f in all_files if f.startswith(prefix))
    if not shards:
        print(f"No shards found for prefix {prefix!r}", file=sys.stderr)
        return 2

    print(f"Downloading {len(shards)} shards ({args.subset}/{args.split}) from {REPO_ID}", file=sys.stderr)
    t0 = time.time()
    total_mb = 0.0
    for i, fn in enumerate(shards, start=1):
        local = hf_hub_download(repo_id=REPO_ID, filename=fn, repo_type="dataset")
        sz_mb = os.path.getsize(local) / 1024 / 1024
        total_mb += sz_mb
        print(f"  [{i:2d}/{len(shards)}] {fn}  ->  {sz_mb:6.0f} MB  "
              f"(cum {total_mb:6.0f} MB, {time.time()-t0:5.0f}s)", file=sys.stderr)

    print(f"\nDone. {total_mb:.0f} MB ({total_mb/1024:.2f} GB) cached.", file=sys.stderr)
    cache_dir = Path(local).parent
    print(f"Cache path: {cache_dir}", file=sys.stderr)

    # Quick post-download verification: expected shard count + nonzero sizes.
    on_disk = sorted(cache_dir.glob("shard_*.jsonl"))
    if len(on_disk) != len(shards):
        print(f"WARNING: expected {len(shards)} shards on disk, found {len(on_disk)}", file=sys.stderr)
        return 1
    for p in on_disk:
        if p.stat().st_size == 0:
            print(f"WARNING: zero-size shard {p}", file=sys.stderr)
            return 1
    print(f"Verified: {len(on_disk)} shards, all non-empty.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
