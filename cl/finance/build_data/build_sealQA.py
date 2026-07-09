#!/usr/bin/env python3
"""Download all SealQA parquet files from Hugging Face dataset repo.

Source:
  https://huggingface.co/datasets/vtllms/sealqa/tree/main

Example:
  python3 data/build_sealQA.py \
    --dataset vtllms/sealqa \
    --revision main \
    --out-dir data/sealQA/parquet
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download all .parquet files from a HF dataset repo.")
    parser.add_argument("--dataset", type=str, default="vtllms/sealqa", help="HF dataset id")
    parser.add_argument("--revision", type=str, default="main", help="Branch/tag/commit")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/sealQA/parquet"),
        help="Destination directory for parquet files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from huggingface_hub import hf_hub_download, list_repo_files  # type: ignore

    repo_files = list_repo_files(
        repo_id=args.dataset,
        repo_type="dataset",
        revision=args.revision,
    )
    parquet_files = sorted(f for f in repo_files if f.lower().endswith(".parquet"))

    if not parquet_files:
        raise RuntimeError(f"No .parquet files found in dataset repo: {args.dataset}@{args.revision}")

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    for rel_path in parquet_files:
        cached_path = hf_hub_download(
            repo_id=args.dataset,
            repo_type="dataset",
            filename=rel_path,
            revision=args.revision,
        )

        # Preserve subfolder structure from repo under out-dir.
        target_path = out_dir / rel_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached_path, target_path)
        downloaded += 1

    print(f"Dataset: {args.dataset}@{args.revision}")
    print(f"Downloaded parquet files: {downloaded}")
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
