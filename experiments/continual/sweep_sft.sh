#!/bin/bash
# Submit SFT baseline runs into an existing sweep tag (so SFT rows live alongside the
# sdft / sdpo / grpo rows when analyze.py aggregates).
#
# Usage:
#   SWEEP_TAG=<tag> bash experiments/continual/sweep_sft.sh                # all 3 orderings
#   ORDERINGS=A SWEEP_TAG=<tag> bash experiments/continual/sweep_sft.sh    # single ordering
#   DRY=1 SWEEP_TAG=<tag> bash experiments/continual/sweep_sft.sh          # print only

set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"

SWEEP_TAG=${SWEEP_TAG:?set SWEEP_TAG to the sweep this SFT batch should join}
read -r -a ORDERINGS <<< "${ORDERINGS:-A B C}"
read -r -a SEEDS     <<< "${SEEDS:-42}"
THINK=${THINK:-false}

echo "[sweep_sft] tag=$SWEEP_TAG  orderings=(${ORDERINGS[*]})  seeds=(${SEEDS[*]})  think=$THINK"

for o in "${ORDERINGS[@]}"; do
  for s in "${SEEDS[@]}"; do
    name="sft_order${o}_s${s}"
    cmd=(sbatch --job-name="$name" run_sequential.sbatch "sft" "$o" "$s" "$THINK" "$SWEEP_TAG")
    if [[ "${DRY:-0}" == "1" ]]; then
      printf '%s\n' "${cmd[*]}"
    else
      "${cmd[@]}"
    fi
  done
done
