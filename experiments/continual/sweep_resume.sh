#!/bin/bash
# Submit resume jobs for failed runs from a prior sweep.
#
# run_sequential_resume.sbatch hardlink-copies the original run_dir into a new one
# tagged "<orig>-resume", then invokes run_sequential.py --resume which skips any
# task whose final_hf is already on disk.
#
# Usage:
#   ORIG_SWEEP_TAG=<tag> bash experiments/continual/sweep_resume.sh
#   ORDERINGS="A B C" METHODS="sdpo grpo" ORIG_SWEEP_TAG=<tag> bash .../sweep_resume.sh
#   DRY=1 ORIG_SWEEP_TAG=<tag> bash experiments/continual/sweep_resume.sh

set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"

ORIG_SWEEP_TAG=${ORIG_SWEEP_TAG:?set ORIG_SWEEP_TAG to the sweep tag to resume}
NEW_SWEEP_TAG=${NEW_SWEEP_TAG:-${ORIG_SWEEP_TAG}-resume}

read -r -a METHODS   <<< "${METHODS:-sdpo}"
read -r -a ORDERINGS <<< "${ORDERINGS:-A B}"
read -r -a SEEDS     <<< "${SEEDS:-42}"
THINK=${THINK:-false}

echo "[resume] orig=$ORIG_SWEEP_TAG  new=$NEW_SWEEP_TAG  methods=(${METHODS[*]})  orderings=(${ORDERINGS[*]})"

for m in "${METHODS[@]}"; do
  for o in "${ORDERINGS[@]}"; do
    for s in "${SEEDS[@]}"; do
      name="${m}_order${o}_s${s}_resume"
      cmd=(sbatch --job-name="$name" run_sequential_resume.sbatch \
           "$m" "$o" "$s" "$THINK" "$NEW_SWEEP_TAG" "$ORIG_SWEEP_TAG")
      if [[ "${DRY:-0}" == "1" ]]; then
        printf '%s\n' "${cmd[*]}"
      else
        "${cmd[@]}"
      fi
    done
  done
done
