#!/bin/bash
# Submit the 9 runs for this-week's sweep: 3 methods × 2 orderings × 1 model cfg × 1 seed = 6;
# NOTE: the user's plan says "couple of orderings" — we submit A and B. If we want C too,
# uncomment the third entry in ORDERINGS below.
#
# Each job reserves 4 GPUs, so Slurm co-schedules 2 jobs per 8-GPU node; with 2 nodes we
# run 4 concurrent jobs, burning through the 6 (or 9) runs in ~1.5 waves.
#
# Every invocation of this script auto-generates a SWEEP_TAG (UTC timestamp + rand4) that
# is shared across all jobs it submits -- so one sweep = one tag, regardless of how many
# (method, ordering, seed) combos you select. That tag prefixes run_tag (see
# run_sequential.py), so runs/<tag>_<base>/ dirs never collide across reruns and wandb
# groups stay clean. Export SWEEP_TAG=... before the call to override (e.g. to resume a
# specific sweep).
#
# Usage:
#   bash experiments/continual/sweep.sh                          # submit all 6
#   METHODS=sdft ORDERINGS=A bash experiments/continual/sweep.sh # single run, still tagged
#   METHODS="sdft sdpo" bash experiments/continual/sweep.sh      # subset (space-separated)
#   DRY=1 bash experiments/continual/sweep.sh                    # print sbatch cmds only
#   SWEEP_TAG=my-tag bash experiments/continual/sweep.sh         # fix the tag explicitly

set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"

# Env overrides arrive as strings (bash inline env-vars can't be arrays); split on whitespace.
read -r -a METHODS   <<< "${METHODS:-sdft sdpo grpo}"
read -r -a ORDERINGS <<< "${ORDERINGS:-A B C}"         # A=bio-first, B=tooluse-first, C=finqa-first (each task heads one run)
read -r -a SEEDS     <<< "${SEEDS:-42}"
THINK=${THINK:-false}                                  # this week: qwen3-8b NO thinking

SWEEP_TAG="${SWEEP_TAG:-sweep-$(date -u +%Y%m%d-%H%M%S)-$(printf '%04x' $RANDOM)}"
echo "[sweep] tag=$SWEEP_TAG  methods=(${METHODS[*]})  orderings=(${ORDERINGS[*]})  seeds=(${SEEDS[*]})  think=$THINK"

for m in "${METHODS[@]}"; do
  for o in "${ORDERINGS[@]}"; do
    for s in "${SEEDS[@]}"; do
      name="${m}_order${o}_s${s}"
      # Always submit as user nayan (wandb attribution, slurm accounting). If invoker is
      # already nayan, sudo is a no-op wrapper.
      if [[ "$(id -un)" == "nayan" ]]; then
        cmd=(sbatch --job-name="$name" run_sequential.sbatch "$m" "$o" "$s" "$THINK" "$SWEEP_TAG")
      else
        cmd=(sudo -n -u nayan sbatch --job-name="$name" "$PWD/run_sequential.sbatch" "$m" "$o" "$s" "$THINK" "$SWEEP_TAG")
      fi
      if [[ "${DRY:-0}" == "1" ]]; then
        printf '%s\n' "${cmd[*]}"
      else
        "${cmd[@]}"
      fi
    done
  done
done
