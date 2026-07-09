#!/bin/bash
# Submit the finance ordering-F sweep: 4 methods × ordering F × 1 seed = 4 runs.
#
# Differences from sweep.sh:
#   - Locked to ordering F (6-year 10-K forward-sentiment chain).
#   - 1 epoch + test_freq=0: eval only at step 0 (val_before_train baseline) and at
#     the end of each phase. Minimum-info, fastest-signal mode — enough to populate
#     the past/future/baseline plot.
#   - skip_chain_best=1: keep only the chain_final loop. Each task starts from the
#     previous task's end-of-training checkpoint (continue-from-final), the right
#     semantics for a temporal-drift study (we want what the trained model does on
#     past/future years, not the per-year-val-best ckpt).
#
# Usage:
#   bash experiments/continual/sweep_finance.sh                    # all 4 methods, seed 42
#   METHODS=sft bash experiments/continual/sweep_finance.sh        # single method
#   SEEDS="42 17" bash experiments/continual/sweep_finance.sh      # multi-seed
#   DRY=1 bash experiments/continual/sweep_finance.sh              # print-only
#   SWEEP_TAG=my-tag bash experiments/continual/sweep_finance.sh   # fix the tag

set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"

read -r -a METHODS <<< "${METHODS:-sft sdpo grpo sdft}"
read -r -a SEEDS   <<< "${SEEDS:-42}"
THINK=${THINK:-false}
ORDERING=F

# Finance-specific cadence knobs. Override at submit time only if you know why.
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
TEST_FREQ=${TEST_FREQ:-0}
SKIP_CHAIN_BEST=${SKIP_CHAIN_BEST:-1}
SKIP_CHAIN_FINAL=${SKIP_CHAIN_FINAL:-0}
RESUME=${RESUME:-0}
NUM_TASKS=${NUM_TASKS:-}

SWEEP_TAG="${SWEEP_TAG:-finance-cl-$(date -u +%Y%m%d-%H%M%S)-$(printf '%04x' $RANDOM)}"
echo "[sweep_finance] tag=$SWEEP_TAG  methods=(${METHODS[*]})  ordering=$ORDERING  seeds=(${SEEDS[*]})  " \
     "epochs=$TOTAL_EPOCHS  test_freq=$TEST_FREQ  skip_chain_best=$SKIP_CHAIN_BEST  skip_chain_final=$SKIP_CHAIN_FINAL"

# Env vars consumed by run_sequential.sbatch. Use `--export=ALL` only; on some slurm
# builds `--export=ALL,VAR=val` causes a launch failure, so we propagate via env.
export TOTAL_EPOCHS TEST_FREQ SKIP_CHAIN_BEST
EXPORT_VARS="ALL"
count=0

for m in "${METHODS[@]}"; do
  for s in "${SEEDS[@]}"; do
    name="${m}_orderF_s${s}"
    args=("$m" "$ORDERING" "$s" "$THINK" "$SWEEP_TAG" "$SKIP_CHAIN_FINAL" "$RESUME" "${NUM_TASKS:-0}")
    cmd=(sbatch --job-name="$name" --export="$EXPORT_VARS" \
           run_sequential.sbatch "${args[@]}")
    if [[ "${DRY:-0}" == "1" ]]; then
      printf '%s\n' "${cmd[*]}"
    else
      "${cmd[@]}"
      # Stagger submissions so concurrent slurmd setup work lands in distinct
      # scheduler ticks (mitigates a cgroup-setup race seen on some clusters).
      remaining=$(( ${#METHODS[@]} * ${#SEEDS[@]} - count ))
      if (( remaining > 0 )); then
        echo "[sweep_finance] sleeping 90s before next submit"
        sleep 90
      fi
    fi
    count=$((count + 1))
  done
done
