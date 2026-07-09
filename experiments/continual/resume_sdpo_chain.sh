#!/bin/bash
# resume_sdpo_chain.sh — submit a slurm dependency chain that completes the
# remaining tasks of an SDPO ordering-F run, one task per slurm job.
#
# Why a chain of single-task jobs: SDPO's intra-job inter-task transition can fail
# with "Failed to register worker to Raylet: IOError EOF" the second time a Ray
# cluster is started inside the same slurm allocation. SFT/GRPO/SDFT do not trigger
# this; only SDPO's rollout.n=8 + max_model_len=18944 + self-distillation path leaves
# Ray state that the next intra-job task can't share cleanly. Putting each remaining
# task in its own slurm job (chained via --dependency=afterany) sidesteps the issue:
# every task starts fresh, and resume-skip in run_sequential.py reuses the prior
# tasks' final_hf checkpoints.
#
# Required env vars: SDPO_ROOT.
# Optional:          SDPO_RUN_ROOT, TAG, SEED, START.
#
# Usage:
#   bash $SDPO_ROOT/experiments/continual/resume_sdpo_chain.sh
#   TAG=foo SEED=42 START=3 bash .../resume_sdpo_chain.sh    # task 1+2 already done

set -euo pipefail

SDPO_ROOT=${CL_HOME:-/workspace/home/nayan}/SDPO
RUN_ROOT=${CL_HOME:-/workspace/home/nayan}/sdpo_seq/runs

TAG="${TAG:?set TAG to the sweep tag of the run to resume}"
SEED="${SEED:-42}"
RUN_DIR="$RUN_ROOT/${TAG}_sdpo_orderF_nothink_s${SEED}"
DETECTED_DONE=0
for i in 1 2 3 4 5 6; do
    YEAR=$((2014 + i))
    if [[ -d "$RUN_DIR/chain_final/task${i}_y${YEAR}_ckpts/final_hf" ]] && \
       compgen -G "$RUN_DIR/chain_final/task${i}_y${YEAR}_ckpts/final_hf/*.safetensors" > /dev/null; then
        DETECTED_DONE=$i
    else
        break
    fi
done
START="${START:-$((DETECTED_DONE + 1))}"

if [[ "$START" -gt 6 ]]; then
    echo "[resume_sdpo_chain] All 6 tasks already complete. Nothing to do."
    exit 0
fi

echo "[resume_sdpo_chain] tag=$TAG seed=$SEED"
echo "[resume_sdpo_chain] detected $DETECTED_DONE tasks done; queueing jobs for tasks $START..6"
echo

export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export TEST_FREQ="${TEST_FREQ:-0}"
export SKIP_CHAIN_BEST="${SKIP_CHAIN_BEST:-1}"

PREV=""
for N in $(seq "$START" 6); do
    DEP_ARG=""
    [[ -n "$PREV" ]] && DEP_ARG="--dependency=afterany:$PREV"
    JOB=$(sbatch --parsable --export=ALL \
        --job-name="sdpo_t${N}" \
        $DEP_ARG \
        "$SDPO_ROOT/experiments/continual/run_sequential.sbatch" \
        sdpo F "$SEED" false "$TAG" 0 1 "$N")
    echo "  queued task ${N} -> job $JOB${DEP_ARG:+ (after $PREV)}"
    PREV="$JOB"
done

echo
echo "[resume_sdpo_chain] $((6 - START + 1)) jobs queued sequentially via afterany dependency"
echo
echo "Watch progress:  squeue -u \"\$USER\""
echo "Latest task log: ls -t $RUN_DIR/chain_final/logs/task*.log | head -1"
