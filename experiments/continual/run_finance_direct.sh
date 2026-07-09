#!/bin/bash
# Direct (non-slurm) launcher for one finance ordering-F method on this host.
# Same python pipeline as run_sequential.sbatch, without sbatch — useful when the
# cluster's slurm/cgroup setup is unavailable.
#
# Single-method, single-host, 8 GPUs. Backgrounds the python process and detaches
# so the shell can be closed. Logs to <run_dir>/manual.log.
#
# Required env vars: SDPO_ROOT, WANDB_API_KEY (or WANDB_MODE=offline).
# Optional:          SDPO_RUN_ROOT, VENV_ACTIVATE, SDPO_ENV.
#
# Usage:
#   bash experiments/continual/run_finance_direct.sh <method> [<seed>] [<sweep_tag>]
# Examples:
#   bash experiments/continual/run_finance_direct.sh sft
#   bash experiments/continual/run_finance_direct.sh sdpo 42 finance-direct
#   for m in sft sdpo grpo sdft; do
#     bash experiments/continual/run_finance_direct.sh $m  # sequential
#   done

set -euo pipefail

METHOD=${1:?usage: $0 <method> [<seed>] [<sweep_tag>]}
SEED=${2:-42}
SWEEP_TAG=${3:-finance-direct}
ENABLE_THINKING=${ENABLE_THINKING:-false}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
TEST_FREQ=${TEST_FREQ:-0}

SDPO_ROOT=${CL_HOME:-/workspace/home/nayan}/SDPO
source ${CL_HOME:-/workspace/home/nayan}/sdpo_seq/env.sh
source ${CL_HOME:-/workspace/home/nayan}/venv-sdpo-v2/bin/activate
export WANDB_API_KEY="${WANDB_API_KEY:?set WANDB_API_KEY in your shell (https://wandb.ai/authorize)}"

# Slurm sometimes sets ROCR_VISIBLE_DEVICES alongside CUDA_VISIBLE_DEVICES; verl rejects the pair.
unset ROCR_VISIBLE_DEVICES

# Per-run Ray temp dir to avoid colliding with another concurrent run on the host.
ray stop --force 2>/dev/null || true
RAY_TS=$(date +%s)
export RAY_TMPDIR="/tmp/ray_direct_${METHOD}_${RAY_TS}"

cd "$SDPO_ROOT"

# Resolve where logs will live (mirror build_run_dir() in run_sequential.py)
RUN_TAG="${SWEEP_TAG}_${METHOD}_orderF_nothink_s${SEED}"
RUN_DIR=${CL_HOME:-/workspace/home/nayan}/sdpo_seq/runs/$RUN_TAG
mkdir -p "$RUN_DIR/logs"
LOG="$RUN_DIR/manual.log"

echo "[direct] $(date -u)  method=$METHOD ordering=F seed=$SEED" | tee -a "$LOG"
echo "[direct] run_dir=$RUN_DIR" | tee -a "$LOG"
echo "[direct] log=$LOG" | tee -a "$LOG"

# nohup so the python survives shell disconnect; --skip-chain-best for chain_final-only.
nohup python experiments/continual/run_sequential.py \
    --method "$METHOD" \
    --ordering F \
    --seed "$SEED" \
    --enable-thinking "$ENABLE_THINKING" \
    --sweep-tag "$SWEEP_TAG" \
    --total-epochs "$TOTAL_EPOCHS" \
    --test-freq "$TEST_FREQ" \
    --skip-chain-best \
    --model Qwen/Qwen3-8B \
    --n-gpus 8 \
    --nnodes 1 \
    >> "$LOG" 2>&1 &

PID=$!
echo "[direct] launched pid=$PID, detaching" | tee -a "$LOG"
disown $PID
echo
echo "Watch with:  tail -F $LOG"
echo "Stop with :  kill $PID"
