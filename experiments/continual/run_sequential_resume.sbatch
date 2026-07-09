#!/bin/bash
#SBATCH --job-name=sdpo_seq_resume
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=7-00:00:00
#SBATCH --output=${CL_HOME:-/workspace/home/nayan}/sdpo_seq/logs/slurm_%x_%j.out
#SBATCH --error=${CL_HOME:-/workspace/home/nayan}/sdpo_seq/logs/slurm_%x_%j.err
#
# Resume a partially-failed run from a prior sweep into a new sweep_tag (e.g.
# `<old_tag>-resume`). Hardlink-copies the old run_dir to a new one (so existing
# checkpoints are preserved untouched), then invokes run_sequential.py --resume which
# skips tasks whose final_hf already exists.
#
# Usage:
#   sbatch run_sequential_resume.sbatch <method> <ordering> <seed> <enable_thinking> \
#                                       <new_sweep_tag> <orig_sweep_tag>
#
# Required env vars: SDPO_ROOT, WANDB_API_KEY (or WANDB_MODE=offline).
# Optional:          SDPO_RUN_ROOT (default $SDPO_ROOT/runs), VENV_ACTIVATE, SDPO_ENV.

set -euo pipefail

METHOD=${1:?usage: $0 <method> <ordering> <seed> <enable_thinking> <new_sweep_tag> <orig_sweep_tag> [skip_chain_final=0|1]}
ORDERING=${2:?ordering A|B|C}
SEED=${3:?seed}
ENABLE_THINKING=${4:-false}
SWEEP_TAG=${5:?new sweep_tag (e.g. <orig>-resume)}
ORIG_SWEEP_TAG=${6:?original sweep_tag whose run_dir we are resuming from}
SKIP_CHAIN_FINAL=${7:-0}  # positional: 1 = pass --skip-chain-final to run_sequential.py

SDPO_ROOT=${CL_HOME:-/workspace/home/nayan}/SDPO
RUN_ROOT=${CL_HOME:-/workspace/home/nayan}/sdpo_seq/runs

source ${CL_HOME:-/workspace/home/nayan}/sdpo_seq/env.sh
source ${CL_HOME:-/workspace/home/nayan}/venv-sdpo-v2/bin/activate

export WANDB_API_KEY="${WANDB_API_KEY:?set WANDB_API_KEY in your shell (https://wandb.ai/authorize)}"

unset ROCR_VISIBLE_DEVICES
ray stop --force 2>/dev/null || true
export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}"

case "${ENABLE_THINKING,,}" in true|1|yes) THINK_TAG=think ;; *) THINK_TAG=nothink ;; esac
BASE_TAG="${METHOD}_order${ORDERING}_${THINK_TAG}_s${SEED}"
RUN_TAG="${SWEEP_TAG}_${BASE_TAG}"
ORIG_RUN_TAG="${ORIG_SWEEP_TAG}_${BASE_TAG}"

NEW_RUN_DIR="${RUN_ROOT}/${RUN_TAG}"
ORIG_RUN_DIR="${RUN_ROOT}/${ORIG_RUN_TAG}"

# Hardlink-copy the original run_dir into the new one if not already done. cp -al keeps the
# original ckpts intact (separate dir entry, same inode); subsequent _hardlink_swap and
# rmtree calls in run_sequential.py operate on NEW_RUN_DIR only and won't touch the original.
if [[ ! -d "$NEW_RUN_DIR" ]]; then
    if [[ ! -d "$ORIG_RUN_DIR" ]]; then
        echo "[resume] no original run_dir at $ORIG_RUN_DIR; will start fresh under $NEW_RUN_DIR"
        mkdir -p "$NEW_RUN_DIR"
    else
        echo "[resume] hardlink-copy $ORIG_RUN_DIR -> $NEW_RUN_DIR"
        cp -al "$ORIG_RUN_DIR" "$NEW_RUN_DIR"
        # The original DONE flag (if any) doesn't apply to the new run.
        rm -f "$NEW_RUN_DIR/DONE"
    fi
fi

cd "$SDPO_ROOT"
EXTRA_ARGS=()
if [[ "${SKIP_CHAIN_FINAL:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--skip-chain-final)
fi
python experiments/continual/run_sequential.py \
    --method "$METHOD" \
    --ordering "$ORDERING" \
    --seed "$SEED" \
    --enable-thinking "$ENABLE_THINKING" \
    --sweep-tag "$SWEEP_TAG" \
    --resume \
    "${EXTRA_ARGS[@]}" \
    --model Qwen/Qwen3-8B \
    --n-gpus 8 \
    --nnodes 1

# Same analysis post-step as run_sequential.sbatch.
RUN_DIR="$NEW_RUN_DIR"
echo "[analyze] RUN_DIR=$RUN_DIR"
if [[ -d "$RUN_DIR/chain_best" && -d "$RUN_DIR/chain_final" ]]; then
    python experiments/continual/analyze.py \
        "$RUN_DIR/chain_best" "$RUN_DIR/chain_final" \
        --results-dir "$RUN_DIR/analysis" || echo "analyze.py failed (non-fatal)"
else
    echo "analyze.py skipped: chain dirs not found under $RUN_DIR" >&2
fi
