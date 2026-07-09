#!/bin/bash
#SBATCH --job-name=sdpo_seq
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=7-00:00:00
#SBATCH --output=${CL_HOME:-/workspace/home/nayan}/sdpo_seq/logs/slurm_%x_%j.out
#SBATCH --error=${CL_HOME:-/workspace/home/nayan}/sdpo_seq/logs/slurm_%x_%j.err
#
# Sequential continual-learning launcher. Submits one job per (method, ordering, seed).
#
# Usage:
#   sbatch experiments/continual/run_sequential.sbatch <method> <ordering> <seed> [<enable_thinking>] [<sweep_tag>]
# Example:
#   sbatch experiments/continual/run_sequential.sbatch sdft A 42 false sweep-0422-a3f7
#
# Required env vars (set in your shell or a sourced env script):
#   SDPO_ROOT       absolute path to the SDPO repo checkout
#   WANDB_API_KEY   wandb auth (or set WANDB_MODE=offline)
#   VENV_ACTIVATE   (optional) path to a venv activate script to source
#
# --exclusive grabs the whole node; slurm sets CUDA_VISIBLE_DEVICES automatically.
# We avoid --gres=gpu:N + --cpus-per-task because cgroup constraints fail without
# CAP_SYS_ADMIN in some container runtimes.

set -euo pipefail

METHOD=${1:?usage: $0 <method> <ordering> <seed> [<enable_thinking=false>] [<sweep_tag>] [skip_chain_final=0|1] [resume=0|1] [num_tasks=0]}
ORDERING=${2:?ordering A|B|C}
SEED=${3:?seed}
ENABLE_THINKING=${4:-false}
SWEEP_TAG=${5:-}
# Optional positional knobs. Avoid `--export=ALL,VAR=val` on submission (some slurm
# builds fail on the inline assignment); these positionals + env vars are reliable.
SKIP_CHAIN_FINAL=${6:-${SKIP_CHAIN_FINAL:-0}}
RESUME=${7:-${RESUME:-0}}
NUM_TASKS=${8:-${NUM_TASKS:-}}

SDPO_ROOT=${CL_HOME:-/workspace/home/nayan}/SDPO
source ${CL_HOME:-/workspace/home/nayan}/sdpo_seq/env.sh
source ${CL_HOME:-/workspace/home/nayan}/venv-sdpo-v2/bin/activate

export WANDB_API_KEY="${WANDB_API_KEY:?set WANDB_API_KEY in your shell (https://wandb.ai/authorize)}"

# Slurm sets ROCR_VISIBLE_DEVICES alongside CUDA_VISIBLE_DEVICES; verl rejects the pair.
unset ROCR_VISIBLE_DEVICES

# Per-job Ray temp dir so workers can't pick up a stale /tmp/ray socket from a prior
# job's raylet, which manifests as "Failed to register worker to Raylet: IOError EOF".
ray stop --force 2>/dev/null || true
export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}"

# Do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True. vLLM's CuMemAllocator
# asserts against it (vllm/device_allocator/cumem.py). The other OOM mitigations
# (8-GPU FSDP, rollout.max_model_len cap, gpu_memory_utilization=0.45) are sufficient.

cd "$SDPO_ROOT"

EXTRA_ARGS=()
if [[ -n "${NUM_TASKS:-}" ]]; then EXTRA_ARGS+=(--num-tasks "$NUM_TASKS"); fi
if [[ "${SKIP_CHAIN_FINAL:-0}" == "1" ]]; then EXTRA_ARGS+=(--skip-chain-final); fi
if [[ "${SKIP_CHAIN_BEST:-0}" == "1" ]]; then EXTRA_ARGS+=(--skip-chain-best); fi
if [[ "${RESUME:-0}" == "1" ]]; then EXTRA_ARGS+=(--resume); fi
# Optional epoch / eval-cadence overrides (env-var only). Used for ordering F's
# initial sweep where we want 1 epoch + eval-only-at-end (matches smoke_finance.sbatch).
if [[ -n "${TOTAL_EPOCHS:-}" ]]; then EXTRA_ARGS+=(--total-epochs "$TOTAL_EPOCHS"); fi
if [[ -n "${TEST_FREQ:-}" ]]; then EXTRA_ARGS+=(--test-freq "$TEST_FREQ"); fi
if [[ -n "${VAL_N:-}" ]]; then EXTRA_ARGS+=(--val-n "$VAL_N"); fi
python experiments/continual/run_sequential.py \
    --method "$METHOD" \
    --ordering "$ORDERING" \
    --seed "$SEED" \
    --enable-thinking "$ENABLE_THINKING" \
    --sweep-tag "$SWEEP_TAG" \
    "${EXTRA_ARGS[@]}" \
    --model Qwen/Qwen3-8B \
    --n-gpus 8 \
    --nnodes 1

# Per-run analysis: produce summary JSON/MD + forgetting matrix / BWT-FWT / curves for
# both chains. RUN_TAG mirrors build_run_dir() in run_sequential.py; keep the two formulas
# in lockstep (sweep_tag is underscore-joined, think/nothink always present).
RUN_ROOT=${CL_HOME:-/workspace/home/nayan}/sdpo_seq/runs
case "${ENABLE_THINKING,,}" in true|1|yes) THINK_TAG=think ;; *) THINK_TAG=nothink ;; esac
BASE_TAG="${METHOD}_order${ORDERING}_${THINK_TAG}_s${SEED}"
RUN_TAG="${SWEEP_TAG:+${SWEEP_TAG}_}${BASE_TAG}"
RUN_DIR="$RUN_ROOT/$RUN_TAG"
echo "[analyze] RUN_DIR=$RUN_DIR"
if [[ -d "$RUN_DIR/chain_best" && -d "$RUN_DIR/chain_final" ]]; then
    # analyze.py takes run dirs positionally; each writes its own summary.json/.md + plots.
    # --results-dir emits the cross-chain aggregate (forgetting_matrix.csv, summary.md).
    python experiments/continual/analyze.py \
        "$RUN_DIR/chain_best" "$RUN_DIR/chain_final" \
        --results-dir "$RUN_DIR/analysis" || echo "analyze.py failed (non-fatal)"
else
    echo "analyze.py skipped: chain dirs not found under $RUN_DIR" >&2
fi
