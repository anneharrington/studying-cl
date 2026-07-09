#!/bin/bash
#SBATCH --job-name=cl_sft_smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=runs/logs/slurm_%x_%j.out
#SBATCH --error=runs/logs/slurm_%x_%j.err
#
# Minimal 1-step end-to-end smoke for the continual-learning trainer
# (experiments/continual/run_sequential.py).
#   - method=sft, ordering=A (domain-shift: bio → finqa → tooluse)
#   - 1 task (bio), 32 train samples, 1 epoch ⇒ 1 train step
#   - skips chain_final (only chain_best built)
#
# Confirms: data adapter, verl Hydra composition, FSDP init, the patched
# verl engine, save/load round-trip. ~15-20 min wall-clock.
#
# Required env in the calling shell (or sbatch --export):
#   WANDB_API_KEY     set, or `export WANDB_MODE=offline` before sbatch
set -uo pipefail

# Repo root from this script's location — survives clone-and-move.
SCRIPTS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPTS/../.." && pwd)
cd "$REPO"
mkdir -p "$REPO/runs/logs"

# Activate the verl venv via the documented `.venv-verl` convention (real dir
# or symlink). See docs/SETUP.md §2.
[ -f "$REPO/.venv-verl/bin/activate" ] || {
    echo "!! $REPO/.venv-verl not found — see docs/SETUP.md §2 (verl venv)"
    exit 2
}
# shellcheck source=/dev/null
source "$REPO/.venv-verl/bin/activate"

# Patched verl engine lives at engine/verl — runs as `python -m verl.trainer.main_ppo`
# from a subprocess, so PYTHONPATH must be exported (not just sys.path).
export PYTHONPATH="$REPO/engine:${PYTHONPATH:-}"

# WANDB: smoke is offline unless caller already set WANDB_API_KEY.
export WANDB_MODE=${WANDB_MODE:-offline}

# Ray hygiene: stale /tmp/ray socket from a prior failed run breaks Raylet
# worker registration with "Failed to register worker to Raylet: IOError EOF".
# Per-job RAY_TMPDIR + a fresh /tmp/ray sidestep that.
unset ROCR_VISIBLE_DEVICES
rm -rf /tmp/ray /tmp/ray_* 2>/dev/null || true
ray stop --force 2>/dev/null || true
export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID:-$$}"
# Default fd cap (1024) starves Ray/vLLM workers; raise it.
ulimit -n 65535 2>/dev/null || true

# Idempotent: wipe any prior smoke run.
rm -rf "$REPO/runs/sft_orderA_nothink_s42"

echo "[smoke] $(date -u)  starting 1-step SFT on bio (orderA)"
python experiments/continual/run_sequential.py \
    --method sft --ordering A --seed 42 \
    --num-tasks 1 --train-samples 32 --train-batch 32 --total-epochs 1 \
    --skip-chain-final
rc=$?
echo "[smoke] $(date -u)  done (rc=$rc)"
exit $rc
