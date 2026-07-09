#!/bin/bash
#SBATCH --job-name=cl_sft_smoke_F
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=runs/logs/slurm_%x_%j.out
#SBATCH --error=runs/logs/slurm_%x_%j.err
#
# Minimal 1-step end-to-end smoke for the temporal_drift family:
#   - method=sft, ordering=F (SEC 10-K finance, 2015-2020)
#   - 1 task (y2015), 32 train samples, 1 epoch => 1 train step
#
# Required env in the calling shell (or sbatch --export):
#   CL_FINANCE_DATA   absolute path to the per-year finance parquets
#                     (built by data/prep/finance/, or symlink an existing copy)
#   WANDB_API_KEY     set, or `export WANDB_MODE=offline` before sbatch

set -uo pipefail

SCRIPTS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPTS/../.." && pwd)
cd "$REPO"
mkdir -p "$REPO/runs/logs"

[ -f "$REPO/.venv-verl/bin/activate" ] || {
    echo "!! $REPO/.venv-verl not found - see docs/SETUP.md S2 (verl venv)"
    exit 2
}
# shellcheck source=/dev/null
source "$REPO/.venv-verl/bin/activate"

: "${CL_FINANCE_DATA:?set CL_FINANCE_DATA to per-year finance parquets dir (see data/prep/finance/)}"

export PYTHONPATH="$REPO/engine:${PYTHONPATH:-}"
export WANDB_MODE=${WANDB_MODE:-offline}

unset ROCR_VISIBLE_DEVICES
rm -rf /tmp/ray /tmp/ray_* 2>/dev/null || true
ray stop --force 2>/dev/null || true
export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID:-$$}"
ulimit -n 65535 2>/dev/null || true

rm -rf "$REPO/runs/sft_orderF_nothink_s42"

echo "[smoke-F] $(date -u)  1-step SFT on y2015 (finance)"
python experiments/continual/run_sequential.py \
    --method sft --ordering F --seed 42 \
    --num-tasks 1 --train-samples 32 --train-batch 32 --total-epochs 1 \
    --skip-chain-final
rc=$?
echo "[smoke-F] $(date -u)  done (rc=$rc)"
exit $rc
