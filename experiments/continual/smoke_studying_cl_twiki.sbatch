#!/bin/bash
#SBATCH --job-name=cl_sft_smoke_T
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=runs/logs/slurm_%x_%j.out
#SBATCH --error=runs/logs/slurm_%x_%j.err
#
# Minimal 1-step end-to-end smoke for the discrete_updates family:
#   - method=sft, ordering=T (TemporalWiki drift ts1 -> ts2 -> ts3)
#   - 1 task (ts1), 32 train samples, 1 epoch => 1 train step
#
# Required env in the calling shell (or sbatch --export):
#   CL_TEMPORAL_DRIFT_DATA   path to TemporalWiki drift parquets (see data/prep/)
#   WANDB_API_KEY            set, or `export WANDB_MODE=offline` before sbatch

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

: "${CL_TEMPORAL_DRIFT_DATA:?set CL_TEMPORAL_DRIFT_DATA to TemporalWiki drift parquets dir (see data/prep/temporalwiki/)}"

export PYTHONPATH="$REPO/engine:${PYTHONPATH:-}"
export WANDB_MODE=${WANDB_MODE:-offline}

unset ROCR_VISIBLE_DEVICES
rm -rf /tmp/ray /tmp/ray_* 2>/dev/null || true
ray stop --force 2>/dev/null || true
export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID:-$$}"
ulimit -n 65535 2>/dev/null || true

rm -rf "$REPO/runs/sft_orderT_nothink_s42"

echo "[smoke-T] $(date -u)  1-step SFT on ts1 (TemporalWiki)"
python experiments/continual/run_sequential.py \
    --method sft --ordering T --seed 42 \
    --num-tasks 1 --train-samples 32 --train-batch 32 --total-epochs 1 \
    --skip-chain-final
rc=$?
echo "[smoke-T] $(date -u)  done (rc=$rc)"
exit $rc
