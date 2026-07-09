#!/bin/bash
#SBATCH --job-name=cartridges_smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=runs/logs/slurm_%x_%j.out
#SBATCH --error=runs/logs/slurm_%x_%j.err
#
# Minimal end-to-end Cartridges smoke on the finance ordering-F chain:
#   - 1 year (y2015)
#   - N_SYNTH=64 (vs 8192 for real runs)
#   - 1 epoch
#   - SKIP_EVAL=1 (no OPENROUTER_API_KEY needed)
#   - WANDB offline
#
# Exercises: corpus build, Tokasaurus boot, synth against live server,
# server teardown + GPU drain, cartridge distillation, checkpoint write.
# Total ~25-40 min on a fresh node (Tokasaurus cold boot ~10 min, warm ~2 min).
#
# Required env in the calling shell:
#   CL_FINANCE_DATA   absolute path to the per-year finance parquets
#                     (built by data/prep/finance/, or symlink an existing copy)
# Optional:
#   CARTRIDGES_FINANCE_RUN_DIR   override the per-job scratch dir
#                                (defaults to $REPO/runs/cartridges_smoke/run_<jobid>)

set -uo pipefail

# Repo root from this script's location.
SCRIPTS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPTS/../.." && pwd)
cd "$REPO"
mkdir -p "$REPO/runs/logs"

: "${CL_FINANCE_DATA:?set CL_FINANCE_DATA to per-year finance parquets dir (see data/prep/finance/)}"

# WANDB: smoke is offline.
export WANDB_MODE=offline
export CARTRIDGES_WANDB_DISABLE=1
unset ROCR_VISIBLE_DEVICES

# Smoke knobs: one year, tiny synth, no eval.
export YEARS="2015"
export CARTRIDGES_FINANCE_N_SYNTH=64
export CARTRIDGES_FINANCE_EPOCHS=1
export SKIP_EVAL=1

# Per-job RUN_DIR so reruns don't pick up stale state. The chain reads
# CARTRIDGES_FINANCE_RUN_DIR (not RUN_DIR), so override that one.
export CARTRIDGES_FINANCE_RUN_DIR=${CARTRIDGES_FINANCE_RUN_DIR:-$REPO/runs/cartridges_smoke/run_${SLURM_JOB_ID:-$$}}
mkdir -p "$CARTRIDGES_FINANCE_RUN_DIR"

echo "[smoke] $(date -u)  starting cartridges finance smoke (year=2015, n_synth=64)"
bash experiments/cartridges_finance/run_chain.sh
rc=$?
echo "[smoke] $(date -u)  done (rc=$rc)"
exit $rc
