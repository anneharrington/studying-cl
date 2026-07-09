#!/bin/bash
#SBATCH --job-name=ttt_finance_smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=${CL_HOME:-/workspace/home/nayan}/sdpo_seq/logs/slurm_%x_%j.out
#SBATCH --error=${CL_HOME:-/workspace/home/nayan}/sdpo_seq/logs/slurm_%x_%j.err
#
# In-Place TTT finance — SMOKE TEST via slurm.
# Single phase (y2015 only), 50 train steps, ~30-45 min wall time. Verifies
# the data → pretrain → ckpt-convert → eval pipeline before committing to the
# full 6-phase production sweep.
#
# Submit:
#     sbatch /home/nayan/scripts/ttt_finance/run_chain_smoke.sbatch

set -euo pipefail

[[ -f ${CL_HOME:-/workspace/home/nayan}/sdpo_seq/env.sh ]] && source ${CL_HOME:-/workspace/home/nayan}/sdpo_seq/env.sh

unset ROCR_VISIBLE_DEVICES
ray stop --force 2>/dev/null || true

export WANDB_API_KEY="${WANDB_API_KEY:?set WANDB_API_KEY in your shell (https://wandb.ai/authorize)}"

# Smoke knobs: single phase, 50 steps so it finishes fast
export YEARS="${YEARS:-2015}"
export MAX_STEPS_PER_PHASE="${MAX_STEPS_PER_PHASE:-50}"

# Smoke runs share a fixed run tag so re-submits resume cleanly
export CARTRIDGES_FINANCE_RUN_TAG="${CARTRIDGES_FINANCE_RUN_TAG:-finance-cl-smoke_ttt_orderF_nothink_s42}"
echo "[sbatch] smoke RUN_TAG=${CARTRIDGES_FINANCE_RUN_TAG}"

bash /home/nayan/scripts/ttt_finance/run_chain.sh
