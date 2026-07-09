#!/bin/bash
#SBATCH --job-name=ttt_finance
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=${CL_HOME:-/workspace/home/nayan}/sdpo_seq/logs/slurm_%x_%j.out
#SBATCH --error=${CL_HOME:-/workspace/home/nayan}/sdpo_seq/logs/slurm_%x_%j.err
#
# In-Place TTT finance ordering-F sweep — full chain via slurm.
# Wraps run_chain.sh; same env-var knobs apply, just submit instead of bash.
#
# Submit (production, defaults match our SDPO context). Use SHARED-FS path so
# both nodes see the same scripts (per-node /home is local):
#     sbatch ${CL_HOME:-/workspace/home/nayan}/_sync/ttt_finance/run_chain.sbatch
#
# Override knobs at submit time:
#     MAX_STEPS_PER_PHASE=200 sbatch ...     # cheaper / shorter pretrain
#     YEARS=2015 sbatch ...                  # single-phase smoke
#     CARTRIDGES_FINANCE_RUN_TAG=foo sbatch ...  # resume into existing run dir
#
# Logs:
#     ${CL_HOME:-/workspace/home/nayan}/sdpo_seq/logs/slurm_ttt_finance_<jobid>.{out,err}
#     ${CL_HOME:-/workspace/home/nayan}/ttt_finance/runs/<RUN_TAG>/logs/

set -euo pipefail

[[ -f ${CL_HOME:-/workspace/home/nayan}/sdpo_seq/env.sh ]] && source ${CL_HOME:-/workspace/home/nayan}/sdpo_seq/env.sh

unset ROCR_VISIBLE_DEVICES
ray stop --force 2>/dev/null || true

export WANDB_API_KEY="${WANDB_API_KEY:?set WANDB_API_KEY in your shell (https://wandb.ai/authorize)}"

# Run-tag in SDPO repo's convention. Each prod submission gets a fresh
# stamp so runs don't collide; pass an explicit CARTRIDGES_FINANCE_RUN_TAG
# (overloaded for ttt for env-var compatibility) to override (e.g. resume).
if [[ -z "${CARTRIDGES_FINANCE_RUN_TAG:-}" ]]; then
    SHA=$(git -C ${CL_HOME:-/workspace/home/nayan}/methods/In-Place-TTT rev-parse --short=4 HEAD 2>/dev/null || echo "nogit")
    TS=$(date -u +%Y%m%d-%H%M%S)
    export CARTRIDGES_FINANCE_RUN_TAG="finance-cl-${TS}-${SHA}_ttt_orderF_nothink_s42"
fi
echo "[sbatch] CARTRIDGES_FINANCE_RUN_TAG=${CARTRIDGES_FINANCE_RUN_TAG}"

bash ${CL_HOME:-/workspace/home/nayan}/_sync/ttt_finance/run_chain.sh
