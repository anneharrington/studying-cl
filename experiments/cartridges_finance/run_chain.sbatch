#!/bin/bash
#SBATCH --job-name=cartridges_finance
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=${CL_HOME:-/workspace/home/nayan}/sdpo_seq/logs/slurm_%x_%j.out
#SBATCH --error=${CL_HOME:-/workspace/home/nayan}/sdpo_seq/logs/slurm_%x_%j.err
#
# Cartridges finance ordering-F sweep — full chain via slurm.
# Wraps run_chain.sh; same env-var knobs apply, just submit instead of bash.
#
# Submit (production, defaults match SDPO context + paper-faithful):
#     sbatch /home/nayan/scripts/cartridges_finance/run_chain.sbatch
#
# Override knobs at submit time:
#     CARTRIDGES_FINANCE_N_SYNTH=4096 sbatch ...    # cheaper synth
#     YEARS=2015 sbatch ...                         # single-phase smoke
#
# Logs:
#     ${CL_HOME:-/workspace/home/nayan}/sdpo_seq/logs/slurm_cartridges_finance_<jobid>.{out,err}
#     ${CL_HOME:-/workspace/home/nayan}/cartridges_finance/runs/finance-cl-cartridges_orderF_nothink_s42/logs/

set -euo pipefail

# Match the verl-sbatch pattern: rely on env script + venv from shared FS so
# the same submission works whether slurm lands on node-0 or node-1.
[[ -f ${CL_HOME:-/workspace/home/nayan}/sdpo_seq/env.sh ]] && source ${CL_HOME:-/workspace/home/nayan}/sdpo_seq/env.sh

unset ROCR_VISIBLE_DEVICES
ray stop --force 2>/dev/null || true

export WANDB_API_KEY="${WANDB_API_KEY:?set WANDB_API_KEY in your shell (https://wandb.ai/authorize)}"

# Run-tag in the SDPO repo's convention:
#   finance-cl-<YYYYMMDD>-<HHMMSS>-<SHA4>_cartridges_orderF_nothink_s42
# matches existing SDPO runs (e.g. finance-cl-20260429-153308-4807_sdpo_orderF_nothink_s42).
# Each prod submission gets a fresh stamp so runs don't collide; pass an
# explicit CARTRIDGES_FINANCE_RUN_TAG to override (e.g. to resume).
if [[ -z "${CARTRIDGES_FINANCE_RUN_TAG:-}" ]]; then
    SHA=$(git -C /home/nayan/methods/cartridges rev-parse --short=4 HEAD 2>/dev/null || echo "nogit")
    TS=$(date -u +%Y%m%d-%H%M%S)
    export CARTRIDGES_FINANCE_RUN_TAG="finance-cl-${TS}-${SHA}_cartridges_orderF_nothink_s42"
fi
echo "[sbatch] CARTRIDGES_FINANCE_RUN_TAG=${CARTRIDGES_FINANCE_RUN_TAG}"

bash /home/nayan/scripts/cartridges_finance/run_chain.sh
