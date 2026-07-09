#!/bin/bash
#SBATCH --job-name=cartridges_twiki
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=${CL_HOME:-/workspace/home/nayan}/sdpo_seq/logs/slurm_%x_%j.out
#SBATCH --error=${CL_HOME:-/workspace/home/nayan}/sdpo_seq/logs/slurm_%x_%j.err
#
# Cartridges TemporalWiki ordering-T sweep — full chain via slurm.
# 3 phases (ts1 → ts2 → ts3), 12 eval cells (3 phases × 4 eval sets).
#
# Submit (production) — invoke from the SHARED /workspace path so node-0 and
# node-1 see the same scripts (per-node /home is local, breaks cross-node
# launches). The sbatch + run_chain.sh + helpers all live in /workspace/_sync/.
#     sbatch ${CL_HOME:-/workspace/home/nayan}/_sync/cartridges_temporalwiki/run_chain.sbatch
# Override (resume into existing run dir; export VARS in shell first, do NOT
# pass --export=ALL,VAR=val — that pattern triggers slurm launch failures
# on this cluster, see SDPO sbatch comment):
#     export CARTRIDGES_TWIKI_RUN_TAG=foo
#     sbatch ${CL_HOME:-/workspace/home/nayan}/_sync/cartridges_temporalwiki/run_chain.sbatch

set -euo pipefail

[[ -f ${CL_HOME:-/workspace/home/nayan}/sdpo_seq/env.sh ]] && source ${CL_HOME:-/workspace/home/nayan}/sdpo_seq/env.sh
unset ROCR_VISIBLE_DEVICES
ray stop --force 2>/dev/null || true
export WANDB_API_KEY="${WANDB_API_KEY:?set WANDB_API_KEY in your shell (https://wandb.ai/authorize)}"
# Wandb init occasionally takes >90s on first call; default-bump so train phases
# don't fail at the wandb.init boundary.
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"

if [[ -z "${CARTRIDGES_TWIKI_RUN_TAG:-}" ]]; then
    SHA=$(git -C ${CL_HOME:-/workspace/home/nayan}/methods/cartridges rev-parse --short=4 HEAD 2>/dev/null || echo "nogit")
    TS=$(date -u +%Y%m%d-%H%M%S)
    export CARTRIDGES_TWIKI_RUN_TAG="twiki-cl-${TS}-${SHA}_cartridges_orderT_nothink_s42"
fi
echo "[sbatch] CARTRIDGES_TWIKI_RUN_TAG=${CARTRIDGES_TWIKI_RUN_TAG}"

bash ${CL_HOME:-/workspace/home/nayan}/_sync/cartridges_temporalwiki/run_chain.sh
