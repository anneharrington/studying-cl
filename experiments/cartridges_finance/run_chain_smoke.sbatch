#!/bin/bash
#SBATCH --job-name=cartridges_finance_smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=${CL_HOME:-/workspace/home/nayan}/sdpo_seq/logs/slurm_%x_%j.out
#SBATCH --error=${CL_HOME:-/workspace/home/nayan}/sdpo_seq/logs/slurm_%x_%j.err
#
# Cartridges finance ordering-F — SMOKE TEST via slurm.
# Single phase (y2015 only), cheap knobs, ~30 min wall time. Verifies the
# corpora -> tokasaurus -> synth -> train -> eval chain end-to-end before
# committing to the full 6-phase production sweep.
#
# Submit:
#     sbatch /home/nayan/scripts/cartridges_finance/run_chain_smoke.sbatch
#
# Logs:
#     ${CL_HOME:-/workspace/home/nayan}/sdpo_seq/logs/slurm_cartridges_finance_smoke_<jobid>.{out,err}
#     ${CL_HOME:-/workspace/home/nayan}/cartridges_finance/runs/finance-cl-cartridges_orderF_nothink_s42/logs/

set -euo pipefail

[[ -f ${CL_HOME:-/workspace/home/nayan}/sdpo_seq/env.sh ]] && source ${CL_HOME:-/workspace/home/nayan}/sdpo_seq/env.sh

unset ROCR_VISIBLE_DEVICES
ray stop --force 2>/dev/null || true

export WANDB_API_KEY="${WANDB_API_KEY:?set WANDB_API_KEY in your shell (https://wandb.ai/authorize)}"

# Smoke knobs: single phase, cheap synth/train, small chunks/p so it finishes fast.
export YEARS="${YEARS:-2015}"
export CARTRIDGES_FINANCE_N_SYNTH="${CARTRIDGES_FINANCE_N_SYNTH:-512}"
export CARTRIDGES_FINANCE_P="${CARTRIDGES_FINANCE_P:-2048}"
export CARTRIDGES_FINANCE_CHUNK_MIN="${CARTRIDGES_FINANCE_CHUNK_MIN:-512}"
export CARTRIDGES_FINANCE_CHUNK_MAX="${CARTRIDGES_FINANCE_CHUNK_MAX:-1024}"
export CARTRIDGES_FINANCE_PACKED_SEQ_LEN="${CARTRIDGES_FINANCE_PACKED_SEQ_LEN:-4096}"
# Smaller tokasaurus KV pool: smoke chunks are <=1024 tokens × 128 parallel = 131k
# aggregate; 64k per-rank × 8 ranks = 512k aggregate is comfortable.
export CARTRIDGES_TOKASAURUS_KV_TOKENS="${CARTRIDGES_TOKASAURUS_KV_TOKENS:-(64 * 1024)}"

bash /home/nayan/scripts/cartridges_finance/run_chain.sh
