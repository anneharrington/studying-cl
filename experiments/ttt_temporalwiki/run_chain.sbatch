#!/bin/bash
#SBATCH --job-name=ttt_twiki
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=${CL_HOME:-/workspace/home/nayan}/sdpo_seq/logs/slurm_%x_%j.out
#SBATCH --error=${CL_HOME:-/workspace/home/nayan}/sdpo_seq/logs/slurm_%x_%j.err
set -euo pipefail
[[ -f ${CL_HOME:-/workspace/home/nayan}/sdpo_seq/env.sh ]] && source ${CL_HOME:-/workspace/home/nayan}/sdpo_seq/env.sh
unset ROCR_VISIBLE_DEVICES
ray stop --force 2>/dev/null || true
export WANDB_API_KEY="${WANDB_API_KEY:?set WANDB_API_KEY in your shell (https://wandb.ai/authorize)}"
SHA=$(git -C ${CL_HOME:-/workspace/home/nayan}/methods/In-Place-TTT rev-parse --short=4 HEAD 2>/dev/null || echo "nogit")
TS=$(date -u +%Y%m%d-%H%M%S)
case "ttt_temporalwiki" in
    ttt_sweep)         export TTT_SWEEP_RUN_TAG="${TTT_SWEEP_RUN_TAG:-twiki-cl-${TS}-${SHA}_ttt_orderT_nothink_s42}" ;;
    ttt_temporalwiki)  export TTT_TWIKI_RUN_TAG="${TTT_TWIKI_RUN_TAG:-twiki-cl-${TS}-${SHA}_ttt_orderT_nothink_s42}" ;;
esac
bash /home/nayan/scripts/ttt_temporalwiki/run_chain.sh
