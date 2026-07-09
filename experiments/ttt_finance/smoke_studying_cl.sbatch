#!/bin/bash
#SBATCH --job-name=ttt_smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --output=runs/logs/slurm_%x_%j.out
#SBATCH --error=runs/logs/slurm_%x_%j.err
#
# Minimal end-to-end In-Place TTT smoke on the finance ordering-F chain:
#   - 1 year (y2015)
#   - MAX_STEPS_PER_PHASE=2 (vs 500 default)
#   - SKIP_EVAL=1 (no 6×6 matrix needed for a smoke)
#
# Exercises: data adapter (jsonl from corpora), VeOmni TTT pretrain
# (Qwen3-8B FSDP, 2 steps), HF checkpoint write + HF promotion.
# Total ~25-35 min on a fresh node.
#
# Required env in the calling shell:
#   CARTRIDGES_FINANCE_CORPORA   Path to corpora built by a prior
#                                cartridges_finance run (any RUN_DIR/corpora/).
#                                TTT reads cumulative_y${i}/corpus.txt; if the
#                                cartridges chain hasn't been run yet, run the
#                                cartridges smoke first.

set -uo pipefail

SCRIPTS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPTS/../.." && pwd)
cd "$REPO"
mkdir -p "$REPO/runs/logs"

export WANDB_MODE=offline
export CARTRIDGES_WANDB_DISABLE=1
unset ROCR_VISIBLE_DEVICES

# Hard requirement - TTT consumes corpora that a prior cartridges_finance run
# produced. Fail fast with a helpful message if not set.
: "${CARTRIDGES_FINANCE_CORPORA:?set CARTRIDGES_FINANCE_CORPORA=<some cartridges run>/corpora -- run experiments/cartridges_finance/smoke_studying_cl.sbatch first}"
[ -d "$CARTRIDGES_FINANCE_CORPORA" ] || {
    echo "!! CARTRIDGES_FINANCE_CORPORA does not exist: $CARTRIDGES_FINANCE_CORPORA"
    exit 2
}

# Smoke knobs: one year, two steps, no eval.
export YEARS="2015"
export MAX_STEPS_PER_PHASE=2
export SKIP_EVAL=1

# Per-job RUN_DIR for clean state. Defaults to repo-relative so a stranger
# without CL_HOME still gets a writable path.
export TTT_FINANCE_RUN_DIR=${TTT_FINANCE_RUN_DIR:-$REPO/runs/ttt_smoke/run_${SLURM_JOB_ID:-$$}}
mkdir -p "$TTT_FINANCE_RUN_DIR"

echo "[smoke] $(date -u)  starting TTT finance smoke (year=2015, max_steps=2)"
bash experiments/ttt_finance/run_chain.sh
rc=$?
echo "[smoke] $(date -u)  done (rc=$rc)"
exit $rc
