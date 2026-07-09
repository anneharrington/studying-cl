#!/bin/bash
#SBATCH --job-name=sdpo_smoke_finance
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=${CL_HOME:-/workspace/home/nayan}/sdpo_seq/logs/slurm_smoke_finance_%j.out
#SBATCH --error=${CL_HOME:-/workspace/home/nayan}/sdpo_seq/logs/slurm_smoke_finance_%j.err
#
# Finance ordering F smoke: 1 task (y2015), no chain_final loop. Goes through the same
# run_sequential.py path as the production sweep so any wiring bug for ordering F
# surfaces here, not in the first sweep job.
#
# val_before_train=true gives a step-0 baseline eval on all 6 year sources before any
# gradient step; the post-run verifier checks the JSONL contains every
# val-core/finance_yr_{2015..2020}/acc/mean@N key.
#
# Required env vars: SDPO_ROOT, WANDB_API_KEY (or WANDB_MODE=offline).
# Optional:          SDPO_RUN_ROOT, VENV_ACTIVATE, SDPO_ENV.

set -euo pipefail

METHOD=${METHOD:-sft}
SEED=${SEED:-42}
SWEEP_TAG=${SWEEP_TAG:-smoke-f}
ENABLE_THINKING=${ENABLE_THINKING:-false}

SDPO_ROOT=${CL_HOME:-/workspace/home/nayan}/SDPO
source ${CL_HOME:-/workspace/home/nayan}/sdpo_seq/env.sh
source ${CL_HOME:-/workspace/home/nayan}/venv-sdpo-v2/bin/activate

export WANDB_API_KEY="${WANDB_API_KEY:?set WANDB_API_KEY in your shell (https://wandb.ai/authorize)}"

# Slurm sets ROCR_VISIBLE_DEVICES alongside CUDA_VISIBLE_DEVICES; verl rejects the pair.
unset ROCR_VISIBLE_DEVICES

# Per-job Ray temp dir: see run_sequential.sbatch.
ray stop --force 2>/dev/null || true
export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}"

cd "$SDPO_ROOT"

echo "[smoke-f] $(date -u)  method=$METHOD ordering=F seed=$SEED sweep_tag=$SWEEP_TAG"

python experiments/continual/run_sequential.py \
    --method "$METHOD" \
    --ordering F \
    --seed "$SEED" \
    --enable-thinking "$ENABLE_THINKING" \
    --sweep-tag "$SWEEP_TAG" \
    --num-tasks 1 \
    --skip-chain-final \
    --total-epochs 1 \
    --test-freq 0 \
    --model Qwen/Qwen3-8B \
    --n-gpus 8 \
    --nnodes 1

echo "[smoke-f] $(date -u)  run_sequential exited cleanly"

# ─── Post-run verification ─────────────────────────────────────────────────────
# Mirror smoke.sbatch's contract checks, adapted for ordering F.
case "${ENABLE_THINKING,,}" in true|1|yes) THINK_TAG=think ;; *) THINK_TAG=nothink ;; esac
RUN_TAG="${SWEEP_TAG}_${METHOD}_orderF_${THINK_TAG}_s${SEED}"
RUN_DIR=${CL_HOME:-/workspace/home/nayan}/sdpo_seq/runs/$RUN_TAG
echo "[smoke-f] === verifying $RUN_DIR ==="

python - <<PY
import json, pathlib, sys
run_dir = pathlib.Path("$RUN_DIR")
ok = True

# 1. final_hf checkpoint written (needed for task handoff in production sweep).
final_hf = run_dir / "chain_best" / "task1_y2015_ckpts" / "final_hf"
has_weights = any(final_hf.glob("*.safetensors")) or any(final_hf.glob("pytorch_model*.bin"))
has_config = (final_hf / "config.json").exists()
if has_weights and has_config:
    print(f"OK final_hf: {final_hf}")
else:
    print(f"FAIL final_hf: weights={has_weights} config={has_config} at {final_hf}")
    ok = False

# 2. FileLogger JSONL has all 6 finance_yr_<YYYY> sources at step 0 (val_before_train).
jsonl = run_dir / "chain_best" / "metrics" / "task1_y2015.jsonl"
if not jsonl.exists() or jsonl.stat().st_size == 0:
    print(f"FAIL: {jsonl} missing or empty")
    ok = False
else:
    sources_at_step0 = set()
    saw_step0 = False
    for raw in jsonl.read_bytes().splitlines():
        if not raw.strip():
            continue
        evt = json.loads(raw)
        if int(evt.get("step", -1)) != 0:
            continue
        saw_step0 = True
        for k in evt.get("data", {}):
            if k.startswith("val-core/"):
                sources_at_step0.add(k.split("/")[1])
    expected = {f"finance_yr_{y}" for y in (2015, 2016, 2017, 2018, 2019, 2020)}
    missing = expected - sources_at_step0
    if not saw_step0:
        print(f"FAIL: no step-0 entry in {jsonl}")
        ok = False
    elif missing:
        print(f"FAIL: step-0 missing sources: {missing} (saw {sources_at_step0})")
        ok = False
    else:
        print(f"OK step-0 baseline: all 6 finance_yr_<YYYY> sources present")
        # Pretty-print the baseline accuracies.
        for raw in jsonl.read_bytes().splitlines():
            if not raw.strip():
                continue
            evt = json.loads(raw)
            if int(evt.get("step", -1)) != 0:
                continue
            for k in sorted(evt.get("data", {})):
                if k.startswith("val-core/finance_yr_") and k.endswith("/acc/mean@8"):
                    print(f"   {k}: {evt['data'][k]:.3f}")
            break

sys.exit(0 if ok else 1)
PY
echo "[smoke-f] $(date -u)  done"
