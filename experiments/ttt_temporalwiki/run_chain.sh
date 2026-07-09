#!/bin/bash
# In-Place TTT TemporalWiki ordering-T sweep — 3 sequential continual-pretrain
# phases (ts1 → ts2 → ts3, each on its slice's training facts), then a 3×4
# eval matrix (3 phases × {drift_s1, drift_s2, drift_s3, stable}).
#
# Output:
#   $RUN_ROOT/<RUN_TAG>/{data,train/ts{1..3}/{dcp,hf},logs,manifest.json,DONE}
#   $PRED_ROOT/ttt/ts{1..3}/{drift_s1,drift_s2,drift_s3,stable}.parquet
set -euo pipefail

# ─── Paths + caches ──────────────────────────────────────────────────────────
SCRIPTS=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPTS/../.." && pwd)

# Redirect torch/HF caches off any read-only overlay FS. CL_HOME is the
# canonical scratch root (see docs/SETUP.md); WORKSPACE_HOME is a legacy alias
# kept for back-compat. Falls back to $HOME for fresh installs.
: "${WORKSPACE_HOME:=${CL_HOME:-$HOME}}"
export TMPDIR="${WORKSPACE_HOME}/.tmp"
export PYTHONPYCACHEPREFIX="${WORKSPACE_HOME}/.pycache"
export TORCHINDUCTOR_CACHE_DIR="${WORKSPACE_HOME}/.cache/torchinductor"
export TRITON_CACHE_DIR="${WORKSPACE_HOME}/.cache/triton"
export HF_HOME="${WORKSPACE_HOME}/.cache/huggingface"
mkdir -p "$TMPDIR" "$PYTHONPYCACHEPREFIX" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$HF_HOME"

if pgrep -u "$USER" -f "venv-ttt.*(multiprocessing-fork|resource_tracker|torchrun)" >/dev/null 2>&1; then
    pkill -9 -u "$USER" -f "venv-ttt.*multiprocessing-fork" 2>/dev/null || true
    pkill -9 -u "$USER" -f "venv-ttt.*resource_tracker" 2>/dev/null || true
    pkill -9 -u "$USER" -f "venv-ttt.*torchrun" 2>/dev/null || true
    sleep 3
fi

# Prefer the documented .venv-ttt convention (SETUP.md §4); CL_HOME/venv-ttt
# stays honored when explicitly set for existing cluster layouts.
: "${TTT_VENV:=${CL_HOME:+$CL_HOME/venv-ttt}}"
: "${TTT_VENV:=$REPO_ROOT/.venv-ttt}"
# IPT_REPO defaults to the vendored copy at methods/compression/in_place_ttt;
# the older ${CL_HOME}/methods/In-Place-TTT lookup was a stale path that no
# longer exists in this repo layout.
: "${IPT_REPO:=$REPO_ROOT/methods/compression/in_place_ttt}"
: "${BASE_MODEL:=Qwen/Qwen3-8B}"
: "${BASE_MODEL_DIR:=$HF_HOME/hub/models--Qwen--Qwen3-8B}"
: "${MAX_STEPS_PER_PHASE:=300}"

[ -x "$TTT_VENV/bin/python" ] || { echo "!! $TTT_VENV missing — see docs/SETUP.md §4 (ttt venv)"; exit 2; }
PYTHON="$TTT_VENV/bin/python"
RUN_TAG=${TTT_TWIKI_RUN_TAG:-twiki-cl-$(date -u +%Y%m%d-%H%M%S)-nogit_ttt_orderT_nothink_s42}
RUN_DIR=${TTT_TWIKI_RUN_DIR:-${CL_HOME:-$REPO_ROOT/runs}/ttt_temporalwiki/runs/${RUN_TAG}}
PRED_ROOT=${TTT_TWIKI_PRED_ROOT:-${CL_HOME:-$REPO_ROOT/runs}/results/temporalwiki_predictions/ttt}
# Eval val parquets come from a prior SFT-on-TemporalWiki run (see
# experiments/continual/run_sequential.py with --method sft --ordering T).
# Override TTT_TWIKI_SDPO_RUN to point at that run's output dir if you've
# already produced one; otherwise eval will skip with a clear warning.
SDPO_TWIKI_RUN=${TTT_TWIKI_SDPO_RUN:-${CL_HOME:-$REPO_ROOT/runs}/sdpo_seq/runs/twiki-cl_sft_orderT_nothink_s42}

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/data" "$RUN_DIR/train"
echo "[chain] $(date -u)  RUN_TAG=$RUN_TAG  run_dir=$RUN_DIR"

read -r -a SLICES_ARR <<< "${SLICES:-ts1 ts2 ts3}"

# Manifest
"$PYTHON" - <<PY
import json, datetime as dt, os
m = {"run_tag": "${RUN_TAG}", "method": "ttt", "ordering": "T",
     "model": "${BASE_MODEL}", "seed": 42,
     "slices": "${SLICES_ARR[*]}".split(),
     "started_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds") + "Z",
     "max_steps_per_phase": int(os.environ.get("MAX_STEPS_PER_PHASE", "300"))}
open("${RUN_DIR}/manifest.json", "w").write(json.dumps(m, indent=2))
PY

# 1. Data adapter
if [[ ! -f "$RUN_DIR/data/ts3/data.jsonl" ]]; then
    echo "[chain] $(date -u)  building VeOmni plaintext JSONL for slices: ${SLICES_ARR[*]}"
    "$PYTHON" "$SCRIPTS/data_adapter.py" --out-dir "$RUN_DIR/data" --slices "${SLICES_ARR[@]}"
fi

# 2. Per-slice TTT pretrain (sequential)
CONVERT_SCRIPT="$IPT_REPO/scripts/merge_dcp_to_hf.py"
TRAIN_SCRIPT="$IPT_REPO/tasks/train_torch.py"
BASE_CONFIG="$SCRIPTS/configs/qwen3_longct_twiki.yaml"

declare -A HF_DIR
prev_hf="$BASE_MODEL"
i=0
for ts in "${SLICES_ARR[@]}"; do
    i=$((i+1))
    out_dcp="$RUN_DIR/train/${ts}/dcp"
    out_hf="$RUN_DIR/train/${ts}/hf"
    log_train="$RUN_DIR/logs/train_p${i}_${ts}.log"
    log_convert="$RUN_DIR/logs/convert_p${i}_${ts}.log"

    if [[ -f "$out_hf/config.json" ]] || [[ "${SKIP_TRAIN:-0}" == "1" ]]; then
        echo "[chain] phase $i $ts train: SKIP ($out_hf)"
        HF_DIR[$ts]="$out_hf"; prev_hf="$out_hf"; continue
    fi
    if [[ -d "$out_dcp" || -d "$out_hf" ]]; then rm -rf "$RUN_DIR/train/${ts}"; fi
    mkdir -p "$out_dcp" "$out_hf"

    train_data="$RUN_DIR/data/${ts}"
    [ -d "$train_data" ] || { echo "  !! missing $train_data"; exit 3; }

    echo "[chain] $(date -u)  phase $i $ts TTT pretrain (source=$prev_hf) → $log_train"
    MODEL_PATH="$prev_hf" TRAIN_PATH="$train_data" OUTPUT_DIR="$out_dcp" \
    WANDB_PROJECT="${CARTRIDGES_WANDB_PROJECT:-sdpo_seq}" \
    WANDB_NAME="${RUN_TAG}_pretrain_${ts}" \
    bash "$IPT_REPO/train.sh" "$TRAIN_SCRIPT" "$BASE_CONFIG" \
        --train.output_dir "$out_dcp" \
        --train.max_steps "$MAX_STEPS_PER_PHASE" \
        --train.wandb_project "${CARTRIDGES_WANDB_PROJECT:-sdpo_seq}" \
        --train.wandb_name "${RUN_TAG}_pretrain_${ts}" \
        --model.model_path "$prev_hf" \
        --data.train_path "$train_data" \
        > "$log_train" 2>&1 \
        || { echo "  !! TTT pretrain failed; see $log_train"; tail -10 "$log_train"; exit 4; }

    "$PYTHON" "$CONVERT_SCRIPT" --load-dir "$out_dcp" --save-dir "$out_hf" \
        --model-assets-dir "$BASE_MODEL_DIR" --shard-size 5000000000 \
        > "$log_convert" 2>&1 \
        || { echo "  !! convert failed; see $log_convert"; tail -10 "$log_convert"; exit 5; }
    HF_DIR[$ts]="$out_hf"; prev_hf="$out_hf"
done

# 3. Eval matrix (3 phases × 4 eval sets = 12 cells)
[[ "${SKIP_EVAL:-0}" == "1" ]] && { touch "$RUN_DIR/DONE"; exit 0; }

declare -A VAL_PARQUETS=(
    ["drift_s1"]=$SDPO_TWIKI_RUN/val_ts1.parquet
    ["drift_s2"]=$SDPO_TWIKI_RUN/val_ts2.parquet
    ["drift_s3"]=$SDPO_TWIKI_RUN/val_ts3.parquet
    ["stable"]=$SDPO_TWIKI_RUN/val_stable.parquet
)

EVAL_PARALLEL=${EVAL_PARALLEL:-8}
declare -a EVAL_JOBS
i=0
for ts_i in "${SLICES_ARR[@]}"; do
    i=$((i+1))
    [[ -z "${HF_DIR[$ts_i]:-}" ]] && continue
    hf="${HF_DIR[$ts_i]}"
    for eval_set in drift_s1 drift_s2 drift_s3 stable; do
        out=$PRED_ROOT/${ts_i}/${eval_set}.parquet
        if [[ -f "$out" ]]; then
            echo "[chain] phase $i eval $eval_set: SKIP"
            continue
        fi
        EVAL_JOBS+=("${i}|${ts_i}|${eval_set}|${hf}|${VAL_PARQUETS[$eval_set]}")
    done
done

echo "[chain] $(date -u)  eval: ${#EVAL_JOBS[@]} cells"
batch=0
while (( ${#EVAL_JOBS[@]} > 0 )); do
    pids=()
    n=$(( ${#EVAL_JOBS[@]} < EVAL_PARALLEL ? ${#EVAL_JOBS[@]} : EVAL_PARALLEL ))
    for ((k=0; k<n; k++)); do
        IFS="|" read -r ii ts_i eval_set hf val_pq <<< "${EVAL_JOBS[k]}"
        gpu=$((k % 8))
        out=$PRED_ROOT/${ts_i}/${eval_set}.parquet
        log=$RUN_DIR/logs/eval_p${ii}_${eval_set}.log
        mkdir -p "$(dirname "$out")"
        echo "[chain]   GPU $gpu: phase $ii eval $eval_set → $log"
        CUDA_VISIBLE_DEVICES=$gpu \
            "$PYTHON" "$SCRIPTS/eval_ttt.py" \
                --hf-ckpt "$hf" --eval-set "$eval_set" --train-slice "$ts_i" --phase-idx "$ii" \
                --val-parquet "$val_pq" --out "$out" --device cuda:0 \
                > "$log" 2>&1 &
        pids+=($!)
    done
    EVAL_JOBS=("${EVAL_JOBS[@]:n}")
    for pid in "${pids[@]}"; do wait $pid || echo "  !! eval $pid failed"; done
    batch=$((batch+1))
done

touch "$RUN_DIR/DONE"
echo "[chain] $(date -u)  done"
