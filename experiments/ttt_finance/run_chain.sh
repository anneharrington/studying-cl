#!/bin/bash
# In-Place TTT finance ordering-F sweep — 6 sequential continual-pretrain
# phases, each producing a Qwen3-8B HF checkpoint, then a 6×6 eval matrix.
#
# Output layout (matches our other 5 methods):
#
#   $RUN_ROOT/<RUN_TAG>/
#       data/y{YYYY}/data.jsonl              VeOmni plaintext (built from cumulative corpora)
#       train/y{YYYY}/dcp/                   raw DCP checkpoints
#       train/y{YYYY}/hf/                    converted HF checkpoints (Qwen3-8B with TTT-updated weights)
#       logs/{train,convert,eval}_p{i}_y{YYYY}.log
#       manifest.json
#       DONE
#
#   $PRED_ROOT/ttt/phase{i}/y{YYYY}.parquet  (Anastasia format; consumed by run_finance_analysis.sh)
#
# Resume-safe: re-running skips phases whose hf ckpt exists.
#
# Required env: WANDB_API_KEY (set by sbatch wrapper)
# Optional:     YEARS, CARTRIDGES_FINANCE_RUN_TAG, SKIP_TRAIN, SKIP_EVAL, MAX_STEPS_PER_PHASE
set -euo pipefail

# ─── Redirect every Python/torch write off the overlay/container FS ──────────
# CL_HOME is the documented scratch root (see docs/SETUP.md); WORKSPACE_HOME is
# the legacy alias kept for back-compat. Falls back to $HOME so a stranger
# without any of these set still gets writable cache dirs.
: "${WORKSPACE_HOME:=${CL_HOME:-$HOME}}"
export TMPDIR="${WORKSPACE_HOME}/.tmp"
export PYTHONPYCACHEPREFIX="${WORKSPACE_HOME}/.pycache"
export TORCHINDUCTOR_CACHE_DIR="${WORKSPACE_HOME}/.cache/torchinductor"
export TRITON_CACHE_DIR="${WORKSPACE_HOME}/.cache/triton"
export HF_HOME="${WORKSPACE_HOME}/.cache/huggingface"
mkdir -p "$TMPDIR" "$PYTHONPYCACHEPREFIX" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$HF_HOME"

# Defensive cleanup: kill any orphan veomni/torch python procs from prior runs
if pgrep -u "$USER" -f "venv-ttt.*(multiprocessing-fork|resource_tracker|torchrun)" >/dev/null 2>&1; then
    echo "[chain] $(date -u)  killing orphan TTT procs from prior runs"
    pkill -9 -u "$USER" -f "venv-ttt.*multiprocessing-fork" 2>/dev/null || true
    pkill -9 -u "$USER" -f "venv-ttt.*resource_tracker" 2>/dev/null || true
    pkill -9 -u "$USER" -f "venv-ttt.*torchrun" 2>/dev/null || true
    sleep 3
fi

# ─── Paths + venvs ────────────────────────────────────────────────────────────
SCRIPTS_EARLY=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPTS_EARLY/../.." && pwd)
# Prefer the documented .venv-ttt convention (SETUP.md §4); legacy CL_HOME/venv-ttt
# is honored when explicitly set, so existing cluster layouts keep working.
: "${TTT_VENV:=${CL_HOME:+$CL_HOME/venv-ttt}}"
: "${TTT_VENV:=$REPO_ROOT/.venv-ttt}"
: "${IPT_REPO:=$REPO_ROOT/methods/compression/in_place_ttt}"
: "${BASE_MODEL:=Qwen/Qwen3-8B}"
: "${BASE_MODEL_DIR:=$HF_HOME/hub/models--Qwen--Qwen3-8B}"   # tells convert script where the assets live
# Default ~1-8 epochs across our 6-37M-token cumulative corpora at gbs=64
# × seq_len=16384 ≈ 1M tokens/step. Override via env if you want more
# pretrain (paper used 5000 steps, but for 20B-token corpora — wildly over
# for our scale). Smoke sbatch sets 50 explicitly anyway.
: "${MAX_STEPS_PER_PHASE:=50}"

[ -x "$TTT_VENV/bin/python" ] || { echo "!! $TTT_VENV missing — run install.sh first"; exit 2; }

# Activate venv-ttt so train.sh's torchrun finds wandb / veomni / etc on PATH.
# train.sh invokes `torchrun` directly without any venv activation; without
# this prepend, torchrun resolves to /usr/local/bin and wandb / veomni are
# absent → "ModuleNotFoundError: No module named 'wandb'" on every rank.
export PATH="$TTT_VENV/bin:$PATH"
export VIRTUAL_ENV="$TTT_VENV"
# IPT's tasks/train_torch.py does `import hf_models` (a sibling dir, not an
# installed package — the editable install was rejected by the build system).
# Mirror what IPT's eval.sh does: prepend the repo root to PYTHONPATH so
# that import resolves.
export PYTHONPATH="$IPT_REPO:${PYTHONPATH:-}"
[ -d "$IPT_REPO" ]            || { echo "!! $IPT_REPO missing — run install.sh first"; exit 2; }

PYTHON="$TTT_VENV/bin/python"
SCRIPTS=$(cd "$(dirname "$0")" && pwd)
RUN_TAG=${CARTRIDGES_FINANCE_RUN_TAG:-finance-cl-$(date -u +%Y%m%d-%H%M%S)-nogit_ttt_orderF_nothink_s42}
RUN_ROOT_DEFAULT=${CL_HOME:-$REPO_ROOT/runs}/ttt_finance/runs
RUN_DIR=${TTT_FINANCE_RUN_DIR:-${RUN_ROOT_DEFAULT}/${RUN_TAG}}
PRED_ROOT=${TTT_FINANCE_PRED_ROOT:-${CL_HOME:-$REPO_ROOT/runs}/results/finance_predictions/ttt}
# Per-year parquets dir: canonical CL_FINANCE_DATA matches bootstrap.sh +
# run_sequential.py; TTT_FINANCE_DATA is a method-specific override.
DATA_DIR=${TTT_FINANCE_DATA:-${CL_FINANCE_DATA:-$REPO_ROOT/data/finance_yearly}}
RETURNS_TABLE=${TTT_FINANCE_RETURNS:-${DATA_DIR%/cl_yearly}/finance_returns_table.parquet}

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/data" "$RUN_DIR/train"
echo "[chain] $(date -u)  RUN_TAG=$RUN_TAG  run_dir=$RUN_DIR"
export TTT_FINANCE_RUN_TAG="$RUN_TAG"

# ─── Years (orderF default) ───────────────────────────────────────────────────
read -r -a YEARS_ARR <<< "${YEARS:-2015 2016 2017 2018 2019 2020}"
YEARS_CSV=$(IFS=,; echo "${YEARS_ARR[*]}")

# Manifest
"$PYTHON" - <<PY
import json, os, datetime as dt, subprocess
m = {
    "run_tag": "${RUN_TAG}",
    "run_dir": "${RUN_DIR}",
    "method": "ttt",
    "ordering": "F",
    "model": "${BASE_MODEL}",
    "seed": 42,
    "years": [${YEARS_CSV}],
    "started_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds") + "Z",
    "max_steps_per_phase": int(os.environ.get("MAX_STEPS_PER_PHASE", "500")),
    "wandb_project": os.environ.get("CARTRIDGES_WANDB_PROJECT", "sdpo_seq"),
}
try:
    m["ipt_commit"] = subprocess.check_output(
        ["git", "-C", "${IPT_REPO}", "rev-parse", "HEAD"], text=True).strip()
except Exception: pass
with open("${RUN_DIR}/manifest.json", "w") as f:
    json.dump(m, f, indent=2)
PY

# ─── 1. Data adapter (idempotent — produces JSONL from cartridges_finance corpora) ─
# Reuse the cartridges_finance corpora since they're already extracted from the
# train parquets. CARTRIDGES_FINANCE_CORPORA is required; produce it first with
# experiments/cartridges_finance/run_chain.sh and pass the run's corpora/ path.
: "${CARTRIDGES_FINANCE_CORPORA:?set CARTRIDGES_FINANCE_CORPORA to a prior cartridges_finance run corpora/ dir; run experiments/cartridges_finance/run_chain.sh first}"
if [[ ! -f "$RUN_DIR/data/cumulative_y6/data.jsonl" ]]; then
    echo "[chain] $(date -u)  building VeOmni plaintext JSONL from $CARTRIDGES_FINANCE_CORPORA"
    [ -d "$CARTRIDGES_FINANCE_CORPORA" ] || { echo "  !! corpora dir missing: $CARTRIDGES_FINANCE_CORPORA"; exit 3; }
    "$PYTHON" "$SCRIPTS/data_adapter.py" \
        --src-dir "$CARTRIDGES_FINANCE_CORPORA" \
        --out-dir "$RUN_DIR/data" \
        --years "${YEARS_ARR[@]}" \
        --cumulative
fi

# ─── 2. Per-phase TTT continual pretrain ──────────────────────────────────────
# Phase i: source = phase (i-1)'s HF ckpt (or BASE_MODEL for phase 1).
#          data  = cumulative_y{i}/data.jsonl
#          out   = train/y{y_i}/{dcp,hf}/

CONVERT_SCRIPT="$IPT_REPO/scripts/merge_dcp_to_hf.py"
TRAIN_SCRIPT="$IPT_REPO/tasks/train_torch.py"
BASE_CONFIG="$SCRIPTS/configs/qwen3_longct_finance.yaml"

declare -A HF_DIR
prev_hf="$BASE_MODEL"   # HF identifier or local dir; first phase starts from base
i=0
for y in "${YEARS_ARR[@]}"; do
    i=$((i+1))

    out_dcp="$RUN_DIR/train/y${y}/dcp"
    out_hf="$RUN_DIR/train/y${y}/hf"
    log_train="$RUN_DIR/logs/train_p${i}_y${y}.log"
    log_convert="$RUN_DIR/logs/convert_p${i}_y${y}.log"

    if [[ -f "$out_hf/config.json" ]] || [[ "${SKIP_TRAIN:-0}" == "1" ]]; then
        echo "[chain] phase $i y${y} train: SKIP ($out_hf)"
        HF_DIR[$y]="$out_hf"
        prev_hf="$out_hf"
        continue
    fi

    # If a partial DCP/HF dir exists from a crashed prior run, wipe so we start clean
    if [[ -d "$out_dcp" || -d "$out_hf" ]]; then
        echo "[chain] phase $i y${y} train: wiping partial $RUN_DIR/train/y${y}/"
        rm -rf "$RUN_DIR/train/y${y}"
    fi
    mkdir -p "$out_dcp" "$out_hf"

    # IMPORTANT: pass the JSONL file explicitly, not the dir. VeOmni's
    # build_iterable_dataset does os.listdir(data_path) and ingests EVERY
    # file, including our data.manifest.json — which has no `content_split`
    # key and trips a KeyError in the data transform. Single-file path
    # avoids this.
    train_data="$RUN_DIR/data/cumulative_y${i}/data.jsonl"
    [ -f "$train_data" ] || { echo "  !! missing train data $train_data"; exit 3; }

    echo "[chain] $(date -u)  phase $i y${y} TTT pretrain (cumulative_y${i}, source=$prev_hf) → $log_train"

    # Train via VeOmni — pass yaml + per-phase overrides via env-var resolution in the yaml
    # plus explicit --field=value overrides for things the yaml uses ${ENV_VAR} for.
    MODEL_PATH="$prev_hf" \
    TRAIN_PATH="$train_data" \
    OUTPUT_DIR="$out_dcp" \
    WANDB_PROJECT="${CARTRIDGES_WANDB_PROJECT:-sdpo_seq}" \
    WANDB_NAME="${RUN_TAG}_pretrain_y${y}" \
    bash "$IPT_REPO/train.sh" "$TRAIN_SCRIPT" "$BASE_CONFIG" \
        --train.output_dir "$out_dcp" \
        --train.max_steps "$MAX_STEPS_PER_PHASE" \
        --train.wandb_project "${CARTRIDGES_WANDB_PROJECT:-sdpo_seq}" \
        --train.wandb_name "${RUN_TAG}_pretrain_y${y}" \
        --model.model_path "$prev_hf" \
        --data.train_path "$train_data" \
        > "$log_train" 2>&1 \
        || { echo "  !! TTT pretrain failed; see $log_train"; tail -10 "$log_train"; exit 4; }

    # Materialize the HF checkpoint at $out_hf. Modern veomni already writes a
    # complete HF model (config + tokenizer + safetensors shards) under
    # dcp/checkpoints/global_step_N/hf_ckpt/ as a side effect of training, so
    # we just promote that to $out_hf instead of re-merging via merge_dcp_to_hf.py
    # (which was looking at $out_dcp/.metadata — wrong path for this veomni layout
    # and the source of the historical "FileNotFoundError: .metadata" failures).
    # If a future veomni version stops writing hf_ckpt, fall back to the merger.
    veomni_hf=$(ls -d "$out_dcp/checkpoints"/global_step_*/hf_ckpt 2>/dev/null | sort -V | tail -1)
    if [[ -n "$veomni_hf" && -f "$veomni_hf/config.json" ]]; then
        echo "[chain] $(date -u)  phase $i y${y} promoting veomni hf_ckpt -> $out_hf"
        rmdir "$out_hf" 2>/dev/null || rm -rf "$out_hf"
        ln -s "$veomni_hf" "$out_hf"
    else
        echo "[chain] $(date -u)  phase $i y${y} DCP→HF convert (no veomni hf_ckpt found) → $log_convert"
        "$PYTHON" "$CONVERT_SCRIPT" \
            --load-dir "$out_dcp" \
            --save-dir "$out_hf" \
            --model-assets-dir "$BASE_MODEL_DIR" \
            --shard-size 5000000000 \
            > "$log_convert" 2>&1 \
            || { echo "  !! ckpt convert failed; see $log_convert"; tail -10 "$log_convert"; exit 5; }
    fi

    [[ -f "$out_hf/config.json" ]] || { echo "  !! HF ckpt promotion/convert returned 0 but no config.json under $out_hf"; exit 5; }
    HF_DIR[$y]="$out_hf"
    prev_hf="$out_hf"
done

# ─── 3. Eval matrix (one HF checkpoint per phase × 6 eval years) ──────────────
if [[ "${SKIP_EVAL:-0}" == "1" ]]; then
    echo "[chain] SKIP_EVAL=1 — done after train"
    touch "$RUN_DIR/DONE"
    exit 0
fi

EVAL_PARALLEL=${EVAL_PARALLEL:-8}
N_GPUS=${TTT_FINANCE_N_GPUS:-8}

declare -a EVAL_JOBS
i=0
for y_i in "${YEARS_ARR[@]}"; do
    i=$((i+1))
    if [[ -z "${HF_DIR[$y_i]:-}" ]]; then
        echo "  !! no HF checkpoint for y${y_i}; can't eval phase $i"
        continue
    fi
    hf="${HF_DIR[$y_i]}"
    for y_j in "${YEARS_ARR[@]}"; do
        out=$PRED_ROOT/phase${i}/y${y_j}.parquet
        if [[ -f "$out" ]]; then
            echo "[chain] phase $i eval y${y_j}: SKIP (parquet exists)"
            continue
        fi
        EVAL_JOBS+=("${i}|${y_i}|${y_j}|${hf}")
    done
done

n_total=${#EVAL_JOBS[@]}
echo "[chain] $(date -u)  eval: $n_total cells to run, $EVAL_PARALLEL in parallel (one GPU each)"

batch_idx=0
while (( ${#EVAL_JOBS[@]} > 0 )); do
    pids=()
    n_to_run=$(( ${#EVAL_JOBS[@]} < EVAL_PARALLEL ? ${#EVAL_JOBS[@]} : EVAL_PARALLEL ))
    declare -a fails=()
    for ((k=0; k<n_to_run; k++)); do
        job=${EVAL_JOBS[k]}
        IFS="|" read -r ii y_i y_j hf <<< "$job"
        gpu=$(( k % N_GPUS ))
        out=$PRED_ROOT/phase${ii}/y${y_j}.parquet
        log=$RUN_DIR/logs/eval_p${ii}_y${y_j}.log
        val_pq=$DATA_DIR/val_y${y_j}.parquet
        mkdir -p "$(dirname "$out")"
        echo "[chain]   batch $batch_idx GPU $gpu: phase $ii eval y${y_j} (HF=$hf) → $log"
        CUDA_VISIBLE_DEVICES=$gpu \
            "$PYTHON" "$SCRIPTS/eval_ttt.py" \
                --hf-ckpt "$hf" \
                --eval-year "$y_j" --train-year "$y_i" --phase-idx "$ii" \
                --val-parquet "$val_pq" --returns-table "$RETURNS_TABLE" \
                --out "$out" --device cuda:0 \
                > "$log" 2>&1 &
        pids+=($!)
    done
    EVAL_JOBS=("${EVAL_JOBS[@]:n_to_run}")
    for pid in "${pids[@]}"; do wait $pid || fails+=("$pid"); done
    if (( ${#fails[@]} > 0 )); then
        echo "  !! ${#fails[@]} eval(s) failed in batch $batch_idx; see $RUN_DIR/logs/eval_*.log"
    fi
    batch_idx=$((batch_idx+1))
done

touch "$RUN_DIR/DONE"
echo
echo "[chain] $(date -u)  done"
echo "  run dir:     $RUN_DIR"
echo "  logs:        $RUN_DIR/logs/"
echo "  predictions: $PRED_ROOT/phase{1..6}/y{2015..2020}.parquet"
echo
echo "next:  bash /home/nayan/scripts/run_finance_analysis.sh   # adds 'ttt' to existing analysis"
