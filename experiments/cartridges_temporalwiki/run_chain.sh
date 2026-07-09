#!/bin/bash
# Cartridges TemporalWiki ordering-T chain — 3 synth + 3 train + 12 eval cells.
# Path T: ts1 → ts2 → ts3 (Nov→Dec, Dec→Jan, Jan→Feb).
#
# Output layout (matches cartridges_sweep):
#   $RUN_DIR/
#       corpora/{ts1,ts2,ts3,cumulative_ts{1,2,3}}/{corpus.txt, manifest.json}
#       synth/<slice>/<TS>-synth_config/twiki_p{i}_<slice>_synth_*/.../artifact/dataset.parquet
#       train/p{i}/<TS>-train_config/<UUID>/cache_last.pt
#       logs/{synth,train,eval}_p{i}_<set>.log
#       manifest.json + DONE
#
#   $PRED_ROOT/p{i}/{drift_s1,drift_s2,drift_s3,stable}.parquet
#
# Resume-safe; same fix-list as cartridges_sweep applied (set_sharing_strategy
# in tokasaurus, slice-scoped train output, first_dir glob expansion, etc.).
#
# Optional knobs: SLICES, CARTRIDGES_TWIKI_N_SYNTH, CARTRIDGES_TWIKI_RUN_TAG,
#                 SKIP_SYNTH, SKIP_TRAIN, SKIP_EVAL.
set -euo pipefail

# Defensive cleanup of any orphan tokasaurus children
if pgrep -u "$USER" -f "venv-tokasaurus.*(multiprocessing-fork|resource_tracker)" >/dev/null 2>&1; then
    echo "[chain] $(date -u)  killing orphan tokasaurus procs from prior runs"
    pkill -9 -u "$USER" -f "venv-tokasaurus.*multiprocessing-fork" 2>/dev/null || true
    pkill -9 -u "$USER" -f "venv-tokasaurus.*resource_tracker" 2>/dev/null || true
    sleep 3
fi

# Path roots — defined here so $REPO_ROOT is available for the CARTRIDGES_DIR
# default below (set -u would otherwise reject the unset var).
SCRIPTS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPTS/../.." && pwd)

: "${CARTRIDGES_DIR:=$REPO_ROOT/methods/compression/cartridges}"
# Prefer the documented .venv-{cartridges,tokasaurus} convention (SETUP.md §3);
# CL_HOME/venv-X stays honored when explicitly set for existing cluster layouts.
: "${CARTRIDGES_VENV:=${CL_HOME:+$CL_HOME/venv-cartridges}}"
: "${CARTRIDGES_VENV:=$REPO_ROOT/.venv-cartridges}"
: "${TOKASAURUS_VENV:=${CL_HOME:+$CL_HOME/venv-tokasaurus}}"
: "${TOKASAURUS_VENV:=$REPO_ROOT/.venv-tokasaurus}"
: "${CARTRIDGES_TOKASAURUS_URL:=http://localhost:10210}"
: "${CARTRIDGES_TOKASAURUS_GPUS:=8}"
: "${CARTRIDGES_TOKASAURUS_MODEL:=Qwen/Qwen3-8B}"
: "${CARTRIDGES_TOKASAURUS_KV_TOKENS:=(256 * 1024)}"

# Auto-source venv-cartridges
if [ -z "${VIRTUAL_ENV:-}" ] || [ "${VIRTUAL_ENV}" != "${CARTRIDGES_VENV}" ]; then
    if [ -f "${CARTRIDGES_VENV}/bin/activate" ]; then
        source "${CARTRIDGES_VENV}/bin/activate"
    else
        echo "!! cartridges venv missing at ${CARTRIDGES_VENV}" >&2; exit 2
    fi
fi

PYTHON=${CARTRIDGES_VENV}/bin/python
TOKA_BIN=${TOKASAURUS_VENV}/bin/toka
TOKA_PYTHON=${TOKASAURUS_VENV}/bin/python

RUN_TAG=${CARTRIDGES_TWIKI_RUN_TAG:-twiki-cl-cartridges_orderT_nothink_s42}
RUN_ROOT_DEFAULT=${CL_HOME:-$REPO_ROOT/runs}/cartridges_temporalwiki/runs
RUN_DIR=${CARTRIDGES_TWIKI_RUN_DIR:-${RUN_ROOT_DEFAULT}/${RUN_TAG}}
PRED_ROOT=${CARTRIDGES_TWIKI_PRED_ROOT:-${CL_HOME:-$REPO_ROOT/runs}/results/temporalwiki_predictions/cartridges}

SLICES=${SLICES:-"ts1 ts2 ts3"}     # ordering T
read -r -a SLICES_ARR <<< "$SLICES"
N_PHASES=${#SLICES_ARR[@]}

# Eval surfaces — matches the SFT/SDPO/GRPO val parquets shipped in the
# canonical cl_drift_data dir. Keys are eval_set names; values are val parquet
# paths. Each phase evals against ALL of these. CL_TEMPORAL_DRIFT_DATA is the
# canonical env var (matches bootstrap.sh + run_sequential.py).
DRIFT_DATA=${CL_TEMPORAL_DRIFT_DATA:-$REPO_ROOT/data/temporalwiki_drift}
declare -A VAL_PARQUETS=(
    ["drift_s1"]="$DRIFT_DATA/val_s1.parquet"
    ["drift_s2"]="$DRIFT_DATA/val_s2.parquet"
    ["drift_s3"]="$DRIFT_DATA/val_s3.parquet"
    ["stable"]="$DRIFT_DATA/val_stable.parquet"
)
EVAL_SETS=${EVAL_SETS:-"drift_s1 drift_s2 drift_s3 stable"}
read -r -a EVAL_SETS_ARR <<< "$EVAL_SETS"

mkdir -p "$RUN_DIR"/{corpora,synth,train,logs} "$PRED_ROOT"

export CARTRIDGES_DIR
export CARTRIDGES_OUTPUT_DIR=$RUN_DIR
export PYTHONPATH="$SCRIPTS:${PYTHONPATH:-}"
: "${CARTRIDGES_WANDB_PROJECT:=sdpo_seq}"
export CARTRIDGES_WANDB_PROJECT
export CARTRIDGES_TWIKI_RUN_TAG="$RUN_TAG"

# Manifest
SLICES_CSV=$(IFS=,; echo "${SLICES_ARR[*]/#/\"}" | sed 's/,/",/g; s/$/"/')
"$PYTHON" - <<PY
import json, datetime as dt
m = {
    "run_tag": "${RUN_TAG}",
    "run_dir": "${RUN_DIR}",
    "method": "cartridges",
    "ordering": "T",
    "model": "${CARTRIDGES_TOKASAURUS_MODEL}",
    "seed": 42,
    "slices": [${SLICES_CSV}],
    "started_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds") + "Z",
}
open("${RUN_DIR}/manifest.json", "w").write(json.dumps(m, indent=2))
print("  manifest -> ${RUN_TAG}/manifest.json")
PY

first_dir() {
    local pattern="$1"; local f
    for f in $pattern; do [ -e "$f" ] && { echo "$f"; return 0; }; done
    return 0
}

# ─── 0. Build per-slice + cumulative corpora from wp_text DBs ────────────────
if [[ ! -f "$RUN_DIR/corpora/cumulative_${SLICES_ARR[$((N_PHASES-1))]}/corpus.txt" ]]; then
    echo "[chain] $(date -u)  building per-slice + cumulative corpora into $RUN_DIR/corpora"
    "$PYTHON" "$SCRIPTS/data_adapter.py" \
        --out-dir "$RUN_DIR/corpora" \
        --slices "${SLICES_ARR[@]}"
fi

# ─── 1. Synth (one tokasaurus, all phases sequentially) ──────────────────────
N_SYNTH=${CARTRIDGES_TWIKI_N_SYNTH:-8192}

need_server=0
if [[ "${SKIP_SYNTH:-0}" != "1" ]]; then
    for i in $(seq 1 $N_PHASES); do
        slice=${SLICES_ARR[$((i-1))]}
        if [[ -z "$(first_dir "$RUN_DIR/synth/${slice}/*-synth_config/twiki_p${i}_${slice}_synth_*/*/artifact/dataset.parquet" || true)" ]]; then
            need_server=1; break
        fi
    done
fi

TOKA_LOG="$RUN_DIR/logs/tokasaurus_server.log"
TOKA_PID=
if (( need_server )); then
    [ -x "$TOKA_BIN" ] || { echo "!! tokasaurus venv missing at $TOKASAURUS_VENV"; exit 2; }
    echo "[chain] $(date -u)  spinning up Tokasaurus → $TOKA_LOG"
    FLASHINFER_CACHE="$HOME/.cache/flashinfer"
    [ -d "$FLASHINFER_CACHE" ] && rm -rf "$FLASHINFER_CACHE"
    nohup "$TOKA_BIN" \
        model="$CARTRIDGES_TOKASAURUS_MODEL" \
        kv_cache_num_tokens="$CARTRIDGES_TOKASAURUS_KV_TOKENS" \
        max_topk_logprobs=20 \
        dp_size="$CARTRIDGES_TOKASAURUS_GPUS" \
        > "$TOKA_LOG" 2>&1 &
    TOKA_PID=$!; disown $TOKA_PID
    : "${CARTRIDGES_TOKASAURUS_BOOT_TIMEOUT_SEC:=1200}"
    deadline=$(( $(date +%s) + CARTRIDGES_TOKASAURUS_BOOT_TIMEOUT_SEC ))
    until "$TOKA_PYTHON" -c "import requests; requests.get('${CARTRIDGES_TOKASAURUS_URL}/v1/models', timeout=2)" 2>/dev/null; do
        if [[ $(date +%s) -gt $deadline ]]; then
            echo "  !! tokasaurus didn't come up in ${CARTRIDGES_TOKASAURUS_BOOT_TIMEOUT_SEC}s; tail of $TOKA_LOG:"
            tail -20 "$TOKA_LOG"; kill $TOKA_PID 2>/dev/null || true; exit 2
        fi
        if ! kill -0 $TOKA_PID 2>/dev/null; then
            echo "  !! tokasaurus exited; see $TOKA_LOG"; tail -20 "$TOKA_LOG"; exit 2
        fi
        sleep 3
    done
    echo "  tokasaurus up (pid=$TOKA_PID, $CARTRIDGES_TOKASAURUS_URL)"
fi
trap '[ -n "$TOKA_PID" ] && kill $TOKA_PID 2>/dev/null || true' EXIT

i=0
for slice in "${SLICES_ARR[@]}"; do
    i=$((i+1))
    synth_pq=$(first_dir "$RUN_DIR/synth/${slice}/*-synth_config/twiki_p${i}_${slice}_synth_*/*/artifact/dataset.parquet" || true)
    if [[ -f "$synth_pq" || "${SKIP_SYNTH:-0}" == "1" ]]; then
        echo "[chain] phase $i $slice synth: SKIP ($synth_pq)"
        continue
    fi
    log="$RUN_DIR/logs/synth_p${i}_${slice}.log"
    echo "[chain] $(date -u)  phase $i $slice synth on cumulative_${slice} (n=$N_SYNTH) → $log"
    CARTRIDGES_TWIKI_SLICE=$slice \
    CARTRIDGES_TWIKI_PHASE=$i \
    CARTRIDGES_TWIKI_CORPUS_ROOT=$RUN_DIR/corpora \
    CARTRIDGES_OUTPUT_DIR=$RUN_DIR/synth/${slice} \
    CARTRIDGES_TWIKI_N_SYNTH=$N_SYNTH \
    CARTRIDGES_TOKASAURUS_URL="$CARTRIDGES_TOKASAURUS_URL" \
        "$PYTHON" "$SCRIPTS/synth_config.py" > "$log" 2>&1 \
        || { echo "  !! synth failed; see $log"; tail -10 "$log"; [ -n "$TOKA_PID" ] && kill $TOKA_PID 2>/dev/null || true; exit 3; }
done

# Tear down toka before train
if [[ -n "$TOKA_PID" ]]; then
    echo "[chain] $(date -u)  tearing down tokasaurus (pid=$TOKA_PID)"
    kill $TOKA_PID 2>/dev/null || true
    wait $TOKA_PID 2>/dev/null || true
    pkill -9 -u "$USER" -f "venv-tokasaurus.*multiprocessing-fork" 2>/dev/null || true
    pkill -9 -u "$USER" -f "venv-tokasaurus.*resource_tracker" 2>/dev/null || true
    drain_deadline=$(( $(date +%s) + 60 ))
    while true; do
        max_used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{if($1+0>max) max=$1+0} END {print max+0}')
        [[ "${max_used_mib:-0}" -lt 1024 ]] && break
        [[ $(date +%s) -gt $drain_deadline ]] && break
        sleep 2
    done
    echo "  GPU memory drained, ready for train"
    TOKA_PID=
fi
trap - EXIT

# ─── 2. Train one cartridge per phase ────────────────────────────────────────
declare -A CART_DIR
i=0
for slice in "${SLICES_ARR[@]}"; do
    i=$((i+1))
    cart_dir=$(first_dir "$RUN_DIR/train/p${i}/*-train_config/*" || true)
    cart_has_cache=0
    if [[ -n "$cart_dir" ]]; then
        if [[ -f "$cart_dir/cache_last.pt" ]] || ls "$cart_dir"/cache-step*.pt >/dev/null 2>&1; then
            cart_has_cache=1
        fi
    fi
    if [[ "$cart_has_cache" == "1" || "${SKIP_TRAIN:-0}" == "1" ]]; then
        CART_DIR[$slice]=$cart_dir
        echo "[chain] phase $i $slice train: SKIP ($cart_dir)"
        continue
    fi
    if [[ -n "$cart_dir" ]]; then
        echo "[chain] phase $i $slice train: empty dir found ($cart_dir) — wiping for retry"
        rm -rf "$RUN_DIR/train/p${i}"
    fi
    synth_pq=$(first_dir "$RUN_DIR/synth/${slice}/*-synth_config/twiki_p${i}_${slice}_synth_*/*/artifact/dataset.parquet" || true)
    if [[ -z "$synth_pq" ]]; then
        echo "  !! no synth parquet for $slice; skipping train"
        continue
    fi
    log="$RUN_DIR/logs/train_p${i}_${slice}.log"
    echo "[chain] $(date -u)  phase $i $slice train (cumulative_${slice}) → $log"
    CARTRIDGES_TWIKI_SLICE=$slice \
    CARTRIDGES_TWIKI_PHASE=$i \
    CARTRIDGES_TWIKI_SYNTH_PARQUET=$synth_pq \
    CARTRIDGES_OUTPUT_DIR=$RUN_DIR/train/p${i} \
        "$PYTHON" "$SCRIPTS/train_config.py" > "$log" 2>&1 \
        || { echo "  !! train failed; see $log"; tail -10 "$log"; exit 4; }
    CART_DIR[$slice]=$(first_dir "$RUN_DIR/train/p${i}/*-train_config/*" || true)
    if [[ -z "${CART_DIR[$slice]:-}" ]]; then
        echo "  !! train succeeded but cartridge dir not found"; exit 4
    fi
    if [[ ! -f "${CART_DIR[$slice]}/cache_last.pt" ]] && \
       ! ls "${CART_DIR[$slice]}"/cache-step*.pt >/dev/null 2>&1; then
        echo "  !! train returned 0 but no cache file under ${CART_DIR[$slice]}"; exit 4
    fi
done

# ─── 3. Eval matrix (3 phases × 4 eval sets = 12 cells, 8-way parallel) ──────
if [[ "${SKIP_EVAL:-0}" == "1" ]]; then
    echo "[chain] SKIP_EVAL=1 — done after train"
    touch "$RUN_DIR/DONE"
    exit 0
fi

EVAL_PARALLEL=${EVAL_PARALLEL:-8}
N_GPUS=${CARTRIDGES_TWIKI_N_GPUS:-8}

declare -a EVAL_JOBS
i=0
for slice_i in "${SLICES_ARR[@]}"; do
    i=$((i+1))
    if [[ -z "${CART_DIR[$slice_i]:-}" ]]; then
        echo "  !! no cartridge for $slice_i; can't eval phase $i"
        continue
    fi
    cart="${CART_DIR[$slice_i]}"
    for eval_set in "${EVAL_SETS_ARR[@]}"; do
        out=$PRED_ROOT/p${i}/${eval_set}.parquet
        if [[ -f "$out" ]]; then
            echo "[chain] phase $i eval $eval_set: SKIP (parquet exists)"
            continue
        fi
        EVAL_JOBS+=("${i}|${slice_i}|${eval_set}|${cart}|${VAL_PARQUETS[$eval_set]}")
    done
done

n_total=${#EVAL_JOBS[@]}
echo "[chain] $(date -u)  eval: $n_total cells to run, $EVAL_PARALLEL in parallel (one GPU each)"
batch_idx=0
while (( ${#EVAL_JOBS[@]} > 0 )); do
    pids=(); fails=()
    n_to_run=$(( EVAL_PARALLEL < ${#EVAL_JOBS[@]} ? EVAL_PARALLEL : ${#EVAL_JOBS[@]} ))
    for ((k=0; k<n_to_run; k++)); do
        IFS='|' read -r ii slice_i eval_set cart val_pq <<< "${EVAL_JOBS[k]}"
        gpu=$(( k % N_GPUS ))
        out=$PRED_ROOT/p${ii}/${eval_set}.parquet
        log="$RUN_DIR/logs/eval_p${ii}_${eval_set}.log"
        mkdir -p "$(dirname "$out")"
        echo "[chain]   batch $batch_idx GPU $gpu: phase $ii eval $eval_set → $log"
        cell_cache="$RUN_DIR/.triton_cache/p${ii}_${eval_set}"
        cell_inductor="$RUN_DIR/.inductor_cache/p${ii}_${eval_set}"
        mkdir -p "$cell_cache" "$cell_inductor"
        CUDA_VISIBLE_DEVICES=$gpu \
        TRITON_CACHE_DIR="$cell_cache" \
        TORCHINDUCTOR_CACHE_DIR="$cell_inductor" \
        TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=ATEN \
            "$PYTHON" "$SCRIPTS/eval_temporalwiki.py" \
                --cartridge "$cart" \
                --eval-set "$eval_set" --train-slice "$slice_i" --phase-idx "$ii" \
                --val-parquet "$val_pq" \
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
echo "  predictions: $PRED_ROOT/p{1..3}/{drift_s1,drift_s2,drift_s3,stable}.parquet"
