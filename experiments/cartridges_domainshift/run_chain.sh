#!/bin/bash
# Cartridges on the DOMAIN-SHIFT sequence (ToolUse -> FinQA -> SciKE-Bio).
#
# Structure mirrors experiments/cartridges_finance/run_chain.sh (Tokasaurus
# can't share the GPU with train, so synth ALL phases first, tear the server
# down, then train ALL phases):
#
#   0. Build per-task + cumulative corpora from $DATA_DIR
#   1. Boot Tokasaurus (Qwen3-8B, dp_size=8) once, synth N samples per phase
#      against the live server
#   2. Tear down Tokasaurus, drain GPU memory
#   3. Train one cartridge per phase (FSDP, uses all 8 GPUs)
#   4. Eval the composed cartridge on every task's val split (eval needs the
#      cartridge served on an OpenAI-compatible endpoint — see EVAL_MODEL)
#
# Resume-safe: re-running skips any synth / train / eval whose output exists.
#
# Required env:  OPENROUTER_API_KEY (eval only), CARTRIDGES_DIR (default ok),
#                CARTRIDGES_VENV (default ok), TOKASAURUS_VENV (default ok)
# Optional:      TASKS, DATA_DIR, RUN_DIR, N_SYNTH, SKIP_SYNTH, SKIP_TRAIN,
#                SKIP_EVAL, EVAL_MODEL, CARTRIDGES_TOKASAURUS_{GPUS,KV_TOKENS}
set -euo pipefail

# Path roots — defined before any defaults below that reference $REPO_ROOT.
SCRIPTS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPTS/../.." && pwd)

# Defensive: kill any orphan tokasaurus python procs from prior runs that died
# via OOM or wrapper-timeout. Their multiprocessing children survive the wrapper
# exit and hold GPU memory, OOMing the next attempt.
if pgrep -u "$USER" -f "venv-tokasaurus.*(multiprocessing-fork|resource_tracker)" >/dev/null 2>&1; then
    echo "[chain] $(date -u)  killing orphan tokasaurus procs from prior runs"
    pkill -9 -u "$USER" -f "venv-tokasaurus.*multiprocessing-fork" 2>/dev/null || true
    pkill -9 -u "$USER" -f "venv-tokasaurus.*resource_tracker" 2>/dev/null || true
    sleep 3
fi

: "${CARTRIDGES_VENV:=${CL_HOME:-$HOME}/venv-cartridges}"
: "${TOKASAURUS_VENV:=${CL_HOME:-$HOME}/venv-tokasaurus}"
: "${CARTRIDGES_DIR:=$REPO_ROOT/methods/compression/cartridges}"
: "${CARTRIDGES_TOKASAURUS_URL:=http://localhost:10210}"
: "${CARTRIDGES_TOKASAURUS_GPUS:=8}"
: "${CARTRIDGES_TOKASAURUS_MODEL:=Qwen/Qwen3-8B}"
# KV-cache budget per data-parallel rank (in tokens). 256k fits Qwen3-8B on
# H100-80GB comfortably (~38 GB KV + 16 GB weights). The cartridges README's
# (512*1024) OOMs.
: "${CARTRIDGES_TOKASAURUS_KV_TOKENS:=(256 * 1024)}"

# Auto-source the cartridges venv (shared MooseFS).
if [ -z "${VIRTUAL_ENV:-}" ] || [ "${VIRTUAL_ENV}" != "${CARTRIDGES_VENV}" ]; then
    if [ -f "${CARTRIDGES_VENV}/bin/activate" ]; then
        # shellcheck source=/dev/null
        source "${CARTRIDGES_VENV}/bin/activate"
    else
        echo "!! cartridges venv missing at ${CARTRIDGES_VENV}; run install.sh first" >&2
        exit 2
    fi
fi

PYTHON=${CARTRIDGES_VENV}/bin/python
TOKA_PYTHON=${TOKASAURUS_VENV}/bin/python
TOKA_BIN=${TOKASAURUS_VENV}/bin/toka

TASKS=${TASKS:-"tooluse finqa sciknoweval_bio"}
read -r -a TASKS_ARR <<< "$TASKS"
DATA_DIR=${DATA_DIR:-$REPO_ROOT/datasets}
RUN_DIR=${RUN_DIR:-${CL_HOME:-$HOME}/cartridges_domainshift/runs/domainshift_cartridges_s42}
N_SYNTH=${N_SYNTH:-8192}
EVAL_MODEL=${EVAL_MODEL:-qwen-3-8b}   # profile must point at the served cartridge endpoint

mkdir -p "$RUN_DIR"/{corpora,synth,train,logs}
export CARTRIDGES_DIR CARTRIDGES_OUTPUT_DIR="$RUN_DIR"
export PYTHONPATH="$SCRIPTS:${PYTHONPATH:-}"

# Resume helper: emit first matching glob path, or nothing.
first_dir() {
    for f in $1; do
        [ -e "$f" ] && { echo "$f"; return 0; }
    done
    return 0
}

# ── 0. Build per-task + cumulative corpora ───────────────────────────────────
echo "[chain] $(date -u)  building corpora from $DATA_DIR into $RUN_DIR/corpora"
"$PYTHON" "$SCRIPTS/data_adapter.py" \
    --data-dir "$DATA_DIR" --tasks "${TASKS_ARR[@]}" \
    --out-dir "$RUN_DIR/corpora" --cumulative \
    > "$RUN_DIR/logs/corpora.log" 2>&1
tail -n 8 "$RUN_DIR/logs/corpora.log" | sed 's/^/  /'

# ── 1. Boot Tokasaurus + synth all phases ────────────────────────────────────
# Decide whether to start the server (skip if all synth parquets present, or
# SKIP_SYNTH=1).
need_server=0
if [[ "${SKIP_SYNTH:-0}" != "1" ]]; then
    i=0
    for task in "${TASKS_ARR[@]}"; do
        i=$((i+1))
        existing=$(first_dir "$RUN_DIR/synth/p${i}/*-synth_config/*/*/artifact/dataset.parquet" || true)
        if [[ -z "$existing" ]]; then need_server=1; break; fi
    done
fi

TOKA_LOG="$RUN_DIR/logs/tokasaurus_server.log"
TOKA_PID=
if (( need_server )); then
    [ -x "$TOKA_BIN" ] || { echo "!! tokasaurus venv missing at $TOKASAURUS_VENV — run install.sh"; exit 2; }
    echo "[chain] $(date -u)  spinning up Tokasaurus (Qwen3-8b dp_size=${CARTRIDGES_TOKASAURUS_GPUS}) → $TOKA_LOG"
    # Nuke shared flashinfer JIT cache: stale .so compiled against a different
    # CUDA toolkit produces `undefined symbol: cudaGetDriverEntryPointByVersion`
    # at load time. ~2-5 min rebuild is cheaper than the debug cycle.
    FLASHINFER_CACHE="$HOME/.cache/flashinfer"
    if [ -d "$FLASHINFER_CACHE" ]; then
        echo "[chain] $(date -u)  clearing flashinfer cache at $FLASHINFER_CACHE"
        rm -rf "$FLASHINFER_CACHE"
    fi
    nohup "$TOKA_BIN" \
        model="$CARTRIDGES_TOKASAURUS_MODEL" \
        kv_cache_num_tokens="$CARTRIDGES_TOKASAURUS_KV_TOKENS" \
        max_topk_logprobs=20 \
        dp_size="$CARTRIDGES_TOKASAURUS_GPUS" \
        > "$TOKA_LOG" 2>&1 &
    TOKA_PID=$!
    disown $TOKA_PID
    # Wait until server is reachable. Cold flashinfer JIT compile can take
    # 8-12 min; subsequent runs hit the venv-isolated cache (<2 min).
    : "${CARTRIDGES_TOKASAURUS_BOOT_TIMEOUT_SEC:=1200}"
    deadline=$(( $(date +%s) + CARTRIDGES_TOKASAURUS_BOOT_TIMEOUT_SEC ))
    until "$TOKA_PYTHON" -c "import requests; requests.get('${CARTRIDGES_TOKASAURUS_URL}/v1/models', timeout=2)" 2>/dev/null; do
        if [[ $(date +%s) -gt $deadline ]]; then
            echo "  !! tokasaurus didn't come up in ${CARTRIDGES_TOKASAURUS_BOOT_TIMEOUT_SEC}s; tail of $TOKA_LOG:"
            tail -20 "$TOKA_LOG"
            kill $TOKA_PID 2>/dev/null || true
            exit 2
        fi
        if ! kill -0 $TOKA_PID 2>/dev/null; then
            echo "  !! tokasaurus exited; see $TOKA_LOG"; tail -20 "$TOKA_LOG"; exit 2
        fi
        sleep 3
    done
    echo "  tokasaurus up (pid=$TOKA_PID, ${CARTRIDGES_TOKASAURUS_URL})"
fi
trap '[ -n "$TOKA_PID" ] && kill $TOKA_PID 2>/dev/null || true' EXIT

# Synth all phases against the live server
i=0
for task in "${TASKS_ARR[@]}"; do
    i=$((i+1))
    corpus="$RUN_DIR/corpora/cumulative_p${i}/corpus.txt"
    synth_pq=$(first_dir "$RUN_DIR/synth/p${i}/*-synth_config/*/*/artifact/dataset.parquet" || true)
    if [[ -f "$synth_pq" || "${SKIP_SYNTH:-0}" == "1" ]]; then
        echo "[chain] phase $i ($task) synth: SKIP ($synth_pq)"
        continue
    fi
    log="$RUN_DIR/logs/synth_p${i}.log"
    echo "[chain] $(date -u)  phase $i ($task) synth on cumulative_p${i} (n=$N_SYNTH) → $log"
    CARTRIDGES_FINANCE_PHASE=$i \
    CARTRIDGES_FINANCE_YEAR=$i \
    CARTRIDGES_FINANCE_CORPUS_FILE="$corpus" \
    CARTRIDGES_FINANCE_N_SYNTH="$N_SYNTH" \
    CARTRIDGES_OUTPUT_DIR="$RUN_DIR/synth/p${i}" \
    CARTRIDGES_TOKASAURUS_URL="$CARTRIDGES_TOKASAURUS_URL" \
        "$PYTHON" "$SCRIPTS/synth_config.py" > "$log" 2>&1 \
        || { echo "  !! synth failed; see $log"; tail -10 "$log"; [ -n "$TOKA_PID" ] && kill $TOKA_PID 2>/dev/null || true; exit 3; }
done

# ── 2. Tear down Tokasaurus, drain GPUs, prepare for train ───────────────────
if [[ -n "$TOKA_PID" ]]; then
    echo "[chain] $(date -u)  tearing down tokasaurus (pid=$TOKA_PID)"
    kill $TOKA_PID 2>/dev/null || true
    wait $TOKA_PID 2>/dev/null || true
    # The toka parent doesn't auto-kill its dp_worker mp.spawn children. Without
    # this they keep their ~28 GB GPU allocations and the train phase OOMs
    # loading Qwen3-8B onto already-full GPUs.
    pkill -9 -u "$USER" -f "venv-tokasaurus.*multiprocessing-fork" 2>/dev/null || true
    pkill -9 -u "$USER" -f "venv-tokasaurus.*resource_tracker" 2>/dev/null || true
    drain_deadline=$(( $(date +%s) + 60 ))
    while true; do
        max_used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{if($1+0>max) max=$1+0} END {print max+0}')
        [[ "${max_used_mib:-0}" -lt 1024 ]] && break
        [[ $(date +%s) -gt $drain_deadline ]] && { echo "  !! GPU memory didn't drain in 60s; max used = ${max_used_mib} MiB"; break; }
        sleep 2
    done
    echo "  GPU memory drained, ready for train"
    TOKA_PID=
fi
trap - EXIT

# ── 3. Train one cartridge per phase ─────────────────────────────────────────
i=0
for task in "${TASKS_ARR[@]}"; do
    i=$((i+1))
    cart_dir=$(first_dir "$RUN_DIR/train/p${i}/*-train_config/*" || true)
    cart_has_cache=0
    if [[ -n "$cart_dir" ]]; then
        if [[ -f "$cart_dir/cache_last.pt" ]] || ls "$cart_dir"/cache-step*.pt >/dev/null 2>&1; then
            cart_has_cache=1
        fi
    fi
    if [[ "$cart_has_cache" == "1" || "${SKIP_TRAIN:-0}" == "1" ]]; then
        echo "[chain] phase $i ($task) train: SKIP ($cart_dir)"
        continue
    fi
    # Wipe stale empty dir from a prior crash so cartridges' pydantic creates
    # a fresh dated subdir for this attempt.
    if [[ -n "$cart_dir" ]]; then
        echo "[chain] phase $i ($task) train: empty dir found ($cart_dir) — wiping for retry"
        rm -rf "$RUN_DIR/train/p${i}"
    fi
    synth_pq=$(first_dir "$RUN_DIR/synth/p${i}/*-synth_config/*/*/artifact/dataset.parquet" || true)
    if [[ -z "$synth_pq" ]]; then
        echo "  !! no synth parquet for p${i}; skipping train"
        continue
    fi
    corpus="$RUN_DIR/corpora/cumulative_p${i}/corpus.txt"
    log="$RUN_DIR/logs/train_p${i}.log"
    echo "[chain] $(date -u)  phase $i ($task) train (cumulative_p${i}) → $log"
    CARTRIDGES_FINANCE_PHASE=$i \
    CARTRIDGES_FINANCE_YEAR=$i \
    CARTRIDGES_FINANCE_CORPUS_FILE="$corpus" \
    CARTRIDGES_FINANCE_SYNTH_PARQUET="$synth_pq" \
    CARTRIDGES_OUTPUT_DIR="$RUN_DIR/train/p${i}" \
        "$PYTHON" "$SCRIPTS/train_config.py" > "$log" 2>&1 \
        || { echo "  !! train failed; see $log"; tail -10 "$log"; exit 4; }
done

# ── 4. Eval the composed cartridge on every task per phase ───────────────────
if [[ "${SKIP_EVAL:-0}" == "1" ]]; then
    echo "[chain] SKIP_EVAL=1 — done after train"
else
    i=0
    for task in "${TASKS_ARR[@]}"; do
        i=$((i+1))
        for et in "${TASKS_ARR[@]}"; do
            "$REPO_ROOT/.venv-harness/bin/python" "$REPO_ROOT/scripts/eval_prompt.py" \
                --task "$et" --model "$EVAL_MODEL" --prompt-text "{question}" \
                > "$RUN_DIR/logs/eval_p${i}_${et}.log" 2>&1 || \
                echo "  (eval p$i/$et needs the served cartridge endpoint; see README)"
        done
    done
fi

touch "$RUN_DIR/DONE"
echo "[chain] $(date -u)  done -> $RUN_DIR"
