#!/bin/bash
# Cartridges finance ordering-F sweep — 6 synth + 6 train + 36 eval cells.
#
# Output layout (matches the verl-trained run conventions used by sft/sdft/grpo/sdpo):
#
#   $RUN_ROOT/finance-cl-cartridges_orderF_nothink_s42/
#       corpora/y{YYYY}/{corpus.txt, manifest.json}                per-year text corpora
#       synth/y{YYYY}/<TIMESTAMP>-synth_config/finance_y{YYYY}_synth_*/<MODEL>_n*/artifact/dataset.parquet   self-study Q/A
#       train/finance_y{YYYY}_cartridge/cache-step{N}.pt           cartridge checkpoints
#       logs/{synth,train,eval}_p{i}_y{YYYY}.log                   per-step logs
#       manifest.json                                              run-level provenance
#       DONE                                                       written when all 36 cells land
#
#   $PRED_ROOT/cartridges/phase{i}/y{YYYY}.parquet         (Anastasia format,
#                                                          consumed by 10k_hf_analysis.py
#                                                          and our run_finance_analysis.sh)
#
# Resume-safe: re-running skips any synth / train / eval whose output exists.
#
# Required env vars:  OPENROUTER_API_KEY, CARTRIDGES_DIR (default ok), CARTRIDGES_VENV (default ok)
# Optional knobs:     YEARS, CARTRIDGES_FINANCE_N_SYNTH, CARTRIDGES_FINANCE_EPOCHS,
#                     CARTRIDGES_FINANCE_LR, SKIP_SYNTH, SKIP_TRAIN, SKIP_EVAL,
#                     CARTRIDGES_WANDB_PROJECT, CARTRIDGES_WANDB_DISABLE

set -euo pipefail

# Path roots — must come BEFORE the default for CARTRIDGES_DIR below references
# $REPO_ROOT. `set -u` treats reading an unset var as an error.
SCRIPTS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPTS/../.." && pwd)

# Defensive cleanup: kill any orphan tokasaurus python procs from prior runs
# that died via OOM or wrapper-timeout. Their multiprocessing children survive
# the slurm wrapper exit and hold GPU memory, which OOMs the next attempt.
if pgrep -u "$USER" -f "venv-tokasaurus.*(multiprocessing-fork|resource_tracker)" >/dev/null 2>&1; then
    echo "[chain] $(date -u)  killing orphan tokasaurus procs from prior runs"
    pkill -9 -u "$USER" -f "venv-tokasaurus.*multiprocessing-fork" 2>/dev/null || true
    pkill -9 -u "$USER" -f "venv-tokasaurus.*resource_tracker" 2>/dev/null || true
    sleep 3
fi

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
# KV-cache budget per data-parallel rank (in tokens). Cartridges README's
# (512*1024) OOMs on Qwen3-8B/H100-80GB: 36 layers × 8 KV heads × 128 head_dim
# × bf16 ≈ 144 KB/token → 512k tok = 74 GB KV + 16 GB weights > 80 GB. 256k
# leaves ~38 GB KV + 16 GB weights = 54 GB, comfortable. Smoke uses 64k.
: "${CARTRIDGES_TOKASAURUS_KV_TOKENS:=(256 * 1024)}"

# Auto-source the cartridges venv (shared MooseFS, both nodes can use).
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
RUN_TAG=${CARTRIDGES_FINANCE_RUN_TAG:-finance-cl-cartridges_orderF_nothink_s42}
RUN_ROOT_DEFAULT=${CL_HOME:-$REPO_ROOT/runs}/cartridges_finance/runs
RUN_DIR=${CARTRIDGES_FINANCE_RUN_DIR:-${RUN_ROOT_DEFAULT}/${RUN_TAG}}
PRED_ROOT=${CARTRIDGES_FINANCE_PRED_ROOT:-${CL_HOME:-$REPO_ROOT/runs}/results/finance_predictions/cartridges}
# Per-year parquets dir: canonical CL_FINANCE_DATA (matches bootstrap.sh + run_sequential.py),
# CARTRIDGES_FINANCE_DATA is a method-specific override.
DATA_DIR=${CARTRIDGES_FINANCE_DATA:-${CL_FINANCE_DATA:-$REPO_ROOT/data/finance_yearly}}
RETURNS_TABLE=${CARTRIDGES_FINANCE_RETURNS:-${DATA_DIR%/cl_yearly}/finance_returns_table.parquet}

YEARS=${YEARS:-"2015 2016 2017 2018 2019 2020"}
read -r -a YEARS_ARR <<< "$YEARS"

mkdir -p "$RUN_DIR"/{corpora,synth,train,logs} "$PRED_ROOT"

# Tell cartridges (and our subscripts) where to live + where to log.
export CARTRIDGES_DIR
export CARTRIDGES_OUTPUT_DIR=$RUN_DIR
export CARTRIDGES_FINANCE_CORPUS_ROOT=$RUN_DIR/corpora
export OPENROUTER_API_BASE
export PYTHONPATH="$SCRIPTS:${PYTHONPATH:-}"
# Same wandb project as sft/sdft/grpo/sdpo so the 5-method comparison lives in
# one dashboard. Group name = run_tag for easy filtering.
: "${CARTRIDGES_WANDB_PROJECT:=sdpo_seq}"
export CARTRIDGES_WANDB_PROJECT
export CARTRIDGES_FINANCE_RUN_TAG="$RUN_TAG"

# Comma-join YEARS_ARR for embedding into the Python heredoc below; bash's
# ${arr[*]} produces space-separated, but Python list literals need commas.
YEARS_CSV=$(IFS=,; echo "${YEARS_ARR[*]}")

# Run-level manifest (provenance). One canonical place.
"$PYTHON" - <<PY
import json, os, sys, datetime as dt, subprocess
m = {
    "run_tag": "${RUN_TAG}",
    "run_dir": "${RUN_DIR}",
    "method": "cartridges",
    "ordering": "F",
    "model": "Qwen/Qwen3-8B",
    "seed": 42,
    "years": [${YEARS_CSV}],
    "started_at": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    "n_synth": int(os.environ.get("CARTRIDGES_FINANCE_N_SYNTH", "2048")),
    "epochs": int(os.environ.get("CARTRIDGES_FINANCE_EPOCHS", "1")),
    "lr": float(os.environ.get("CARTRIDGES_FINANCE_LR", "2e-2")),
    "openrouter_model": "qwen/qwen3-8b",
    "openrouter_provider_pin": "alibaba",
    "wandb_project": os.environ.get("CARTRIDGES_WANDB_PROJECT"),
}
try:
    m["cartridges_commit"] = subprocess.check_output(
        ["git", "-C", os.environ["CARTRIDGES_DIR"], "rev-parse", "HEAD"], text=True).strip()
except Exception: pass
with open("${RUN_DIR}/manifest.json", "w") as f:
    json.dump(m, f, indent=2)
print(f"  manifest -> {os.path.basename('${RUN_DIR}')}/manifest.json")
PY

# ────────────────────────────── 0. Build per-year + cumulative corpora ───────
# Cumulative-corpus CL recipe: each phase's cartridge trains on union(y1..y_i),
# with docs shuffled so KVFromRandomText init isn't biased toward early years.
echo "[chain] $(date -u)  building per-year + cumulative corpora into $RUN_DIR/corpora"
"$PYTHON" "$SCRIPTS/data_adapter.py" \
    --data-dir "$DATA_DIR" \
    --years "${YEARS_ARR[@]}" \
    --out-dir "$RUN_DIR/corpora" \
    --cumulative \
    > "$RUN_DIR/logs/corpora.log" 2>&1
tail -10 "$RUN_DIR/logs/corpora.log" | sed 's/^/  /'

# ────────────────────────────── helper: pick first matching dir ───────────────
# NB: must NOT quote the glob — ls doesn't do globbing itself, only the shell
# does. We rely on bash's pathname expansion at the loop. If the pattern has
# no matches, `for f in $pattern` iterates once with the literal string, which
# fails the -e test and yields nothing.
first_dir() {
    local pattern="$1"
    local f
    for f in $pattern; do
        [ -e "$f" ] && { echo "$f"; return 0; }
    done
    return 0
}

# ────────────────────────────── 1. Synthesize per phase ───────────────────────
# Strategy: spin up ONE Tokasaurus server (in venv-tokasaurus) before any synth,
# do all 6 synth jobs against it (sequentially per phase), tear it down before
# train (so we get all 8 GPUs back for FSDP).
N_SYNTH=${CARTRIDGES_FINANCE_N_SYNTH:-8192}

# Decide whether we need to start the server at all (skip if every phase has
# its synth parquet already, or SKIP_SYNTH=1).
need_server=0
if [[ "${SKIP_SYNTH:-0}" != "1" ]]; then
    for y in "${YEARS_ARR[@]}"; do
        if [[ -z "$(first_dir "$RUN_DIR/synth/y${y}/*-synth_config/finance_y${y}_synth_*/*/artifact/dataset.parquet" || true)" ]]; then
            need_server=1; break
        fi
    done
fi

TOKA_LOG="$RUN_DIR/logs/tokasaurus_server.log"
TOKA_PID=
if (( need_server )); then
    [ -x "$TOKA_BIN" ] || { echo "!! tokasaurus venv missing at $TOKASAURUS_VENV — run install.sh"; exit 2; }
    echo "[chain] $(date -u)  spinning up Tokasaurus (Qwen3-8b dp_size=${CARTRIDGES_TOKASAURUS_GPUS}) → $TOKA_LOG"
    # Nuke the shared flashinfer JIT cache. flashinfer hardcodes ~/.cache/flashinfer/<arch>/
    # and ignores any env override; if a stale .so was compiled under a different CUDA
    # toolkit on PATH (e.g. system /usr/local/cuda 12.5+ vs torch's bundled libcudart 12.4),
    # loads fail with `undefined symbol: cudaGetDriverEntryPointByVersion`. Cheaper to
    # rebuild from scratch (~2-5 min cold) than to debug poisoning.
    FLASHINFER_CACHE="$HOME/.cache/flashinfer"
    if [ -d "$FLASHINFER_CACHE" ]; then
        echo "[chain] $(date -u)  clearing flashinfer cache at $FLASHINFER_CACHE"
        rm -rf "$FLASHINFER_CACHE"
    fi
    # Server flags follow cartridges README §1.2 (Option B: Local Tokasaurus).
    # NB: the actual attribute on sabri/batch is `max_topk_logprobs`
    # (see tokasaurus/common_types.py); the README's `max_top_logprobs` was
    # the older name.
    nohup "$TOKA_BIN" \
        model="$CARTRIDGES_TOKASAURUS_MODEL" \
        kv_cache_num_tokens="$CARTRIDGES_TOKASAURUS_KV_TOKENS" \
        max_topk_logprobs=20 \
        dp_size="$CARTRIDGES_TOKASAURUS_GPUS" \
        > "$TOKA_LOG" 2>&1 &
    TOKA_PID=$!
    disown $TOKA_PID
    # Wait until the server is reachable. Cold flashinfer JIT compile of all
    # sm90 attention kernels can take 8-12 min on first run; subsequent runs
    # use the venv-isolated cache and come up in <2 min. Default 20 min
    # deadline accommodates both.
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

i=0
for y in "${YEARS_ARR[@]}"; do
    i=$((i+1))
    # Cartridges' synthesize writes to <output_dir>/<run_id>/artifact/dataset.parquet
    synth_pq=$(first_dir "$RUN_DIR/synth/y${y}/*-synth_config/finance_y${y}_synth_*/*/artifact/dataset.parquet" || true)
    if [[ -f "$synth_pq" || "${SKIP_SYNTH:-0}" == "1" ]]; then
        echo "[chain] phase $i y${y} synth: SKIP ($synth_pq)"
        continue
    fi
    log="$RUN_DIR/logs/synth_p${i}_y${y}.log"
    echo "[chain] $(date -u)  phase $i y${y} synth on cumulative_y${i} (n=$N_SYNTH) → $log"
    CARTRIDGES_FINANCE_YEAR=$y \
    CARTRIDGES_FINANCE_PHASE=$i \
    CARTRIDGES_FINANCE_CORPUS_ROOT=$RUN_DIR/corpora \
    CARTRIDGES_OUTPUT_DIR=$RUN_DIR/synth/y${y} \
    CARTRIDGES_FINANCE_N_SYNTH=$N_SYNTH \
    CARTRIDGES_TOKASAURUS_URL="$CARTRIDGES_TOKASAURUS_URL" \
        "$PYTHON" "$SCRIPTS/synth_config.py" > "$log" 2>&1 \
        || { echo "  !! synth failed; see $log"; tail -10 "$log"; [ -n "$TOKA_PID" ] && kill $TOKA_PID 2>/dev/null || true; exit 3; }
done

# Tear down tokasaurus before train (we want all 8 GPUs back for FSDP).
if [[ -n "$TOKA_PID" ]]; then
    echo "[chain] $(date -u)  tearing down tokasaurus (pid=$TOKA_PID)"
    kill $TOKA_PID 2>/dev/null || true
    wait $TOKA_PID 2>/dev/null || true
    # The toka parent doesn't auto-kill its 8 dp_worker mp.spawn children.
    # Without this, those workers keep their ~28 GB GPU allocations and the
    # subsequent train phase OOMs trying to load Qwen3-8B onto already-full GPUs.
    pkill -9 -u "$USER" -f "venv-tokasaurus.*multiprocessing-fork" 2>/dev/null || true
    pkill -9 -u "$USER" -f "venv-tokasaurus.*resource_tracker" 2>/dev/null || true
    # Drain GPU memory: poll until used < 1 GiB on every GPU (or 60s timeout)
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

# ────────────────────────────── 2. Train one cartridge per phase ──────────────
declare -A CART_DIR
i=0
for y in "${YEARS_ARR[@]}"; do
    i=$((i+1))
    # train_config.py sets name=f"finance_y{YEAR}_cartridge"; cartridges writes
    # to <output_dir>/<TIMESTAMP>-train_config/<UUID>/. Resume detection MUST
    # require an actual cache file (cache_last.pt or cache-step*.pt) — not just
    # the run dir, because pydrantic creates the dir before training even
    # starts, so an OOM during model.to(device) leaves an empty dir that the
    # naive existence check would falsely report as "trained".
    cart_dir=$(first_dir "$RUN_DIR/train/y${y}/*-train_config/*" || true)
    cart_has_cache=0
    if [[ -n "$cart_dir" ]]; then
        if [[ -f "$cart_dir/cache_last.pt" ]] || \
           ls "$cart_dir"/cache-step*.pt >/dev/null 2>&1; then
            cart_has_cache=1
        fi
    fi
    if [[ "$cart_has_cache" == "1" || "${SKIP_TRAIN:-0}" == "1" ]]; then
        CART_DIR[$y]=$cart_dir
        echo "[chain] phase $i y${y} train: SKIP ($cart_dir)"
        continue
    fi
    # Stale empty dir from a prior crash — wipe so cartridges' pydrantic
    # creates a fresh dated subdir for this attempt (otherwise the next
    # first_dir glob may pick the empty one again).
    if [[ -n "$cart_dir" ]]; then
        echo "[chain] phase $i y${y} train: empty dir found ($cart_dir) — wiping for retry"
        rm -rf "$RUN_DIR/train/y${y}"
    fi
    synth_pq=$(first_dir "$RUN_DIR/synth/y${y}/*-synth_config/finance_y${y}_synth_*/*/artifact/dataset.parquet" || true)
    if [[ -z "$synth_pq" ]]; then
        echo "  !! no synth parquet for y${y}; skipping train"
        continue
    fi
    log="$RUN_DIR/logs/train_p${i}_y${y}.log"
    echo "[chain] $(date -u)  phase $i y${y} train (cumulative_y${i}) → $log"
    CARTRIDGES_FINANCE_YEAR=$y \
    CARTRIDGES_FINANCE_PHASE=$i \
    CARTRIDGES_FINANCE_CORPUS_ROOT=$RUN_DIR/corpora \
    CARTRIDGES_FINANCE_SYNTH_PARQUET=$synth_pq \
    CARTRIDGES_OUTPUT_DIR=$RUN_DIR/train/y${y} \
        "$PYTHON" "$SCRIPTS/train_config.py" > "$log" 2>&1 \
        || { echo "  !! train failed; see $log"; tail -10 "$log"; exit 4; }
    CART_DIR[$y]=$(first_dir "$RUN_DIR/train/y${y}/*-train_config/*" || true)
    if [[ -z "${CART_DIR[$y]:-}" ]]; then
        echo "  !! train succeeded but cartridge dir not found under $RUN_DIR/train/y${y}/"
        exit 4
    fi
    # Train returned 0 but no cache file was actually written — usually means
    # the train script silently caught an exception or save_every_n_steps was
    # never reached; either way, eval would fail to load a cartridge.
    if [[ ! -f "${CART_DIR[$y]}/cache_last.pt" ]] && \
       ! ls "${CART_DIR[$y]}"/cache-step*.pt >/dev/null 2>&1; then
        echo "  !! train returned 0 but no cache_last.pt or cache-step*.pt under ${CART_DIR[$y]}"
        exit 4
    fi
done

# ────────────────────────────── 3. Eval matrix (one cartridge per phase) ─────
# Per phase i: load ONE cartridge (year y_i), eval on every val year y_j.
# 36 cells total. M[i,j] = cartridge_y_i evaluated on val_y_j → tells us
# forward/backward transfer of a year-specific cartridge.
if [[ "${SKIP_EVAL:-0}" == "1" ]]; then
    echo "[chain] SKIP_EVAL=1 — done after train"
    exit 0
fi

# Parallel eval: each cell loads a fresh Qwen3-8B (~16 GB) onto ONE GPU and
# does 50 prompts × 8 rollouts. We have 8 GPUs, so up to 8 cells in parallel.
# 36 cells total → ~5 batches of 8 → wall time = 5× single-cell time (~10 min)
# = ~50 min vs ~6 hr serial. Set EVAL_PARALLEL=1 to disable parallelism.
EVAL_PARALLEL=${EVAL_PARALLEL:-8}
N_GPUS=${CARTRIDGES_FINANCE_N_GPUS:-8}

# Build the full job list (skip cells whose parquet already exists)
# Note: explicit `=()` init — under `set -u`, a bare `declare -a` leaves the
# variable unbound until first assignment, so `${#EVAL_JOBS[@]}` below
# triggers nounset when every cell is SKIP on a resume.
declare -a EVAL_JOBS=()
i=0
for y_i in "${YEARS_ARR[@]}"; do
    i=$((i+1))
    if [[ -z "${CART_DIR[$y_i]:-}" ]]; then
        echo "  !! no cartridge for y${y_i}; can't eval phase $i"
        continue
    fi
    cart="${CART_DIR[$y_i]}"
    for y_j in "${YEARS_ARR[@]}"; do
        out=$PRED_ROOT/phase${i}/y${y_j}.parquet
        if [[ -f "$out" ]]; then
            echo "[chain] phase $i eval y${y_j}: SKIP (parquet exists)"
            continue
        fi
        EVAL_JOBS+=("${i}|${y_i}|${y_j}|${cart}")
    done
done

# Run jobs in batches of EVAL_PARALLEL, each pinned to one GPU via
# CUDA_VISIBLE_DEVICES so they don't fight over GPU 0.
echo "[chain] $(date -u)  eval: ${#EVAL_JOBS[@]} cells to run, $EVAL_PARALLEL in parallel (one GPU each)"
batch_idx=0
while (( ${#EVAL_JOBS[@]} > 0 )); do
    pids=()
    fails=()
    n_to_run=$(( EVAL_PARALLEL < ${#EVAL_JOBS[@]} ? EVAL_PARALLEL : ${#EVAL_JOBS[@]} ))
    for ((k=0; k<n_to_run; k++)); do
        job="${EVAL_JOBS[k]}"
        IFS='|' read -r ii y_i y_j cart <<< "$job"
        gpu=$(( k % N_GPUS ))
        out=$PRED_ROOT/phase${ii}/y${y_j}.parquet
        log="$RUN_DIR/logs/eval_p${ii}_y${y_j}.log"
        val_pq=$DATA_DIR/val_y${y_j}.parquet
        echo "[chain]   batch $batch_idx GPU $gpu: phase $ii eval y${y_j} → $log"
        # 8 parallel evals contended on the shared ~/.triton/cache during
        # flex_attention compilation, causing NoValidChoicesError (autotune
        # filelock chaos). Give each cell its own cache so they don't race.
        # Also force ATEN as a fallback if Triton autotune still can't pick a
        # kernel for some shape (defensive belt+suspenders).
        cell_cache="$RUN_DIR/.triton_cache/p${ii}_y${y_j}"
        cell_inductor="$RUN_DIR/.inductor_cache/p${ii}_y${y_j}"
        mkdir -p "$cell_cache" "$cell_inductor"
        CUDA_VISIBLE_DEVICES=$gpu \
        TRITON_CACHE_DIR="$cell_cache" \
        TORCHINDUCTOR_CACHE_DIR="$cell_inductor" \
        TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=ATEN \
            "$PYTHON" "$SCRIPTS/eval_finance.py" \
                --cartridge "$cart" \
                --eval-year "$y_j" --train-year "$y_i" --phase-idx "$ii" \
                --val-parquet "$val_pq" --returns-table "$RETURNS_TABLE" \
                --out "$out" --device cuda:0 \
                > "$log" 2>&1 &
        pids+=($!)
    done
    # Drop the jobs we just dispatched
    EVAL_JOBS=("${EVAL_JOBS[@]:n_to_run}")
    # Wait for the batch
    for pid in "${pids[@]}"; do
        wait $pid || fails+=("$pid")
    done
    if (( ${#fails[@]} > 0 )); then
        echo "  !! $((${#fails[@]})) eval(s) failed in batch $batch_idx; see $RUN_DIR/logs/eval_*.log"
        # don't exit — keep going so we get partial results
    fi
    batch_idx=$((batch_idx + 1))
done

# ────────────────────────────── done marker ──────────────────────────────────
date -u +%FT%TZ > "$RUN_DIR/DONE"
echo
echo "[chain] $(date -u)  done"
echo "  run dir:     $RUN_DIR"
echo "  logs:        $RUN_DIR/logs/"
echo "  predictions: $PRED_ROOT/phase{1..6}/y{2015..2020}.parquet"
echo
echo "next:  bash /home/nayan/scripts/run_finance_analysis.sh   # adds 'cartridges' to existing analysis"
