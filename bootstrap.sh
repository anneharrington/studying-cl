#!/usr/bin/env bash
# Bootstrap studying-cl for a fresh machine.
#
# Creates one venv per subsystem (harness / verl / cartridges / ttt),
# installs deps from envs/requirements-*.txt, validates required env vars,
# and prints a short next-steps message. Idempotent — re-runs skip existing
# venvs and only top up missing pieces.
#
# Usage:
#   ./bootstrap.sh             # all subsystems
#   ./bootstrap.sh harness     # one subsystem
#   ./bootstrap.sh harness verl
#
# Required env vars (validated, not set):
#   WANDB_API_KEY    weight-update methods (verl) — get at https://wandb.ai/authorize
#   OPENROUTER_API_KEY   prompt-based + cartridges synthesis  — https://openrouter.ai
# Optional:
#   PORTKEY_API_KEY      alternate LLM provider for prompt methods
#   CL_HOME              prefix for venvs / cached data (default: $HOME)

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

# ── color helpers ───────────────────────────────────────────────────────────
_g() { printf '\033[32m%s\033[0m\n' "$*"; }
_y() { printf '\033[33m%s\033[0m\n' "$*"; }
_r() { printf '\033[31m%s\033[0m\n' "$*"; }

# ── pick uv if present, else fall back to python -m venv ───────────────────
have_uv=0; command -v uv >/dev/null 2>&1 && have_uv=1
make_venv() {  # make_venv <path> <python-version>
    local target="$1" py="$2"
    if [[ -f "$target/bin/activate" ]]; then
        _y "  $target already exists — skipping create"
        return 0
    fi
    if (( have_uv )); then
        uv venv "$target" --python "$py"
    else
        "python${py}" -m venv "$target"
    fi
}
pip_install() {  # pip_install <venv> <args...>
    local v="$1"; shift
    if (( have_uv )); then
        uv pip install --python "$v/bin/python" "$@"
    else
        "$v/bin/pip" install "$@"
    fi
}

# ── per-subsystem installers ───────────────────────────────────────────────
setup_harness() {  # prompt-based + cl/ shared eval lib
    _g ">> harness (prompt-based + shared eval lib)"
    local v="$REPO/.venv-harness"
    make_venv "$v" 3.11
    pip_install "$v" -e .
    pip_install "$v" -r envs/requirements-harness.txt
    _g "   ok"
}

setup_verl() {  # offline + online weight updates (SFT / SDFT / GRPO / SDPO)
    _g ">> verl (weight-update methods; needs NVIDIA GPU)"
    local v="$REPO/.venv-verl"
    make_venv "$v" 3.12
    pip_install "$v" torch==2.5.1 torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu124
    pip_install "$v" -r envs/requirements-verl.txt
    pip_install "$v" flash-attn --no-build-isolation
    # vLLM is required for verl rollouts (GRPO/SDPO/online weight updates).
    # 0.8.4 is the GH200/Hopper baseline used in the SDPO experiments; override
    # with `VERL_VLLM=vllm==0.12.0` etc. for Blackwell or older builds for Ampere.
    pip_install "$v" "${VERL_VLLM:-vllm==0.8.4}" || \
        _y "   NOTE: vLLM install failed (likely GPU/CUDA mismatch). Install manually:" && \
        _y "         $v/bin/pip install vllm==<your-version>   (see envs/requirements-verl.txt header)"
    _g "   ok"
}

setup_cartridges() {  # compression: Cartridges
    _g ">> cartridges (compression; needs Tokasaurus or SGLang for synthesis)"
    local v="$REPO/.venv-cartridges"
    make_venv "$v" 3.12
    pip_install "$v" -e methods/compression/cartridges
    _g "   ok"
}

setup_ttt() {  # compression: In-Place TTT
    _g ">> ttt (In-Place TTT compression; torch 2.8 + flash-attn)"
    local v="$REPO/.venv-ttt"
    make_venv "$v" 3.11
    pip_install "$v" torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu128
    pip_install "$v" -e methods/compression/in_place_ttt
    pip_install "$v" flash-attn --no-build-isolation
    _g "   ok"
}

# ── env-var validator (warns, doesn't fail) ────────────────────────────────
validate_env() {
    _g ">> env-var check"
    local need_msg=0
    for var in WANDB_API_KEY OPENROUTER_API_KEY; do
        if [[ -z "${!var:-}" ]]; then
            _y "  $var is unset (needed for the relevant subsystem)"
            need_msg=1
        else
            _g "  $var: set"
        fi
    done
    if (( need_msg )); then
        cat <<'EOF'

   Set missing keys in your shell, e.g.:
     export WANDB_API_KEY=...        # https://wandb.ai/authorize
     export OPENROUTER_API_KEY=...   # https://openrouter.ai/keys
   Or add them to a .env file at the repo root (python-dotenv auto-loads).
EOF
    fi
}

# ── data check ─────────────────────────────────────────────────────────────
# Paths intentionally match where the actual consumers look:
#   data/sciknoweval/raw_data/Chemistry/L3/  — cl/evals/sciknoweval.py
#   data/parquet/{finqa,tooluse}/            — configs/tasks/{finqa,tooluse}.yaml
#   $CL_FINANCE_DATA/                        — experiments/continual/run_sequential.py
#   $CL_TEMPORAL_DRIFT_DATA/                 — experiments/continual/run_sequential.py
check_data() {
    _g ">> data check (domain_shift datasets, parquet-format)"
    local missing=()
    [[ -d data/sciknoweval ]]       || missing+=("data/sciknoweval/   (run: bash scripts/download_data.sh)")
    [[ -f data/parquet/finqa/train.parquet ]]  || missing+=("data/parquet/finqa/{train,val}.parquet   (parquet build not yet in data/prep/ — see docs/TASKS.md)")
    [[ -f data/parquet/tooluse/train.parquet ]] || missing+=("data/parquet/tooluse/{train,val}.parquet   (parquet build not yet in data/prep/ — see docs/TASKS.md)")
    if (( ${#missing[@]} > 0 )); then
        _y "  missing dataset paths (needed for ./run.sh <method> domain_shift):"
        for d in "${missing[@]}"; do _y "    - $d"; done
    else
        _g "  domain_shift datasets present"
    fi
    _g ">> data check (temporal_drift + discrete_updates env vars)"
    [[ -n "${CL_FINANCE_DATA:-}"        ]] && _g "  CL_FINANCE_DATA=$CL_FINANCE_DATA" \
                                          || _y "  CL_FINANCE_DATA unset (needed for temporal_drift; build with data/prep/prep_finance_yearly.py)"
    [[ -n "${CL_TEMPORAL_DRIFT_DATA:-}" ]] && _g "  CL_TEMPORAL_DRIFT_DATA=$CL_TEMPORAL_DRIFT_DATA" \
                                          || _y "  CL_TEMPORAL_DRIFT_DATA unset (needed for discrete_updates; build with data/prep/prep_temporalwiki_drift.py)"
}

# ── main dispatch ──────────────────────────────────────────────────────────
all="harness verl cartridges ttt"
targets=("$@")
[[ ${#targets[@]} -eq 0 ]] && targets=($all)

for t in "${targets[@]}"; do
    case "$t" in
        harness)    setup_harness ;;
        verl)       setup_verl ;;
        cartridges) setup_cartridges ;;
        ttt)        setup_ttt ;;
        *)          _r "unknown subsystem: $t (choices: $all)"; exit 1 ;;
    esac
done

validate_env
check_data

_g ""
_g "bootstrap complete. Next:"
_g "  ./run.sh sft domain_shift              # weight-update smoke"
_g "  ./run.sh gepa domain_shift             # prompt-based smoke"
_g "  ./run.sh cartridges discrete_updates   # compression smoke"
_g "  make smoke                              # all-subsystem smoke (~30 min on 1 GPU)"
