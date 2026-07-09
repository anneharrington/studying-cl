#!/bin/bash
# One-time setup for the cartridges-on-finance pipeline.
#
# Creates a SEPARATE venv on shared FS (visible from both nodes) because
# cartridges pins transformers>=4.49,<=4.55 + needs FlexAttention's bleeding-edge
# torch — incompatible with our verl/vllm venv at venv-sdpo-v2.
#
# Output venv:    ${CL_HOME:-/workspace/home/nayan}/venv-cartridges/   (shared MooseFS)
#
# Usage:
#   bash /home/nayan/scripts/cartridges_finance/install.sh
#
# Then before any cartridges run:
#   source ${CL_HOME:-/workspace/home/nayan}/venv-cartridges/bin/activate
#   export OPENROUTER_API_KEY=...
#   bash /home/nayan/scripts/cartridges_finance/run_chain.sh

set -euo pipefail

VENV=${CARTRIDGES_VENV:-${CL_HOME:-/workspace/home/nayan}/venv-cartridges}
CART_DIR=${CARTRIDGES_DIR:-/home/nayan/methods/cartridges}
PYBIN=${SYSTEM_PYTHON:-python3.12}

[ -d "$CART_DIR" ] || { echo "missing $CART_DIR — clone https://github.com/HazyResearch/cartridges first"; exit 1; }
command -v "$PYBIN" >/dev/null || { echo "missing $PYBIN — cartridges requires python>=3.12"; exit 1; }

# Make/refresh venv
if [ ! -d "$VENV" ]; then
    echo "[install] creating venv at $VENV (shared FS)"
    "$PYBIN" -m venv "$VENV"
else
    echo "[install] reusing existing venv at $VENV"
fi

# shellcheck source=/dev/null
source "$VENV/bin/activate"

# Fast-skip when everything is already installed correctly. Re-resolving deps
# on shared MooseFS is slow (~minutes), and uv re-checks even when nothing
# changed. Pass FORCE=1 to override.
export CARTRIDGES_DIR="$CART_DIR"
export CARTRIDGES_OUTPUT_DIR="${CARTRIDGES_OUTPUT_DIR:-${CL_HOME:-/workspace/home/nayan}/cartridges_finance/runs}"
mkdir -p "$CARTRIDGES_OUTPUT_DIR"

if [ "${FORCE:-0}" != "1" ] && python -c "
import cartridges, torch
from cartridges.models import FlexQwen3ForCausalLM
assert torch.cuda.is_available(), 'cuda unavailable'
" >/dev/null 2>&1; then
    echo "[install] cartridges + CUDA already healthy — skipping pip steps"
    echo "[install] (pass FORCE=1 to reinstall)"
    SKIP_PIP=1
else
    SKIP_PIP=0
fi

if [ "$SKIP_PIP" != "1" ]; then
    python -m pip install --upgrade pip uv >/dev/null

    echo "[install] uv pip install -e $CART_DIR"
    uv pip install -e "$CART_DIR"

    # Cartridges' default `torch` resolves to a CUDA-13 wheel, but our host driver
    # is CUDA 12.8. Force the cu128 build so the GPUs are usable.
    echo "[install] pinning torch to cu128 build (matches host driver)"
    uv pip install --reinstall --index-url https://download.pytorch.org/whl/cu128 \
        "torch==2.11.0"
fi

echo
echo "[install] sanity imports + CUDA check"
python <<'PY'
import cartridges
import cartridges.train
import cartridges.synthesize
from cartridges.models import FlexQwen3ForCausalLM
from cartridges.initialization import KVFromText
from cartridges.clients.openai import OpenAIClient
from cartridges.data.resources import TextFileResource
from cartridges.synthesizers.self_study import SelfStudySynthesizer
from cartridges.utils.wandb import WandBConfig
import torch, transformers
print("OK: cartridges imports clean")
print(f"   cartridges:    {cartridges.__file__}")
print(f"   torch:         {torch.__version__}")
print(f"   transformers:  {transformers.__version__}")
print(f"   cuda build:    {torch.version.cuda}")
print(f"   cuda avail:    {torch.cuda.is_available()}  devices={torch.cuda.device_count()}")
assert torch.cuda.is_available(), "CUDA NOT AVAILABLE — torch wheel mismatch with driver"
PY

# Verify env vars exist for downstream scripts.
# Tokasaurus venv (separate from cartridges because tokasaurus pins
# torch==2.6.0 and transformers==4.53.0 — would clobber cartridges' versions
# if installed in the same venv).
TOKA_VENV=${TOKASAURUS_VENV:-${CL_HOME:-/workspace/home/nayan}/venv-tokasaurus}
TOKA_DIR=${TOKASAURUS_DIR:-/home/nayan/methods/tokasaurus}
if [ ! -d "$TOKA_DIR" ]; then
    echo "[install] cloning tokasaurus (cartridges branch)"
    GIT_SSH_COMMAND="ssh -i /home/nayan/.ssh/id_ed25519_github -o IdentitiesOnly=yes" \
        git clone https://github.com/ScalingIntelligence/tokasaurus.git "$TOKA_DIR"
    git -C "$TOKA_DIR" checkout geoff/cartridges
fi

if [ ! -d "$TOKA_VENV" ]; then
    echo "[install] creating venv at $TOKA_VENV"
    "$PYBIN" -m venv "$TOKA_VENV"
    "$TOKA_VENV/bin/python" -m pip install --upgrade pip uv >/dev/null
fi
if ! "$TOKA_VENV/bin/python" -c "import tokasaurus" >/dev/null 2>&1; then
    echo "[install] uv pip install -e $TOKA_DIR (into $TOKA_VENV)"
    "$TOKA_VENV/bin/python" -m uv pip install -e "$TOKA_DIR"
else
    echo "[install] tokasaurus already installed in $TOKA_VENV"
fi

echo
echo "[install] env vars (set in your shell before invoking run_chain.sh):"
declare -A REQ=(
    [CARTRIDGES_DIR]="cartridges checkout (default $CART_DIR)"
    [TOKASAURUS_DIR]="tokasaurus checkout (default $TOKA_DIR)"
    [CARTRIDGES_OUTPUT_DIR]="output root (default ${CL_HOME:-/workspace/home/nayan}/cartridges_finance/runs)"
    [CARTRIDGES_VENV]="cartridges venv (default ${CL_HOME:-/workspace/home/nayan}/venv-cartridges)"
    [TOKASAURUS_VENV]="tokasaurus venv (default ${CL_HOME:-/workspace/home/nayan}/venv-tokasaurus)"
    [CARTRIDGES_TOKASAURUS_GPUS]="GPUs for tokasaurus server (default 2; uses dp_size)"
    [CARTRIDGES_WANDB_PROJECT]="wandb project (default 'sdpo_seq' — same as the verl methods)"
    [CARTRIDGES_WANDB_ENTITY]="wandb entity (default unset = personal)"
    [WANDB_API_KEY]="wandb auth (or set WANDB_MODE=offline)"
)
for v in "${!REQ[@]}"; do
    val="${!v:-}"
    if [ -n "$val" ]; then printf "  %-28s ✓ set\n" "$v"
    else printf "  %-28s   %s\n" "$v" "${REQ[$v]}"; fi
done

# Sync the venv path to node-1 so slurm jobs landing there can use it.
echo
echo "[install] $VENV is on shared MooseFS — visible from both nodes."
echo "[install] done. Next: bash run_chain.sh"
