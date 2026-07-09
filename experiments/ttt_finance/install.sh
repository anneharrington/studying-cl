#!/bin/bash
# Install venv-ttt for the In-Place TTT method on shared FS.
#
# Pin guidance (from In-Place-TTT README):
#   torch 2.8.0+cu128, flash-attn 2.8.3, transformers 4.57.3,
#   veomni @ git+https://github.com/ByteDance-Seed/VeOmni.git@9b91e164b...
#   liger-kernel, byted-wandb, opt_einsum, einops
#
# Critical isolation guarantees:
#   - venv lives at $VENV (default ${CL_HOME:-/workspace/home/nayan}/venv-ttt) — separate
#     from venv-cartridges, venv-tokasaurus, venv-sdpo, venv-sdpo-v2.
#   - python explicitly 3.11 to match upstream flash-attn cp311 wheel; existing
#     venvs are all 3.12 so no version clash.
#   - In-Place-TTT clone goes to ${CL_HOME:-/workspace/home/nayan}/methods/In-Place-TTT
#     (NOT /home/nayan/methods/, which is overlay-backed and small).
#   - PIP_CACHE_DIR + TMPDIR pointed at /workspace so we never fill /tmp.
#
# Run with FORCE=1 to override the fast-skip guard.
set -euo pipefail

: "${VENV:=${CL_HOME:-/workspace/home/nayan}/venv-ttt}"
: "${METHODS_DIR:=${CL_HOME:-/workspace/home/nayan}/methods}"
: "${IPT_REPO:=$METHODS_DIR/In-Place-TTT}"
: "${PYTHON_BIN:=/usr/bin/python3.11}"

# Route all heavy I/O to /workspace (MooseFS, ~213 TB free) — never to overlay
# or /tmp which can fill quickly.
export PIP_CACHE_DIR="${CL_HOME:-/workspace/home/nayan}/.cache/pip"
export TMPDIR="${CL_HOME:-/workspace/home/nayan}/.tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

if [[ -x "$VENV/bin/python" && "${FORCE:-0}" != "1" ]]; then
    echo "[install] $VENV exists; skipping. Run with FORCE=1 to recreate."
    exit 0
fi

# Sanity — python 3.11 must exist
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "!! $PYTHON_BIN not found. Install python3.11 first or override PYTHON_BIN."
    exit 2
fi

mkdir -p "$METHODS_DIR"

# Step 0 — clone In-Place-TTT to shared FS (so both nodes see it)
if [[ ! -d "$IPT_REPO/.git" ]]; then
    echo "[install] cloning In-Place-TTT to $IPT_REPO"
    git clone https://github.com/ByteDance-Seed/In-Place-TTT.git "$IPT_REPO"
else
    echo "[install] In-Place-TTT clone already at $IPT_REPO"
fi

# Step 1 — create venv with python 3.11 explicitly
echo "[install] creating venv at $VENV (using $PYTHON_BIN)"
"$PYTHON_BIN" -m venv "$VENV"
# shellcheck source=/dev/null
source "$VENV/bin/activate"
pip install --upgrade pip wheel

# Step 2 — torch 2.8.0+cu128 (per upstream README)
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Step 3 — flash-attn 2.8.3 cp311 wheel (matches our venv-ttt python 3.11).
# pip needs the actual wheel filename (not a generic "fa.whl") so we keep the
# original name when downloading.
FA_WHL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
FA_WHL_NAME=$(basename "$FA_WHL_URL")
WHL="$TMPDIR/$FA_WHL_NAME"
if wget --tries=3 --timeout=60 -O "$WHL" "$FA_WHL_URL" 2>/dev/null && [[ -s "$WHL" ]]; then
    pip install "$WHL"
    rm -f "$WHL"
else
    echo "[install] flash-attn wheel download failed; falling back to source build (slow, ~10-20 min)"
    pip install flash-attn==2.8.3 --no-build-isolation
fi

# Step 4 — VeOmni at the validated commit
pip install "veomni @ git+https://github.com/ByteDance-Seed/VeOmni.git@9b91e164bea9e17f17ed490aab5e076c2335ca25"

# Step 5 — remaining deps
pip install liger-kernel torchdata blobfile datasets diffusers tiktoken timm
pip install transformers==4.57.3
pip install opt_einsum einops

# byted-wandb (their wandb wrapper) replaces vanilla wandb
pip uninstall -y byted-wandb wandb || true
pip install byted-wandb

# Step 6 — install In-Place-TTT in editable mode (pulls inference_model + eval_config helpers)
pip install -e "$IPT_REPO"

# Step 7 — sanity imports
python -c "
import sys
print('python:', sys.version.split()[0])
import torch; print('torch:', torch.__version__, '  cuda:', torch.version.cuda)
import flash_attn; print('flash-attn:', flash_attn.__version__)
import veomni; print('veomni: imported')
import transformers; print('transformers:', transformers.__version__)
"

echo
echo "[install] done."
echo "  venv:    $VENV"
echo "  repo:    $IPT_REPO"
echo "  python:  $($VENV/bin/python --version)"
