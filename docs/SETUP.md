# Setup

The repo spans three subsystems with **incompatible dependency pins** (notably
`transformers` and `torch`). Create **one virtualenv per subsystem** — never install
them into the same environment. [uv](https://docs.astral.sh/uv/) is recommended.

## 1. Harness (prompt-based: GEPA / ACE / OpenEvolve) + shared eval

```bash
uv venv .venv-harness --python 3.11 && source .venv-harness/bin/activate
uv pip install -e .          # installs cl/ + methods/prompt_based/ from pyproject.toml
```

API keys (auto-loaded from a `.env` at repo root via python-dotenv):
```
OPENROUTER_API_KEY=sk-or-...   # qwen-3-8b model profile
PORTKEY_API_KEY=pk-...         # gemini / portkey profiles
```
The vendored ACE lives at `methods/prompt_based/ace/` and is imported directly (not
pip-installed); `ace_runner.py` puts it on `sys.path` automatically.

## 2. verl engine (offline + online weight updates: SFT / SDFT / GRPO / SDPO)

Needs NVIDIA GPU(s). Python 3.12.
```bash
uv venv .venv-verl --python 3.12 && source .venv-verl/bin/activate
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r envs/requirements-verl.txt
pip install flash-attn --no-build-isolation
export PYTHONPATH=$PWD/engine:$PYTHONPATH     # makes `verl` importable
# install a vLLM build matching your GPU (see comments in envs/requirements-verl.txt)
```
`run.sh` and `scripts/verl_training.sh` set `PYTHONPATH=engine` themselves, so the
export above is only needed for ad-hoc `python -m verl...` calls.

## 3. Cartridges (compression)

Python 3.12; needs a Tokasaurus or SGLang inference server for synthesis.
```bash
uv venv .venv-cartridges --python 3.12 && source .venv-cartridges/bin/activate
pip install -e methods/compression/cartridges
```

## 4. In-Place TTT (compression)

VeOmni-based; Python 3.11, torch 2.8 + flash-attn.
```bash
uv venv .venv-ttt --python 3.11 && source .venv-ttt/bin/activate
pip install torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e methods/compression/in_place_ttt
pip install flash-attn --no-build-isolation
```

## Data

See [TASKS.md](TASKS.md). Large data, checkpoints, and run outputs are gitignored;
populate them locally with `scripts/download_data.sh` and `data/prep/`.
