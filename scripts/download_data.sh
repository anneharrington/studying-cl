#!/usr/bin/env bash
# Download the raw evaluation datasets used by configs/tasks/*.yaml and the
# prompt-based eval harness. Writes to ./data/ (matches what the configs read).
#
# Usage:   bash scripts/download_data.sh
#
# What this WILL fetch:
#   - data/sciknoweval/                       HF clone (consumed by cl/evals/sciknoweval.py)
#   - data/finqa_test.json                    raw FinQA test split (GitHub)
#   - data/gsm8k_test.jsonl                   GSM8K test split (HF)
#   - data/livebench_math.jsonl               LiveBench math split (HF)
#   - data/toolalpaca_eval_simulated.json     ToolAlpaca eval (GitHub)
#
# What's NOT yet covered (needs a parquet conversion that lives in data/prep/):
#   - data/parquet/finqa/{train,val}.parquet  expected by configs/tasks/finqa.yaml
#   - data/parquet/tooluse/{train,val}.parquet expected by configs/tasks/tooluse.yaml
# If you're running the weight-update domain_shift family, build those parquets
# from FinQA + ToolAlpaca first; see docs/TASKS.md.
#
# Requires: curl, git-lfs (auto-installed via apt/brew if missing), python3 with
# `datasets` installed (covered by envs/requirements-harness.txt).

set -euo pipefail

DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)/data"
mkdir -p "$DATA_DIR"

echo "Downloading raw eval datasets to $DATA_DIR ..."

# GSM8K (test split) — HF datasets
GSM8K_FILE="$DATA_DIR/gsm8k_test.jsonl"
if [ -f "$GSM8K_FILE" ]; then
  echo "  GSM8K already downloaded, skipping"
else
  echo "  GSM8K test split (openai/gsm8k)"
  python3 -c "
from datasets import load_dataset
import json
ds = load_dataset('openai/gsm8k', 'main', split='test')
with open('$GSM8K_FILE', 'w') as f:
    for ex in ds:
        f.write(json.dumps(ex) + '\n')
print(f'    Downloaded {len(ds)} examples to $GSM8K_FILE')
"
fi

# LiveBench Math — HF datasets
LIVEBENCH_FILE="$DATA_DIR/livebench_math.jsonl"
if [ -f "$LIVEBENCH_FILE" ]; then
  echo "  LiveBench Math already downloaded, skipping"
else
  echo "  LiveBench Math (livebench/math)"
  python3 -c "
from datasets import load_dataset
import json
ds = load_dataset('livebench/math', split='test')
with open('$LIVEBENCH_FILE', 'w') as f:
    for ex in ds:
        row = {k: v for k, v in ex.items() if k != 'release_date'}
        if 'livebench_release_date' in row and row['livebench_release_date']:
            row['livebench_release_date'] = str(row['livebench_release_date'])
        f.write(json.dumps(row) + '\n')
print(f'    Downloaded {len(ds)} examples to $LIVEBENCH_FILE')
"
fi

# ToolAlpaca (eval_simulated) — GitHub
TOOLALPACA_FILE="$DATA_DIR/toolalpaca_eval_simulated.json"
if [ -f "$TOOLALPACA_FILE" ]; then
  echo "  ToolAlpaca already downloaded, skipping"
else
  echo "  ToolAlpaca (tangqiaoyu/ToolAlpaca)"
  curl -fSL -o "$TOOLALPACA_FILE" \
    "https://raw.githubusercontent.com/tangqiaoyu/ToolAlpaca/main/data/toolalpaca_eval_simulated.json"
fi

# FinQA (test split) — GitHub
FINQA_FILE="$DATA_DIR/finqa_test.json"
if [ -f "$FINQA_FILE" ]; then
  echo "  FinQA test already downloaded, skipping"
else
  echo "  FinQA test (czyssrs/FinQA)"
  curl -fSL -o "$FINQA_FILE" \
    "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/test.json"
fi

# SciKnowEval — HF clone via git-lfs (matches the layout cl/evals/sciknoweval.py expects)
SCIKNOWEVAL_DIR="$DATA_DIR/sciknoweval"
if [ -d "$SCIKNOWEVAL_DIR" ]; then
  echo "  SciKnowEval already downloaded, skipping"
else
  echo "  SciKnowEval (hicai-zju/SciKnowEval)"
  if ! command -v git-lfs &> /dev/null; then
    echo "    git-lfs not found, installing..."
    brew install git-lfs 2>/dev/null || apt-get install -y git-lfs 2>/dev/null || {
      echo "ERROR: Please install git-lfs manually"; exit 1;
    }
    git lfs install
  fi
  git clone https://huggingface.co/datasets/hicai-zju/SciKnowEval "$SCIKNOWEVAL_DIR"
fi

echo ""
echo "Done. Contents of $DATA_DIR:"
ls -lh "$DATA_DIR" 2>&1 | head -20

cat <<'EOF'

Next steps:
  - Run ./bootstrap.sh   to verify all required dataset paths.
  - For weight-update domain_shift, you also need parquets at
    data/parquet/{finqa,tooluse}/ — see docs/TASKS.md for the conversion.
  - For temporal_drift and discrete_updates, build the per-year / per-slice
    parquets with data/prep/prep_finance_yearly.py and
    data/prep/prep_temporalwiki_drift.py respectively.
EOF
