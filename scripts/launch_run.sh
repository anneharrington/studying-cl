#!/usr/bin/env bash
# Unified tmux launcher for long runs.
#
# Usage:
#   scripts/launch_run.sh <run_name> <python_command>
#
# Behavior:
#   - Appends today's date (YYYYMMDD) to <run_name> → <run_name>_<date>
#   - Substitutes the resulting name into:
#       * tmux session name
#       * --output-dir   (replaces results/<run_name> in the command)
#       * tee log path   (results/<run_name>_<date>.log)
#   - The python command must contain `--output-dir results/<run_name>`
#     literally — that token gets replaced with the dated version.
#
# Example:
#   scripts/launch_run.sh ace_anne \
#     "TOOLUSE_STRICT=1 python -u scripts/run.py \
#        --method ace --strategy sequential \
#        --model qwen-3-8b-alibaba \
#        --tasks tooluse finqa sciknoweval_bio \
#        --config configs/runs/seq_ace_paperfaith_tfb_es.yaml \
#        --train-n 400 --val-n 50 --eval-n 50 \
#        --output-dir results/ace_anne"

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <run_name> <python_command>" >&2
  echo "  python_command must contain: --output-dir results/<run_name>" >&2
  exit 1
fi

RUN_NAME_RAW="$1"
shift
PY_CMD="$*"

DATE_TAG="$(date +%Y%m%d)"
RUN_NAME="${RUN_NAME_RAW}_${DATE_TAG}"

LOG_PATH="results/${RUN_NAME}.log"
OUT_DIR="results/${RUN_NAME}"

# Replace the placeholder output_dir (results/<raw>) with the dated one.
PY_CMD_FINAL="${PY_CMD//results\/${RUN_NAME_RAW}/${OUT_DIR}}"

mkdir -p results

if tmux has-session -t "${RUN_NAME}" 2>/dev/null; then
  echo "tmux session '${RUN_NAME}' already exists — refusing to overwrite." >&2
  echo "  attach: tmux attach -t ${RUN_NAME}" >&2
  echo "  kill:   tmux kill-session -t ${RUN_NAME}" >&2
  exit 2
fi

echo "Run name:      ${RUN_NAME}"
echo "Output dir:    ${OUT_DIR}"
echo "Log file:      ${LOG_PATH}"
echo "Tmux session:  ${RUN_NAME}"
echo "Command:"
echo "  ${PY_CMD_FINAL}"
echo

tmux new-session -d -s "${RUN_NAME}" \
  "${PY_CMD_FINAL} 2>&1 | tee ${LOG_PATH}; exec bash"

echo "Launched. Attach with:  tmux attach -t ${RUN_NAME}"
echo "Tail log with:          tail -f ${LOG_PATH}"
