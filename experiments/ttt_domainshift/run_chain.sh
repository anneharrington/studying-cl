#!/bin/bash
# In-Place TTT on the DOMAIN-SHIFT sequence (ToolUse -> FinQA -> SciKE-Bio).
#
# Sequential continual-pretrain: phase i starts from phase (i-1)'s HF checkpoint
# (phase 1 from BASE_MODEL), TTT-pretrains on task t_i's text, converts the DCP
# checkpoint to HF, then evaluates on every task's val split via the shared cl/
# eval harness.
#
# Mirrors experiments/ttt_finance/run_chain.sh but is parameterized by TASKS and
# routes eval through scripts/eval_prompt.py. TEMPLATE: needs the ttt venv
# (envs/requirements-ttt.txt), a GPU, and the corpora built by the cartridges
# domain-shift adapter. Resume-safe (skips phases whose hf ckpt exists).
#
# Required env: WANDB_API_KEY (or CARTRIDGES_WANDB_DISABLE=1)
# Optional:     TASKS, RUN_DIR, MAX_STEPS_PER_PHASE, SKIP_TRAIN, SKIP_EVAL, EVAL_MODEL
set -euo pipefail

SCRIPTS=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPTS/../.." && pwd)

: "${TTT_VENV:=${CL_HOME:-$HOME}/venv-ttt}"
: "${IPT_REPO:=$REPO_ROOT/methods/compression/in_place_ttt}"
: "${BASE_MODEL:=Qwen/Qwen3-8B}"
: "${BASE_MODEL_DIR:=${HF_HOME:-$HOME/.cache/huggingface}/hub/models--Qwen--Qwen3-8B}"
: "${MAX_STEPS_PER_PHASE:=500}"
export PYTHONPATH="$IPT_REPO:${PYTHONPATH:-}"
[ -f "$TTT_VENV/bin/activate" ] && source "$TTT_VENV/bin/activate" || \
    echo "!! ttt venv missing at $TTT_VENV (see install.sh / docs/SETUP.md)"

TASKS=${TASKS:-"tooluse finqa sciknoweval_bio"}
read -r -a TASKS_ARR <<< "$TASKS"
RUN_DIR=${RUN_DIR:-${CL_HOME:-$HOME}/ttt_domainshift/runs/domainshift_ttt_s42}
EVAL_MODEL=${EVAL_MODEL:-qwen-3-8b}

TRAIN_SCRIPT="$IPT_REPO/tasks/train_torch.py"
CONVERT_SCRIPT="$IPT_REPO/scripts/merge_dcp_to_hf.py"
BASE_CONFIG="$SCRIPTS/configs/qwen3_longct_domainshift.yaml"
mkdir -p "$RUN_DIR"/{data,train,logs}

# ── 0. Build VeOmni jsonl from the domain-shift corpora ────────────────────────
# (corpora come from the cartridges domain-shift adapter — build them first.)
CORPORA=${CORPORA:-${CL_HOME:-$HOME}/cartridges_domainshift/runs/domainshift_cartridges_s42/corpora}
python "$SCRIPTS/data_adapter.py" --src "$CORPORA" --out "$RUN_DIR/data" \
    --tasks "${TASKS_ARR[@]}" > "$RUN_DIR/logs/data.log" 2>&1
tail -n 8 "$RUN_DIR/logs/data.log" | sed 's/^/  /'

# ── 1..3. Sequential continual pretraining ─────────────────────────────────────
prev_hf="$BASE_MODEL"
phase=0
for task in "${TASKS_ARR[@]}"; do
    phase=$((phase + 1))
    train_data="$RUN_DIR/data/cumulative_p${phase}/data.jsonl"   # union(task_1..task_i)
    out_dcp="$RUN_DIR/train/p${phase}/dcp"
    out_hf="$RUN_DIR/train/p${phase}/hf"
    echo "[chain] phase $phase ($task)  from=$prev_hf  data=$train_data"

    if [[ "${SKIP_TRAIN:-0}" != "1" && ! -d "$out_hf" ]]; then
        bash "$IPT_REPO/train.sh" "$TRAIN_SCRIPT" "$BASE_CONFIG" \
            --model.model_path "$prev_hf" \
            --data.train_path "$train_data" \
            --train.output_dir "$out_dcp" \
            --train.max_steps "$MAX_STEPS_PER_PHASE" \
            --train.wandb_name "domainshift_ttt_p${phase}_${task}" \
            > "$RUN_DIR/logs/train_p${phase}.log" 2>&1
        python "$CONVERT_SCRIPT" --load-dir "$out_dcp" --save-dir "$out_hf" \
            --model-assets-dir "$BASE_MODEL_DIR" \
            > "$RUN_DIR/logs/convert_p${phase}.log" 2>&1
    fi
    prev_hf="$out_hf"

    # eval the converted HF checkpoint on every task via the shared harness
    # (serve $out_hf with vLLM and point EVAL_MODEL's profile at it).
    if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
        for et in "${TASKS_ARR[@]}"; do
            "$REPO_ROOT/.venv-harness/bin/python" "$REPO_ROOT/scripts/eval_prompt.py" \
                --task "$et" --model "$EVAL_MODEL" --prompt-text "{question}" \
                > "$RUN_DIR/logs/eval_p${phase}_${et}.log" 2>&1 || \
                echo "  (eval p$phase/$et needs the served TTT checkpoint; see README)"
        done
    fi
done

touch "$RUN_DIR/DONE"
echo "[chain] done -> $RUN_DIR"
