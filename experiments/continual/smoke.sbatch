#!/bin/bash
#SBATCH --job-name=sdpo_smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=${CL_HOME:-/workspace/home/nayan}/sdpo_seq/logs/slurm_smoke_%j.out
#SBATCH --error=${CL_HOME:-/workspace/home/nayan}/sdpo_seq/logs/slurm_smoke_%j.err
#
# Minimal end-to-end smoke. One GRPO training step on bio, with val_before_train
# exercising all three data sources (bio/finqa/tooluse) so we validate per-source metric
# aggregation. Also writes an hf_model checkpoint so we verify the task-handoff path that
# run_sequential.py depends on.
#
# Uses --config-name=sdpo (same config path as the real sweep) with loss_mode=vanilla to
# neutralize self-distillation -- this keeps the smoke on the identical Hydra composition
# path as the real runs, so any config issues surface here, not in the first sweep job.

set -euo pipefail

SDPO_ROOT=${CL_HOME:-/workspace/home/nayan}/SDPO
SMOKE_DIR=${CL_HOME:-/workspace/home/nayan}/sdpo_seq/smoke
mkdir -p "$SMOKE_DIR/metrics" "$SMOKE_DIR/ckpts"

source ${CL_HOME:-/workspace/home/nayan}/sdpo_seq/env.sh
source ${CL_HOME:-/workspace/home/nayan}/venv-sdpo/bin/activate
cd "$SDPO_ROOT"

# Slurm sets ROCR_VISIBLE_DEVICES alongside CUDA_VISIBLE_DEVICES inside the allocation;
# verl.single_controller's _setup_env_cuda_visible_devices rejects the pair. We only use
# NVIDIA GPUs, so drop the AMD/HIP var before Ray spawns workers.
unset ROCR_VISIBLE_DEVICES

# Verl env-ref safeties (same defaults run_sequential.py injects)
export USER=${USER:-nayan}
export EXPERIMENT=smoke
export TASK=bio
export VERL_FILE_LOGGER_PATH="$SMOKE_DIR/metrics/smoke.jsonl"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

echo "[smoke] $(date -u)  starting verl main_ppo"

python -m verl.trainer.main_ppo \
    --config-name=sdpo \
    "data.train_files=[${SDPO_ROOT}/datasets/sciknoweval/biology/train.parquet]" \
    "data.val_files=[${SDPO_ROOT}/datasets/sciknoweval/biology/test.parquet,${SDPO_ROOT}/datasets/finqa/test.parquet,${SDPO_ROOT}/datasets/tooluse/test.parquet]" \
    data.train_batch_size=8 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.apply_chat_template_kwargs.enable_thinking=false \
    data.shuffle=true \
    max_model_len=2048 \
    actor_rollout_ref.model.path=Qwen/Qwen3-8B \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
    "actor_rollout_ref.actor.checkpoint.save_contents=[model,hf_model,extra]" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.max_model_len=2048 \
    actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
    actor_rollout_ref.rollout.val_kwargs.n=2 \
    critic.model.path=Qwen/Qwen3-8B \
    "custom_reward_function.path=${SDPO_ROOT}/engine/verl/utils/reward_score/feedback/__init__.py" \
    trainer.project_name=sdpo_seq \
    trainer.experiment_name=smoke \
    trainer.group_name=smoke \
    trainer.default_local_dir="$SMOKE_DIR/ckpts" \
    trainer.total_training_steps=1 \
    trainer.total_epochs=999 \
    trainer.test_freq=1 \
    trainer.save_freq=1 \
    trainer.val_before_train=true \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.max_actor_ckpt_to_keep=1 \
    "trainer.logger=[wandb,console,file]" \
    vars.dir="$SDPO_ROOT" \
    vars.task=bio \
    vars.ckpt_dir="$SMOKE_DIR" \
    vars.log_dir="$SMOKE_DIR/logs" \
    +trainer.seed=42

echo "[smoke] $(date -u)  verl exited cleanly"

# Verify the 4 contract checkpoints the real runs depend on:
echo "[smoke] === post-run verification ==="
python - <<'PY'
import json, pathlib, sys
smoke = pathlib.Path("${CL_HOME:-/workspace/home/nayan}/sdpo_seq/smoke")
ok = True

# 1. hf_model checkpoint written (needed for task-to-task handoff in run_sequential)
ckpts = sorted(smoke.glob("ckpts/global_step_*/actor/huggingface"))
if not ckpts:
    print("FAIL: no hf_model checkpoint under ckpts/global_step_*/actor/huggingface")
    ok = False
else:
    hf_path = ckpts[-1]
    has_weights = any(hf_path.glob("*.safetensors")) or any(hf_path.glob("pytorch_model*.bin"))
    has_config = (hf_path / "config.json").exists()
    print(f"OK hf_model: {hf_path}  (weights={has_weights} config={has_config})")
    ok &= has_weights and has_config

# 2. FileLogger JSONL with val-core/<source>/acc entries for all 3 sources
jsonl = smoke / "metrics" / "smoke.jsonl"
if not jsonl.exists() or jsonl.stat().st_size == 0:
    print(f"FAIL: {jsonl} missing or empty")
    ok = False
else:
    sources = set()
    for line in jsonl.read_bytes().splitlines():
        if not line.strip(): continue
        evt = json.loads(line)
        for k in evt.get("data", {}):
            if k.startswith("val-core/"):
                sources.add(k.split("/")[1])
    expected = {"sciknoweval", "finqa", "tooluse"}
    if not expected.issubset(sources):
        print(f"FAIL: metric sources {sources} missing at least one of {expected}")
        ok = False
    else:
        print(f"OK FileLogger: per-source metrics for {sources}")

sys.exit(0 if ok else 1)
PY
echo "[smoke] $(date -u)  done"
