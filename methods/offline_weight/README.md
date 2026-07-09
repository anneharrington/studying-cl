# Offline weight updates — SFT & SDFT

Family from the paper: **offline weight updates**. Both methods run through the
vendored verl engine (`engine/verl`) via `verl.trainer.main_ppo`, selected by a
Hydra config. They differ only in the loss/teacher configuration.

| Method | verl config | What it does |
|---|---|---|
| **SFT**  | `engine/verl/trainer/config/sft.yaml`  | Supervised fine-tuning on dataset gold (one-hot teacher; `self_distillation.teacher_regularization: gold`, `rollout.use_gold_response: true`). Reduces to standard cross-entropy NLL. |
| **SDFT** | `engine/verl/trainer/config/sdft.yaml` | Self-distillation fine-tuning (forward-KL, `self_distillation.alpha: 0.0`, demo-based teacher; `rollout.n: 1`). Follows Shenfeld et al. |

The SDFT/SDPO self-distillation implementations come from the
[Self-Distillation](https://github.com/idanshen/Self-Distillation) (Shenfeld et al.)
and [SDPO](https://github.com/lasgroup/SDPO) (Hübotter et al.) repos, built on verl;
in `engine/verl/` that code lives in:
`engine/verl/trainer/ppo/core_algos.py` (`compute_self_distillation_loss`),
`engine/verl/workers/actor/dp_actor.py`, `engine/verl/workers/config/{actor,rollout}.py`,
and `engine/verl/trainer/ppo/ray_trainer.py` (`_substitute_gold_responses`).

## Run (needs the verl env — see `docs/SETUP.md` and `envs/requirements-verl.txt`)

```bash
# unified entry: continual run over a task family (domain_shift|temporal_drift|discrete_updates)
./run.sh sft  domain_shift
./run.sh sdft domain_shift

# under the hood this calls experiments/continual/run_sequential.py --method {sft,sdft}
# --ordering {A,F,T}; extra args pass through, e.g. ./run.sh sdft domain_shift --seed 7
```

For a single-dataset run or custom Hydra overrides, use the generic driver directly:
`bash scripts/verl_training.sh <exp_name> {sft,sdft} <data_path> [overrides...]`
(`<data_path>` is a verl-format parquet — see `data/prep/`).
