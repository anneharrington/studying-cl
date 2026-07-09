# Online weight updates — GRPO & SDPO

Family from the paper: **online weight updates**. Both run through the vendored
verl engine (`engine/verl`) via `verl.trainer.main_ppo`, selected by a Hydra config.

| Method | verl config | What it does |
|---|---|---|
| **GRPO** | `engine/verl/trainer/config/baseline_grpo.yaml` | Group-relative policy optimization. `algorithm.adv_estimator: grpo`, `policy_loss.loss_mode: vanilla` (no self-distillation). |
| **SDPO** | `engine/verl/trainer/config/sdpo.yaml`          | Self-distillation policy optimization: RL + distillation from the model's own high-reward trajectories. `policy_loss.loss_mode: sdpo`, EMA teacher (`self_distillation.*`). |

SDPO is from Hübotter et al. ([lasgroup/SDPO](https://github.com/lasgroup/SDPO)),
built on verl; that self-distillation code lives in the verl files listed in
`../offline_weight/README.md` (GRPO is verl's own estimator).

## Run (needs the verl env — see `docs/SETUP.md` and `envs/requirements-verl.txt`)

```bash
# unified entry: continual run over a task family (domain_shift|temporal_drift|discrete_updates)
./run.sh grpo domain_shift
./run.sh sdpo domain_shift

# under the hood this calls experiments/continual/run_sequential.py --method {grpo,sdpo}
# --ordering {A,F,T}; extra args pass through, e.g. ./run.sh sdpo domain_shift --seed 7
```

For a single-dataset run or custom Hydra overrides, use the generic driver directly:
`bash scripts/verl_training.sh <exp_name> {baseline_grpo,sdpo} <data_path> [overrides...]`.
Larger sweeps over orderings live in `experiments/continual/` and
`experiments/generalization/`.
