# Experiments — continual-learning sweeps & run-chains

These orchestrate the paper's methods over the three task families. They are
**reference / template scripts** from the original SLURM cluster runs: the data and
output locations are not in git, so treat them as recipes to adapt, not turnkey jobs.

## Layout

| Dir | Methods | Task family |
|---|---|---|
| `continual/` | SFT, SDFT, GRPO, SDPO (verl) | domain shift orderings + finance; see `sweep.sh`, `sweep_finance.sh`, `sweep_sft.sh`, `run_sequential.py` |
| `generalization/` | GRPO, SDPO (verl) | per-task domain-shift sweeps (`run_baseline_grpo_all.sh`, `run_sdpo_all.sh`) |
| `rich_feedback/` | GRPO, SDPO (verl) | reward-model feedback variant |
| `cartridges_domainshift/`, `cartridges_finance/`, `cartridges_temporalwiki/` | Cartridges | domain shift / temporal drift / discrete updates |
| `ttt_domainshift/`, `ttt_finance/`, `ttt_temporalwiki/`, `ttt/` | In-Place TTT | domain shift / temporal drift / discrete updates |

Each `*_finance` / `*_temporalwiki` dir has the same shape: `data_adapter.py`
(parquet → method input), `synth_config.py`/`train_config.py` or `eval_*.py`, and a
`run_chain.sh` (+ `.sbatch`) that chains phases.

## Paths

The shell/sbatch launchers reference host paths via an overridable env var:

```bash
export CL_HOME=/your/data/root   # defaults to /workspace/home/nayan if unset
```

The Python config/adapter files (`data_adapter.py`, `synth_config.py`,
`train_config.py`, `eval_*.py`, `run_sequential.py`) still contain the original
author's absolute paths — edit the path constants near the top of each before running.
Run `grep -rl /workspace/home experiments --include='*.py'` to find them.

Which environment each needs (see `../docs/SETUP.md`): `continual/`,
`generalization/`, `rich_feedback/` → verl env; `cartridges_*` → cartridges env;
`ttt_*` → ttt env.
