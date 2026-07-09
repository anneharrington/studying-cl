# Cartridges-as-CL — TemporalWiki, ordering T

Mirror of `scripts/cartridges_finance/` but for the 3-slice TemporalWiki
temporal-drift benchmark. Same cartridges recipe (self-study synth +
context-distillation training), same Qwen3-8B base, same eval surface as our
existing SFT/SDPO/GRPO baselines (drift_s1/s2/s3 + stable val sets).

Source: <https://github.com/HazyResearch/cartridges>

## CL framing

```
phase 1: corpus = slice_1 (Nov→Dec)                    --synth--> Q/A --train--> cartridge_ts1
phase 2: corpus = slice_1 + slice_2 (Nov→Jan)          --synth--> Q/A --train--> cartridge_ts2
phase 3: corpus = slice_1 + slice_2 + slice_3 (Nov→Feb)--synth--> Q/A --train--> cartridge_ts3

eval at each phase: drift_s1, drift_s2, drift_s3, stable
```

## Slice metadata (from temporal_manifest.json — same as our other methods)

| Slice | Date pair | tag |
|---|---|---|
| ts1 | 2025-11-20 → 2025-12-01 | `pair_20251120_20251201` |
| ts2 | 2025-12-01 → 2026-01-01 | `pair_20251201_20260101` |
| ts3 | 2026-01-01 → 2026-02-01 | `pair_20260101_20260201` |

## Layout (planned)

```
scripts/cartridges_temporalwiki/
├── README.md              # this file
├── data_adapter.py        # per-slice Wikipedia revision text → corpus.txt + cumulative
├── synth_config.py        # SelfStudySynthesizer over the wiki diff slices
├── train_config.py        # KVFromRandomText, p=2048, lr=2e-2 (matches cartridges_finance)
├── eval_ttt.py            # per-cell eval against drift_s1/s2/s3 + stable
├── run_chain.sh           # 3-phase synth → train → eval matrix
└── run_chain.sbatch       # slurm wrapper
```

(`install.sh` shared with `cartridges_finance` — venv-cartridges + venv-tokasaurus
already cover all 3 benchmarks.)

## Data format note

TemporalWiki training data is per-slice Wikipedia revision text (article body
between two snapshot dates). Cartridges synth chunks of these articles via
SelfStudySynthesizer with the standard 4 seed prompts. At eval time, the val
parquets are open-ended fact-recall questions; we use the body-strip logic
from `cartridges_finance/eval_finance.py` so the cartridge supplies the
relevant article context.

## Inputs

Per-slice training data:

```
/workspace/home/nayan/sdpo_seq/data/temporalwiki/<slice_tag>/train.parquet
```

Val data (drift + stable):

```
/workspace/home/nayan/sdpo_seq/data/temporalwiki/<slice_tag>/{drift,stable}/val.parquet
```

## Outputs

```
/workspace/home/nayan/cartridges_temporalwiki/runs/<RUN_TAG>/
├── corpora/ts{1,2,3}/{corpus.txt, cumulative_ts{1,2,3}}
├── synth/ts{1,2,3}/<TIMESTAMP>-synth_config/.../dataset.parquet
├── train/ts{1,2,3}/<TIMESTAMP>-train_config/<UUID>/cache_last.pt
├── logs/
└── manifest.json

/workspace/home/nayan/results/temporalwiki_predictions/cartridges/ts{1,2,3}/{drift_s1,drift_s2,drift_s3,stable}.parquet
```

Run tag: `twiki-cl-<TS>-<SHA4>_cartridges_orderT_nothink_s42`

## Status

- [ ] data_adapter.py
- [ ] synth_config.py
- [ ] train_config.py
- [ ] eval_ttt.py
- [ ] run_chain.sh + sbatch

Scaffolding placeholder only. Flesh out after cartridges_finance prod completes.
