# In-Place TTT — TemporalWiki, ordering T

Mirror of `scripts/ttt_finance/` but for the 3-slice TemporalWiki temporal-drift
benchmark. Same TTT recipe, same Qwen3-8B base, same eval surface as our
existing SFT/SDPO/GRPO baselines (drift_s1/s2/s3 + stable val sets).

Source: <https://github.com/ByteDance-Seed/In-Place-TTT>

## CL framing

```
phase 1: base Qwen3-8B   --TTT pretrain on slice_1 (Nov→Dec wiki diff)--> ckpt_ts1
phase 2: ckpt_ts1        --TTT pretrain on slice_2 (Dec→Jan wiki diff)--> ckpt_ts2
phase 3: ckpt_ts2        --TTT pretrain on slice_3 (Jan→Feb wiki diff)--> ckpt_ts3

eval at each phase: drift_s1, drift_s2, drift_s3, stable
```

## Slice data (from temporal_manifest.json — same as our other methods)

| Slice | Date pair | tag |
|---|---|---|
| ts1 | 2025-11-20 → 2025-12-01 | `pair_20251120_20251201` |
| ts2 | 2025-12-01 → 2026-01-01 | `pair_20251201_20260101` |
| ts3 | 2026-01-01 → 2026-02-01 | `pair_20260101_20260201` |

## Layout (planned)

```
scripts/ttt_temporalwiki/
├── README.md
├── data_adapter.py           # slice train data → plaintext JSONL
├── configs/
│   └── qwen3_longct_twiki.yaml
├── run_chain.sh              # 3-phase pretrain → convert → eval matrix
├── run_chain.sbatch
└── eval_ttt.py               # per-cell eval against drift_s1/s2/s3 + stable
```

(`install.sh` shared with `ttt_finance` — venv-ttt covers all 3 benchmarks.)

## Inputs

Per-slice training data (same data SFT/SDPO/GRPO consumed):

```
/workspace/home/nayan/sdpo_seq/data/temporalwiki/<slice_tag>/train.parquet
```

Val data:

```
/workspace/home/nayan/sdpo_seq/data/temporalwiki/<slice_tag>/{drift,stable}/val.parquet
```

## Outputs

```
/workspace/home/nayan/ttt_temporalwiki/runs/<RUN_TAG>/
├── data/ts{1,2,3}/data.jsonl
├── train/ts{1,2,3}/{dcp,hf}/
├── logs/
└── manifest.json

/workspace/home/nayan/results/temporalwiki_predictions/ttt/ts{1,2,3}/{drift_s1,drift_s2,drift_s3,stable}.parquet
```

Run tag: `twiki-cl-<TS>-<SHA4>_ttt_orderT_nothink_s42`

## Status

- [ ] data_adapter.py
- [ ] configs/qwen3_longct_twiki.yaml
- [ ] run_chain.sh
- [ ] eval_ttt.py
- [ ] sbatch wrappers

Scaffolding placeholder only. Flesh out after cartridges_finance prod completes.
