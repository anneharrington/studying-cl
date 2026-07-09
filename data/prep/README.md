# Continual-learning dataset prep

Two task families used by `experiments/continual/run_sequential.py`:

| Task family | Ordering | Phases | Source |
|---|---|---|---|
| Finance 10-K forward sentiment | `F` | 6 (years 2015..2020) | [`JanosAudran/financial-reports-sec`](https://huggingface.co/datasets/JanosAudran/financial-reports-sec) |
| TemporalWiki monthly drift     | `T` | 3 chronological slices | [TemporalWiki probe pipeline](https://github.com/joeljang/temporalwikidatasets) |

Both prep scripts emit verl-PPO-format parquets:
```
columns = ['prompt', 'embedding', 'system', 'data_source', 'ability',
           'reward_model', 'extra_info']
```
where `prompt` is a `[{role,content}, ...]` chat array and the gold answer
lives at `reward_model.ground_truth`.

## Finance

```bash
# 1. Download the HF shards (~11 GB cached)
python scripts/data/download_finance_data.py

# 2. Build per-year parquets (writes to $CL_FINANCE_DATA, default ./data/finance_yearly)
python scripts/data/prep_finance_yearly.py
```

Output: `train_y{2015..2020}.parquet` (500 ex/yr), `val_y{2015..2020}.parquet`
(50 ex/yr), `manifest.json`.

## TemporalWiki

The probe CSVs (`changed.csv`, `unchanged.csv` per `pair_<old>_<new>` slice
dir) are the output of the upstream TemporalWiki probe pipeline — building
them requires Wikipedia/Wikidata snapshots and is out-of-scope for this repo.

Once the probe dir exists:
```bash
# Builds per-slice parquets (writes to $CL_TEMPORAL_DRIFT_DATA, default ./data/temporalwiki_drift)
TEMPORALWIKI_PROBES=/path/to/probes \
    python scripts/data/prep_temporalwiki_drift.py
```

Output: `train_s{1..3}.parquet`, `val_s{1..3}.parquet`, `val_stable.parquet`,
`manifest.json`.

## Pointing the trainer at the data

```bash
export CL_FINANCE_DATA=/path/to/finance_yearly
export CL_TEMPORAL_DRIFT_DATA=/path/to/temporalwiki_drift
sbatch experiments/continual/run_sequential.sbatch sft F 42       # finance
sbatch experiments/continual/run_sequential.sbatch sft T 42       # temporalwiki
```
