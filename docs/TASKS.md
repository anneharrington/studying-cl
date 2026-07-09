# Tasks & data

Three task families, each a sequence of phases the model must adapt to.

## Domain shift — ToolUse → FinQA → SciKE-Bio
Sequential training across distinct domains. Eval harnesses:
`cl/evals/tooluse.py`, `cl/evals/finqa.py`, `cl/evals/sciknoweval_bio.py`
(registered in `cl/tasks.py` as `tooluse`, `finqa`, `sciknoweval_bio`).
- Prompt-based: pulled by `scripts/download_data.sh` (FinQA, SciKnowEval) and built
  from SDPO splits (ToolUse, SciKE-Bio) — see comments in that script.
- verl: parquet datasets under `datasets/<task>/` (schema below).

## Temporal drift — SEC 10-K filings (2015–2020)
Predict forward stock movement from yearly filings; the distribution drifts year over
year. Eval: `cl/evals/finance_yr.py`, `cl/evals/sentiment10k.py`.
- Build verl parquets: `data/prep/download_finance_data.py` then
  `data/prep/prep_finance_yearly.py` (per-year train/val).

## Discrete updates — TemporalWiki
Wikipedia fact changes across monthly snapshots (Nov 2025 – Feb 2026); same
(subject, relation) keys, changing objects. Eval: `cl/evals/temporalwiki.py`.
- Build verl parquets: `data/prep/prep_temporalwiki_drift.py`
  (`train_s{1..4}`, `val_s{1..4}`, `val_stable`).

## verl parquet schema
Datasets consumed by the verl engine share one schema:
```
prompt        list[ {role, content} ]      conversation
system        str                          system prompt
data_source   str                          selects the reward function
ability       str                          reward style (exact-match, mcq, …)
reward_model  { style, ground_truth }      gold answer(s)
extra_info    dict                         split / index / metadata
```

## Notes
- Large data and built parquets are gitignored (`data/`, `datasets/`, `*.parquet`).
  Keep `data/prep/` (the builders) in git; regenerate data locally.
- Compression methods consume per-phase corpora derived from the same finance /
  temporalwiki data via the `experiments/{cartridges,ttt}_*/data_adapter.py` scripts.
