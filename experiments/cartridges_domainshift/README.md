# Cartridges on the domain-shift sequence (ToolUse → FinQA → SciKE-Bio)

Wires the Cartridges compression method into the **domain-shift** task family, so
all four method families cover all three task families.

A phase = a task (not a drifting corpus). Each phase: build a text corpus from the
task's verl train parquet → synthesize self-study Q/A → context-distill a cartridge
→ evaluate the composed cartridge on every task's val split, scored through the
shared `cl/` eval harnesses.

## Files
- `data_adapter.py` — `datasets/<task>/train.parquet` → per-task + cumulative `corpus.txt`.
- `run_chain.sh` — drives data_adapter → synth → train → eval over `TASKS`.
- `synth_config.py`, `train_config.py` — the cartridges synth/train launchers
  (env-driven; reused as-is — `CARTRIDGES_FINANCE_CORPUS_FILE`/`_PHASE`/`_YEAR`
  select the corpus and phase, so their finance-flavored names are cosmetic here).
- `install.sh` — sets up the cartridges venv.

## Run (cartridges env + GPU + Tokasaurus server; see ../../docs/SETUP.md)
```bash
export CL_HOME=/your/scratch
TASKS="tooluse finqa sciknoweval_bio" bash run_chain.sh
```
This is a **template** (like the finance/temporalwiki chains): it needs a GPU, the
cartridges venv, a running Tokasaurus server (start it as in
`../cartridges_finance/run_chain.sh`), and the task parquets under `datasets/`.

## Eval
The chain serves the composed cartridge (OpenAI-compatible endpoint) and scores it
on each task with `scripts/eval_prompt.py`, which uses the same `cl.evals.<task>`
metrics every other method reports against. Point `EVAL_MODEL` at a
`configs/models/*.yaml` profile whose `api_base` is the served cartridge endpoint.
