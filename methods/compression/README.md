# Compression-based methods — Cartridges & In-Place TTT

Family from the paper: **compression-based** continual learning. Both are third-party
implementations included here with their licenses (Cartridges, In-Place TTT), each with
its **own dependencies and environment** — they do not use the `cl/` harness or the verl
engine. See `docs/SETUP.md` for the per-method venvs (`envs/requirements-cartridges.txt`,
`envs/requirements-ttt.txt`).

Run via the unified entry point (routes to the `experiments/*_<family>/run_chain.sh`):
`./run.sh cartridges <task_family>` / `./run.sh in_place_ttt <task_family>`
(`task_family` = `domain_shift` | `temporal_drift` | `discrete_updates`).

## `cartridges/` — KV-cache compression via self-study
Two-stage: (1) synthesize synthetic Q/A about a corpus, (2) train a small trainable
KV cache ("cartridge") via context distillation.
- Entrypoints: `cartridges/synthesize.py` (`SynthesizeConfig`), `cartridges/train.py`
  (`TrainConfig`); patterns in `cartridges/examples/arxiv/`.
- Needs a Tokasaurus/SGLang inference server for synthesis.

## `in_place_ttt/` — continual pretraining via test-time training
Updates MLP down-projection "fast weights" during training. VeOmni-based.
- Train: `in_place_ttt/train.sh tasks/train_torch.py configs/pretrain/qwen3_longct.yaml ...`
- Convert DCP→HF: `in_place_ttt/scripts/merge_dcp_to_hf.py`
- Eval: `in_place_ttt/eval.sh` (OpenCompass RULER configs in `eval_config/`)

## Task-family wiring
Both are wired for all three task families via the `experiments/` run-chains:
**domain shift** (`cartridges_domainshift/`, `ttt_domainshift/`), **temporal drift /
finance** (`cartridges_finance/`, `ttt_finance/`), and **discrete updates /
TemporalWiki** (`cartridges_temporalwiki/`, `ttt_temporalwiki/`). Each `run_chain.sh`
chains phases: adapt on phase *i* data → save checkpoint → evaluate on per-phase val
splits (domain-shift eval routes through the shared `cl.evals` harnesses).
