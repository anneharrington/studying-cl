# Studying-CL quickstart targets.
#
# Common entry points so external users don't need to remember the per-subsystem
# venv + env-var setup. Each target is idempotent.
#
#   make setup        bootstrap all four venvs from envs/requirements-*.txt
#   make setup-verl   bootstrap only the verl venv (weight-update methods)
#   make smoke        1-step smoke for every subsystem (≈45 min on 1 GPU)
#   make smoke-weight 1-step SFT smoke for each of the three task families
#   make data         (placeholder) download + prep public datasets
#   make clean-runs   remove the runs/ directory (training output, not code)
#
# Environment knobs (set in your shell or .env):
#   WANDB_API_KEY        weight-update methods (verl)
#   OPENROUTER_API_KEY   prompt-based + cartridges synthesis
#   CL_TEMPORAL_DRIFT_DATA  path to TemporalWiki drift parquets
#   CL_FINANCE_DATA      path to finance per-year parquets

SHELL := /usr/bin/env bash
REPO  := $(shell pwd)

.PHONY: setup setup-harness setup-verl setup-cartridges setup-ttt \
        smoke smoke-weight smoke-weight-A smoke-weight-T smoke-weight-F \
        smoke-prompt-imports smoke-prompt \
        data clean-runs help

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk -F':.*?## ' '{printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}' || \
	  awk 'NR<=18 {print}' $(MAKEFILE_LIST)

setup: setup-harness setup-verl setup-cartridges setup-ttt  ## bootstrap all venvs
	./bootstrap.sh

setup-harness:   ## bootstrap .venv-harness only (prompt-based + shared eval)
	./bootstrap.sh harness
setup-verl:      ## bootstrap .venv-verl only (weight-update methods; needs GPU)
	./bootstrap.sh verl
setup-cartridges:## bootstrap .venv-cartridges only (Cartridges compression)
	./bootstrap.sh cartridges
setup-ttt:       ## bootstrap .venv-ttt only (In-Place TTT compression)
	./bootstrap.sh ttt

smoke: smoke-weight  ## end-to-end smoke for every subsystem (weight only for now)

smoke-weight: smoke-weight-A smoke-weight-T smoke-weight-F  ## SFT 1-step on each task family

smoke-weight-A:  ## 1-step SFT on domain_shift (orderA, bio)
	sbatch -A nayan experiments/continual/smoke_studying_cl.sbatch
smoke-weight-T:  ## 1-step SFT on discrete_updates (orderT, TemporalWiki ts1)
	sbatch -A nayan experiments/continual/smoke_studying_cl_twiki.sbatch
smoke-weight-F:  ## 1-step SFT on temporal_drift (orderF, finance y2015)
	sbatch -A nayan experiments/continual/smoke_studying_cl_finance.sbatch

smoke-prompt-imports:  ## verify .venv-harness imports + sequential.yaml parses (no API key needed)
	@source .venv-harness/bin/activate 2>/dev/null || { \
	    echo "!! .venv-harness missing — run 'make setup-harness'"; exit 2; }; \
	python -c "import cl, cl.config; cfg = cl.config.load_config('configs/runs/sequential.yaml'); \
	print(f'  sequential.yaml -> {cfg!r}'); print('  harness imports + run config OK')"

smoke-prompt:  ## minimal GEPA run on sciknoweval_bio (needs OPENROUTER_API_KEY + data/parquet/sciknoweval_bio/)
	@source .venv-harness/bin/activate 2>/dev/null || { \
	    echo "!! .venv-harness missing — run 'make setup-harness'"; exit 2; }; \
	[ -n "$$OPENROUTER_API_KEY" ] || { echo "!! OPENROUTER_API_KEY unset — see bootstrap.sh"; exit 2; }; \
	[ -f data/parquet/sciknoweval_bio/train.parquet ] || { \
	    echo "!! data/parquet/sciknoweval_bio/train.parquet missing — bootstrap.sh check_data tells you how"; exit 2; }; \
	python scripts/run.py --method gepa --strategy sequential \
	    --config configs/runs/sequential.yaml --tasks sciknoweval_bio \
	    --train-n 4 --val-n 4 --eval-n 4 --model qwen-3-8b

data:  ## (placeholder) run dataset prep — see data_prep/ and scripts/download_data.sh
	@echo "data-prep is dataset-specific; see scripts/download_data.sh, data/prep/, and docs/TASKS.md"

clean-runs:  ## remove training output (does not touch code/checkpoints in S3)
	rm -rf runs/ outputs/
