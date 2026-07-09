# Cartridges as a 5th method on the finance ordering-F benchmark

This directory turns the [HazyResearch Cartridges](https://github.com/HazyResearch/cartridges)
recipe (test-time-trained KV-cache prefixes via self-study) into a continual-learning
baseline that drops into our finance ordering-F sweep alongside SFT / SDFT / SDPO / GRPO.

## Pipeline (per phase i = year y_i in 2015..2020)

```
                      ┌──────────────────────────────────────┐
  train_y{y_i}.parq → │ data_adapter.py                      │ → corpora/y{y_i}/corpus.txt
                      │   bodies of 500 10-Ks concatenated   │
                      └──────────────────────────────────────┘

  corpus.txt ──┐
               │
               ▼
            ┌────────────────────────────────────────────────┐
            │ synth_config.py                                │
            │   SelfStudySynthesizer + OpenRouter Qwen3-8B   │  ──► dataset.parquet
            │   (Alibaba pin) self-asks N≈2k Q/A on chunks   │      (~2k convos)
            └────────────────────────────────────────────────┘

  dataset.parquet  +  corpus.txt
            │              │
            ▼              ▼
            ┌────────────────────────────────────────────────┐
            │ train_config.py                                │
            │   KVFromText → TrainableCache; context-distill │  ──► cartridge_y{y_i}/
            │   on the synth dataset (Qwen3-8B, 1 epoch)     │      (state dict)
            └────────────────────────────────────────────────┘

  Compose cartridges 1..i (concatenate KV along seq dim)
            │
            ▼
            ┌────────────────────────────────────────────────┐
            │ eval_finance.py                                │
            │   load composed cartridge → Qwen3-8B forward   │  ──► <pred_root>/cartridges/
            │   on val_y{y_j}.parquet ×8 rollouts/prompt     │      phase{i}/y{y_j}.parquet
            └────────────────────────────────────────────────┘
```

The 6×6 matrix of resulting parquets matches the schema Anastasia's
`10k_hf_analysis.py` consumes (same columns as our verl-path dump).

## Files

| File | Purpose |
|---|---|
| `data_adapter.py`     | finance train parquet → per-year `corpus.txt` + `manifest.json` |
| `synth_config.py`     | SelfStudySynthesizer config (OpenRouter Qwen3-8B, Alibaba-pinned) |
| `train_config.py`     | TrainConfig (KVFromText init + Qwen3-8B context-distill) |
| `eval_finance.py`     | Load composed cartridges → val rollouts → score → Anastasia-format parquet |
| `openrouter_client.py`| OpenAIClient subclass that injects the Alibaba pin + `/no_think` |
| `run_chain.sh`        | Sweep driver: 6 synth + 6 train + 36 eval cells |
| `install.sh`          | One-time `uv pip install -e cartridges` into our venv |

## One-time setup

Cartridges has its own python deps (`transformers>=4.49,<=4.55` + bleeding-edge
`torch` for FlexAttention) that conflict with our verl/vllm pin in
`venv-sdpo-v2`. So `install.sh` makes a **separate venv** at
`/workspace/home/nayan/venv-cartridges/` (on shared MooseFS, visible from both
nodes — same approach as Anastasia's slurm-trap fix in the temporal-drift
handoff).

```bash
# 1. Make the cartridges venv + install (~3 min)
bash /home/nayan/scripts/cartridges_finance/install.sh

# 2. (Optional) build per-year corpora once. run_chain.sh does this anyway.
source /workspace/home/nayan/venv-cartridges/bin/activate
python /home/nayan/scripts/cartridges_finance/data_adapter.py
```

## Running the full chain

```bash
export OPENROUTER_API_KEY=...
bash /home/nayan/scripts/cartridges_finance/run_chain.sh
```

Phases auto-resume — re-running skips synth, train, or eval cells whose outputs
already exist.

### Knobs

| Env var | Default | Effect |
|---|---|---|
| `YEARS`                       | `"2015 ... 2020"` | which phases to run |
| `CARTRIDGES_FINANCE_N_SYNTH`  | `2048`            | self-study conversations per phase |
| `CARTRIDGES_FINANCE_EPOCHS`   | `1`               | training epochs |
| `CARTRIDGES_FINANCE_LR`       | `2e-2`            | cartridge LR (paper recommends 2e-2) |
| `SKIP_SYNTH=1`                | —                 | reuse existing synth datasets |
| `SKIP_TRAIN=1`                | —                 | reuse existing cartridges |
| `SKIP_EVAL=1`                 | —                 | only do synth + train |

### Cost / time estimate

| Step | Per phase | Total (6 phases) | Hardware |
|---|---|---|---|
| Synth (OpenRouter Qwen3-8B, Alibaba pin) | ~$1-2 API, ~10-15 min wall | ~$6-12, ~1 hr | API only |
| Train (Qwen3-8B context-distill, FSDP+nccl) | n=2048 packed=2048 ⇒ ~64 steps; ~5-15 sec/step on 8×H100 ⇒ ~5-15 min | ~30-90 min | local 8 GPUs |
| Eval (36 cells, 50 prompts × 8 rollouts each) | num_return_sequences=8 shares prefill ⇒ ~10-15 sec/prompt ⇒ ~10 min/cell | ~6 hr | local 8 GPUs |
| **Total** | | **~7-9 GPU hr + ~$10 API** | |

The eval step is the dominant cost: 50 val prompts × 36 cells = 1800 prompts,
each with a ~13k-token finance filing as the prompt. We batch the 8 rollouts
per prompt into a single `generate()` call (`num_return_sequences=8`), which
shares the prefill — the largest cost on long inputs. Looping `generate()`
8 times instead would cost ~8× more.

### wandb

Logs to the same `sdpo_seq` project as our verl-trained methods so the
5-method comparison lives in one dashboard. Run names follow the canonical
pattern:

```
finance-cl-cartridges_orderF_nothink_s42_synth_y2015     # synth phase 1
finance-cl-cartridges_orderF_nothink_s42_train_y2015     # train phase 1
finance-cl-cartridges_orderF_nothink_s42_synth_y2016     # synth phase 2
… etc.
```

All runs share `group=finance-cl-cartridges_orderF_nothink_s42` so they're easy
to filter as a single chain. Override the project with `CARTRIDGES_WANDB_PROJECT=...`
or disable wandb entirely with `CARTRIDGES_WANDB_DISABLE=1`.

## Caveats

1. **Composition by KV concatenation** is the canonical recipe per the paper, but
   it grows the prefix length linearly with phase count. For phase 6, the
   composed cache holds 6 × 32k = 192k tokens of KV — possibly larger than the
   model's natively-trained max RoPE position. If you hit OOM or attention sinks
   collapsing, try halving `max_tokens` in `train_config.py`'s `KVFromText.Config`.

2. **OpenRouter Alibaba pin is mandatory** for Qwen3-8B's long context. Without
   it, OpenRouter may route to a provider with a 32k cap and the chunker will
   fail on the longest 10-K filings.

3. **`Tokasaurus` is what the cartridges paper used**; we substitute OpenRouter
   for cost/availability. If signal looks weak or the synthesizer hits rate
   limits, switch to `infra/modal_deploy_tokasaurus.py` per the cartridges
   README.
