#!/usr/bin/env python3
"""Cartridges train config — context-distill ONE finance year's synthesized
self-study dataset into a trainable KV cache (the "cartridge") on Qwen3-8B.

Per CARTRIDGES_FINANCE_YEAR env var, this:
  - Initializes a TrainableCache from the year's corpus.txt (KVFromText)
  - Fine-tunes the cache using the synthesized dataset for that year
  - Saves the trained cartridge (state dict) to
    $CARTRIDGES_OUTPUT_DIR/finance_y{YEAR}_cartridge_*/...

Reference: examples/arxiv/arxiv_train.py in the cartridges repo.
"""
from __future__ import annotations

import os
from pathlib import Path

import pydrantic

from cartridges.initialization.text import KVFromRandomText
from cartridges.train import TrainConfig
from cartridges.models import HFModelConfig, FlexQwen3ForCausalLM
from cartridges.datasets import DataSource, TrainDataset
from cartridges.utils.wandb import WandBConfig


YEAR = int(os.environ.get("CARTRIDGES_FINANCE_YEAR", "2015"))
PHASE = int(os.environ.get("CARTRIDGES_FINANCE_PHASE", "1"))
CORPUS_ROOT = Path(os.environ.get(
    "CARTRIDGES_FINANCE_CORPUS_ROOT",
    "/workspace/home/nayan/cartridges_finance/corpora",
))
# Cumulative-corpus CL recipe: cartridge_i is trained on union(y1..y_i).
# data_adapter.py emits cumulative_y{i}/corpus.txt (shuffled doc order so
# KV init isn't biased to early years).
CORPUS_PATH = Path(os.environ.get(
    "CARTRIDGES_FINANCE_CORPUS_FILE",
    str(CORPUS_ROOT / f"cumulative_y{PHASE}" / "corpus.txt"),
))
# The synth pass writes a parquet here; we set this path per-year via env.
SYNTH_PARQUET = os.environ.get(
    "CARTRIDGES_FINANCE_SYNTH_PARQUET",
    f"/workspace/home/nayan/cartridges_finance/runs/"
    f"finance_y{YEAR}_synth_qwen-qwen3-8b_n2048/dataset.parquet",
)
EPOCHS = int(os.environ.get("CARTRIDGES_FINANCE_EPOCHS", "1"))
# Same as arxiv reference example. Earlier I halved this on a wrong assumption
# that packed_seq had to be 9× larger to "fit the full filing chunk" — but
# cartridges training inputs are just messages (user Q + assistant A, ~500
# tokens); the chunk lives in synth's system_prompt and is consumed by the
# CARTRIDGE during training, not by the model's forward pass. So packed_seq
# can stay tight (2048) and batch can stay at the paper-aligned 32.
GLOBAL_BATCH_SIZE = int(os.environ.get("CARTRIDGES_FINANCE_GLOBAL_BATCH", "32"))
LR = float(os.environ.get("CARTRIDGES_FINANCE_LR", "2e-2"))


config = TrainConfig(
    model=HFModelConfig(
        pretrained_model_name_or_path="Qwen/Qwen3-8B",
        model_cls=FlexQwen3ForCausalLM,
    ),
    # Paper recommends KVFromRandomText (README: "usually the best choice").
    # max_tokens here = the cartridge size p (paper notation). README tutorial
    # uses p=2048; that's a 16 KB-class KV cache regardless of corpus size.
    # Larger p improves quality but costs memory + train time linearly.
    kv_cache_initializer=KVFromRandomText.Config(
        max_tokens=int(os.environ.get("CARTRIDGES_FINANCE_P", "2048")),
    ),

    lr=LR,
    epochs=EPOCHS,
    global_batch_size=GLOBAL_BATCH_SIZE,

    dataset=TrainDataset.Config(
        data_sources=[DataSource(path=SYNTH_PARQUET, type="local")],
        top_k_logits=20,
        # packed_seq_length: per-batch packed token cap. The training input is
        # ONLY messages (user Q + assistant A) — the synth's system_prompt
        # (which holds the chunk) is replaced by the cartridge KV. So the
        # worst-case conversation is ~50 (Q) + ~1024 (A) + chat template
        # overhead ≈ 1500 tokens. 2048 matches the arxiv reference exactly
        # and keeps per-GPU activation memory in budget.
        packed_seq_length=int(os.environ.get("CARTRIDGES_FINANCE_PACKED_SEQ_LEN", "2048")),
        packing_mode="truncate",
    ),

    # No loss/gen evals during training — we run the eval matrix separately
    # against the val parquets so the eval surface matches our other 4 methods.
    # NB: cartridges' train.py:347 does `step % every_n_steps` unguarded, so 0
    # triggers ZeroDivisionError. Use a huge number to effectively disable
    # (we set loss_evals=[] and generate_evals=[] so no eval runs anyway).
    loss_eval_every_n_steps=10**9,
    loss_evals=[],
    generate_eval_every_n_steps=10**9,
    generate_evals=[],

    # NCCL for GPU collectives (the arxiv example used "gloo" which is for CPU
    # collectives — much slower for FSDP across H100s).
    distributed_backend="nccl",
    save_every_n_steps=512,
    name=f"finance_y{YEAR}_cartridge",

    # wandb: same project as the other finance methods (sdpo_seq) so all 5
    # methods land in one dashboard. Run name follows canonical pattern.
    # Disable with CARTRIDGES_WANDB_DISABLE=1.
    wandb=None if os.environ.get("CARTRIDGES_WANDB_DISABLE") == "1" else WandBConfig(
        project=os.environ.get("CARTRIDGES_WANDB_PROJECT", "sdpo_seq"),
        entity=os.environ.get("CARTRIDGES_WANDB_ENTITY"),
        # name + group both follow the SDPO convention:
        #   <RUN_TAG>_train_y{YEAR}
        name=os.environ.get(
            "CARTRIDGES_FINANCE_RUN_TAG",
            "finance-cl-cartridges_orderF_nothink_s42",
        ) + f"_train_y{YEAR}",
        group=os.environ.get(
            "CARTRIDGES_FINANCE_RUN_TAG",
            "finance-cl-cartridges_orderF_nothink_s42",
        ),
        tags=["cartridges", "train", "orderF", f"y{YEAR}"],
        notes=f"Cartridge training for finance ordering F, year {YEAR}",
    ),
)


if __name__ == "__main__":
    pydrantic.main(config)
