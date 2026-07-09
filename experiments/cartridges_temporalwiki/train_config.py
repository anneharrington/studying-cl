#!/usr/bin/env python3
"""Cartridges train config — context-distill ONE TemporalWiki phase's
synthesized self-study dataset into a trainable KV cache (the "cartridge")
on Qwen3-8B.

Mirrors cartridges_sweep/train_config.py — same lr/epochs/batch/p — just
points at the temporalwiki-specific synth parquet and uses temporalwiki
run-tag conventions.
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


PHASE = int(os.environ.get("CARTRIDGES_TWIKI_PHASE", "1"))
SLICE_ID = os.environ.get("CARTRIDGES_TWIKI_SLICE", "ts1")
SYNTH_PARQUET = os.environ.get(
    "CARTRIDGES_TWIKI_SYNTH_PARQUET",
    f"/workspace/home/nayan/cartridges_temporalwiki/runs/"
    f"twiki_p{PHASE}_{SLICE_ID}_synth_qwen-qwen3-8b_n8192/dataset.parquet",
)
EPOCHS = int(os.environ.get("CARTRIDGES_TWIKI_EPOCHS", "1"))
GLOBAL_BATCH_SIZE = int(os.environ.get("CARTRIDGES_TWIKI_GLOBAL_BATCH", "32"))
LR = float(os.environ.get("CARTRIDGES_TWIKI_LR", "2e-2"))


config = TrainConfig(
    model=HFModelConfig(
        pretrained_model_name_or_path="Qwen/Qwen3-8B",
        model_cls=FlexQwen3ForCausalLM,
    ),
    kv_cache_initializer=KVFromRandomText.Config(
        max_tokens=int(os.environ.get("CARTRIDGES_TWIKI_P", "2048")),
    ),

    lr=LR,
    epochs=EPOCHS,
    global_batch_size=GLOBAL_BATCH_SIZE,

    dataset=TrainDataset.Config(
        data_sources=[DataSource(path=SYNTH_PARQUET, type="local")],
        top_k_logits=20,
        packed_seq_length=int(os.environ.get("CARTRIDGES_TWIKI_PACKED_SEQ_LEN", "2048")),
        packing_mode="truncate",
    ),

    loss_eval_every_n_steps=10**9,
    loss_evals=[],
    generate_eval_every_n_steps=10**9,
    generate_evals=[],

    distributed_backend="nccl",
    save_every_n_steps=512,
    name=f"twiki_p{PHASE}_{SLICE_ID}_cartridge",

    wandb=None if os.environ.get("CARTRIDGES_WANDB_DISABLE") == "1" else WandBConfig(
        project=os.environ.get("CARTRIDGES_WANDB_PROJECT", "sdpo_seq"),
        entity=os.environ.get("CARTRIDGES_WANDB_ENTITY"),
        name=os.environ.get(
            "CARTRIDGES_TWIKI_RUN_TAG",
            "twiki-cl-cartridges_orderT_nothink_s42",
        ) + f"_train_p{PHASE}_{SLICE_ID}",
        group=os.environ.get(
            "CARTRIDGES_TWIKI_RUN_TAG",
            "twiki-cl-cartridges_orderT_nothink_s42",
        ),
        tags=["cartridges", "train", "orderT", f"p{PHASE}", SLICE_ID],
        notes=f"Cartridge training for cartridges/temporalwiki ordering T, phase {PHASE} ({SLICE_ID})",
    ),
)


if __name__ == "__main__":
    pydrantic.main(config)
