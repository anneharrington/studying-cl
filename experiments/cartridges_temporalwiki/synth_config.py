#!/usr/bin/env python3
"""Cartridges synthesizer config — generate Q/A self-study conversations for
ONE TemporalWiki slice's article corpus via a LOCAL Tokasaurus Qwen3-8B server.

Mirrors cartridges_sweep/synth_config.py — env-var driven; reads the cumulative
article corpus produced by data_adapter.py for slice ts{i}, chunks article bodies
at 1024-2048 tokens (Wikipedia articles are typically 5-50K chars / 1-12K tokens
each), and emits N self-study conversations.
"""
from __future__ import annotations

import os
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.data.chunkers import TokenChunker
from cartridges.data.resources import TextFileResource
from cartridges.synthesize import SynthesizeConfig
from cartridges.synthesizers.self_study import SelfStudySynthesizer
from cartridges.utils.wandb import WandBConfig


SLICE_ID = os.environ.get("CARTRIDGES_TWIKI_SLICE", "ts1")
PHASE = int(os.environ.get("CARTRIDGES_TWIKI_PHASE", "1"))
CORPUS_ROOT = Path(os.environ.get(
    "CARTRIDGES_TWIKI_CORPUS_ROOT",
    "/workspace/home/nayan/cartridges_temporalwiki/corpora",
))
# Cumulative-corpus CL recipe: cartridge_i is trained on union of slices 1..i.
CORPUS_PATH = Path(os.environ.get(
    "CARTRIDGES_TWIKI_CORPUS_FILE",
    str(CORPUS_ROOT / f"cumulative_{SLICE_ID}" / "corpus.txt"),
))
SYNTH_PARQUET = os.environ.get(
    "CARTRIDGES_TWIKI_SYNTH_PARQUET",
    f"/workspace/home/nayan/cartridges_temporalwiki/runs/"
    f"twiki_p{PHASE}_{SLICE_ID}_synth_qwen-qwen3-8b_n8192/dataset.parquet",
)
N_SAMPLES = int(os.environ.get("CARTRIDGES_TWIKI_N_SYNTH", "8192"))

client = TokasaurusClient.Config(
    url=os.environ.get("CARTRIDGES_TOKASAURUS_URL", "http://localhost:10210"),
    model_name="Qwen/Qwen3-8B",
)

config = SynthesizeConfig(
    synthesizer=SelfStudySynthesizer.Config(
        client=client,
        max_rounds=1,
        prob_thinking=0.0,
        tools=[],
        resources=[
            TextFileResource.Config(
                path=str(CORPUS_PATH),
                # Same 4 seed prompts as finance/sweep — task-tailored variety.
                seed_prompts=["use_case", "question", "summarization", "structuring"],
                chunker=TokenChunker.Config(
                    tokenizer="Qwen/Qwen3-8B",
                    # Wikipedia articles are 1-12K tokens each; 1-2K chunks
                    # cover most subsections in one piece.
                    min_tokens_per_chunk=int(os.environ.get(
                        "CARTRIDGES_TWIKI_CHUNK_MIN", "1024")),
                    max_tokens_per_chunk=int(os.environ.get(
                        "CARTRIDGES_TWIKI_CHUNK_MAX", "2048")),
                ),
            ),
        ],
    ),
    num_samples=N_SAMPLES,
    batch_size=1,
    max_num_batches_in_parallel=int(os.environ.get(
        "CARTRIDGES_SYNTH_PARALLEL", "256")),

    name=FormatStringVariable(
        f"twiki_p{PHASE}_{SLICE_ID}_synth_{{synthesizer.client.model_name}}_n{{num_samples}}"
    ),
    run_id=FormatStringVariable("{name}"),
    output_dir=os.environ.get(
        "CARTRIDGES_OUTPUT_DIR",
        "/workspace/home/nayan/cartridges_temporalwiki/runs",
    ),

    wandb=None if os.environ.get("CARTRIDGES_WANDB_DISABLE") == "1" else WandBConfig(
        project=os.environ.get("CARTRIDGES_WANDB_PROJECT", "sdpo_seq"),
        entity=os.environ.get("CARTRIDGES_WANDB_ENTITY"),
        name=os.environ.get(
            "CARTRIDGES_TWIKI_RUN_TAG",
            "twiki-cl-cartridges_orderT_nothink_s42",
        ) + f"_synth_p{PHASE}_{SLICE_ID}",
        group=os.environ.get(
            "CARTRIDGES_TWIKI_RUN_TAG",
            "twiki-cl-cartridges_orderT_nothink_s42",
        ),
        tags=["cartridges", "synth", "orderT", f"p{PHASE}", SLICE_ID],
        notes=f"Self-study synth for cartridges/temporalwiki ordering T, phase {PHASE} ({SLICE_ID})",
    ),
    upload_to_wandb=False,
    save_wandb_preview=False,
    upload_to_hf=False,
    hf_repo_id="local/{wandb_run_id}",
)


if __name__ == "__main__":
    pydrantic.main([config])
