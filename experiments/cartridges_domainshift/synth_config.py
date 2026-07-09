#!/usr/bin/env python3
"""Cartridges synthesizer config — generate Q/A self-study conversations for
ONE finance year corpus via a LOCAL Tokasaurus Qwen3-8B server.

Why Tokasaurus and not OpenRouter/vLLM-OAI:
  - Cartridges training is logit-distillation on top-k logprobs WITH token IDs
    (see cartridges/datasets.py:71 and train.py:383-395). The OpenAI Chat
    Completions API doesn't return token IDs (only token strings) — cartridges'
    OpenAIClient stuffs token_ids with -1 placeholders, which silently breaks
    the distillation target during training.
  - Tokasaurus is the inference server cartridges' authors built and use; it
    returns clean (token_id, top_k_logprobs) per position. TokasaurusClient is
    a first-class citizen of the cartridges codebase.

Per CARTRIDGES_FINANCE_YEAR env var, this synthesizes N self-study conversations
on that year's corpus and writes them to the output dir.

Run via cartridges' pydrantic main:
    CARTRIDGES_TOKASAURUS_URL=http://localhost:8080 \
    CARTRIDGES_FINANCE_YEAR=2015 \
    python /home/nayan/scripts/cartridges_finance/synth_config.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.data.chunkers import TokenChunker
from cartridges.data.resources import TextFileResource
from cartridges.synthesize import SynthesizeConfig
from cartridges.synthesizers.self_study import SelfStudySynthesizer
from cartridges.utils.wandb import WandBConfig


YEAR = int(os.environ.get("CARTRIDGES_FINANCE_YEAR", "2015"))
PHASE = int(os.environ.get("CARTRIDGES_FINANCE_PHASE", "1"))
CORPUS_ROOT = Path(os.environ.get(
    "CARTRIDGES_FINANCE_CORPUS_ROOT",
    "/workspace/home/nayan/cartridges_finance/corpora",
))
# Cumulative-corpus CL recipe: synth chunks come from union(y1..y_i).
CORPUS_PATH = Path(os.environ.get(
    "CARTRIDGES_FINANCE_CORPUS_FILE",
    str(CORPUS_ROOT / f"cumulative_y{PHASE}" / "corpus.txt"),
))
# Generous default — paper's HF-published arxiv synth dataset is n=8192. Smoke
# at n=512 (5-10 min), production at n=8192 (~1 hr per phase via Tokasaurus).
N_SAMPLES = int(os.environ.get("CARTRIDGES_FINANCE_N_SYNTH", "8192"))

client = TokasaurusClient.Config(
    url=os.environ.get("CARTRIDGES_TOKASAURUS_URL", "http://localhost:8080"),
    model_name="Qwen/Qwen3-8B",
)

config = SynthesizeConfig(
    synthesizer=SelfStudySynthesizer.Config(
        client=client,
        max_rounds=1,                     # paper default
        prob_thinking=0.0,                # Qwen3 reasoning off (clean short answers)
        tools=[],
        resources=[
            TextFileResource.Config(
                path=str(CORPUS_PATH),
                # Bias seed types toward the actual downstream task:
                #   - "use_case": "given this filing, predict X" — direct match
                #   - "question": general Q/A about filings
                #   - "summarization": compress to sentiment-bearing language
                #   - "structuring": extract (company, sector, drivers) tuples
                seed_prompts=["use_case", "question", "summarization", "structuring"],
                chunker=TokenChunker.Config(
                    tokenizer="Qwen/Qwen3-8B",
                    # Chunk size matches SDPO's eval-time prompt length (16384).
                    # Each 10-K body is ~13k tokens — at chunk size 8-16k, each
                    # synth chunk is roughly one whole filing in context, which
                    # is what the model sees at eval time. Paper-default 512-1024
                    # would only show the synth model paragraph-sized snippets.
                    min_tokens_per_chunk=int(os.environ.get(
                        "CARTRIDGES_FINANCE_CHUNK_MIN", "8192")),
                    max_tokens_per_chunk=int(os.environ.get(
                        "CARTRIDGES_FINANCE_CHUNK_MAX", "16384")),
                ),
            ),
        ],
    ),
    num_samples=N_SAMPLES,
    batch_size=1,
    # 256 in-flight matches cartridges' arxiv reference example exactly.
    # The earlier "received 0 items of ancdata" stall was fixed at the
    # tokasaurus side (entry.py + every worker calls
    # mp.set_sharing_strategy('file_system')), so we can run at full throttle.
    max_num_batches_in_parallel=int(os.environ.get(
        "CARTRIDGES_SYNTH_PARALLEL", "256")),

    name=FormatStringVariable(
        f"finance_y{YEAR}_synth_{{synthesizer.client.model_name}}_n{{num_samples}}"
    ),
    run_id=FormatStringVariable("{name}"),
    output_dir=os.environ.get(
        "CARTRIDGES_OUTPUT_DIR",
        "/workspace/home/nayan/cartridges_finance/runs",
    ),

    # wandb: same project as our other finance methods (sdpo_seq) so the
    # 5-method comparison lives in one dashboard. Run names follow the
    # canonical pattern. Disable with CARTRIDGES_WANDB_DISABLE=1.
    wandb=None if os.environ.get("CARTRIDGES_WANDB_DISABLE") == "1" else WandBConfig(
        project=os.environ.get("CARTRIDGES_WANDB_PROJECT", "sdpo_seq"),
        entity=os.environ.get("CARTRIDGES_WANDB_ENTITY"),
        # name + group both follow the SDPO convention:
        #   <RUN_TAG>_synth_y{YEAR}
        # so re-running with a fresh sbatch (which sets a fresh timestamp+SHA
        # in RUN_TAG) creates a fresh wandb group; resume by setting RUN_TAG.
        name=os.environ.get(
            "CARTRIDGES_FINANCE_RUN_TAG",
            "finance-cl-cartridges_orderF_nothink_s42",
        ) + f"_synth_y{YEAR}",
        group=os.environ.get(
            "CARTRIDGES_FINANCE_RUN_TAG",
            "finance-cl-cartridges_orderF_nothink_s42",
        ),
        tags=["cartridges", "synth", "orderF", f"y{YEAR}"],
        notes=f"Self-study synth for cartridges/finance ordering F, year {YEAR}",
    ),
    upload_to_wandb=False,
    save_wandb_preview=False,
    upload_to_hf=False,
    hf_repo_id="local/{wandb_run_id}",
)


if __name__ == "__main__":
    pydrantic.main([config])
