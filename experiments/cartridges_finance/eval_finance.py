#!/usr/bin/env python3
"""Eval one (chain_phase, eval_year) cell of the cartridges finance matrix.

Loads the composed cartridge (cartridges 1..phase concatenated) onto Qwen3-8B
and runs the model on val_y{eval_year}.parquet's prompts. Scores every prompt
with finance.compute_score (up/down). Writes one row per prompt to a parquet
matching Anastasia's 10k_hf_analysis.py contract — same as our verl-path dump
(equity, filing_date, raw_return_30d, pred_label, gold_label, ...).

Usage:
    python eval_finance.py \
        --cartridges /workspace/.../finance_y2015_cartridge \
                     /workspace/.../finance_y2016_cartridge \
        --eval-year 2017 \
        --val-parquet /workspace/.../val_y2017.parquet \
        --returns-table /workspace/.../finance_returns_table.parquet \
        --out /workspace/.../cartridges_predictions/phase2/y2017.parquet
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List

import pandas as pd
import pyarrow.parquet as pq
import torch
# Disable inductor max-autotune BEFORE cartridges imports anything that uses
# torch.compile. cartridges/models/attention.py wraps flex_attention with
# `mode="max-autotune-no-cudagraphs"` and `dynamic=True` variants; under
# 8-way parallel eval this races on the autotune choice-enumeration filelock
# and bails with "NoValidChoicesError: No choices exist for backend".
# Forcing ATEN as a backend choice + disabling max-autotune gives a stable
# (slightly slower) path that doesn't depend on Triton kernel selection.
import torch._inductor.config as _inductor_cfg
_inductor_cfg.max_autotune = False
_inductor_cfg.max_autotune_gemm = False
_inductor_cfg.max_autotune_gemm_backends = "ATEN"
from transformers import AutoTokenizer

# Cartridges imports — this script must be invoked with `cartridges` installed
# in the active python env (uv pip install -e /home/nayan/methods/cartridges).
from cartridges.cache import TrainableCache, AttnConfig
from cartridges.models import FlexQwen3ForCausalLM
from cartridges.generation import flex_generate


_UP = re.compile(r"\bup\b", re.IGNORECASE)
_DOWN = re.compile(r"\bdown\b", re.IGNORECASE)


def _strip_chat_tail(s: str) -> str:
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL)
    for term in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
        idx = s.find(term)
        if idx != -1:
            s = s[:idx]
    return s


def _parse_label(text: str) -> str:
    if not isinstance(text, str): return ""
    t = _strip_chat_tail(text)
    if _UP.search(t): return "up"
    if _DOWN.search(t): return "down"
    return ""


def _label_to_score(L: str) -> float:
    return 1.0 if L == "up" else -1.0 if L == "down" else 0.0


def load_cartridge(p: Path, device: str = "cuda") -> TrainableCache:
    """Load a single cartridge using cartridges' own from_pretrained() — that
    knows the actual save format (trainable_keys, trainable_values, frozen_keys,
    frozen_values). We just locate the highest-step checkpoint under the run dir.
    """
    if p.is_dir():
        # Prefer cache_last.pt if present (final state), else highest cache-step{N}.pt
        last = p / "cache_last.pt"
        if last.exists():
            ckpt = last
        else:
            cands = sorted(p.glob("cache-step*.pt"),
                           key=lambda q: int(q.stem.replace("cache-step", "")))
            if not cands:
                raise FileNotFoundError(f"no cache-step*.pt or cache_last.pt under {p}")
            ckpt = cands[-1]
    else:
        ckpt = p
    cache = TrainableCache.from_pretrained(str(ckpt), device=device)
    # from_pretrained loads keys/values to device via map_location, but the
    # _seq_ids buffer is created CPU-side inside __init__ (no device arg in
    # torch.full). Calling .to(device) moves every registered buffer
    # (frozen_keys, trainable_keys, _init_seq_ids, _seq_ids) onto the GPU.
    cache = cache.to(device)
    return cache


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cartridge", type=Path, default=None,
                    help="Run dir (or path to cache-step*.pt) of ONE cartridge to load. "
                         "None = baseline zero-shot (no cartridge).")
    ap.add_argument("--eval-year", type=int, required=True)
    ap.add_argument("--val-parquet", type=Path, required=True)
    ap.add_argument("--returns-table", type=Path,
                    default=Path("/workspace/home/nayan/finance_data/finance_returns_table.parquet"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--phase-idx", type=int, required=True,
                    help="Which CL phase position this eval is for (1..6 for ordering F).")
    ap.add_argument("--train-year", type=int, required=True,
                    help="The year that *this phase* trained on (= 2014+phase).")
    ap.add_argument("--n-rollouts", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=32)  # binary classification: "up"/"down"+chat tail
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # ---- Load model + tokenizer
    print(f"[eval] loading model {args.model}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = FlexQwen3ForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device,
    )
    model.eval()

    # ---- Load the single cartridge for this phase (or run baseline if None).
    # No composition: cartridges-as-CL is interpreted as "one cartridge per
    # year, eval each independently against every val year". Composition by
    # KV concatenation is unsupported in upstream cartridges' codebase.
    cache = None
    if args.cartridge is not None:
        print(f"[eval] loading cartridge: {args.cartridge}", file=sys.stderr)
        cache = load_cartridge(args.cartridge, device=args.device)

    # ---- Returns lookup
    rdf = pd.read_parquet(args.returns_table)
    rdf["filing_date"] = rdf["filing_date"].astype(str)
    rdf["cik"] = rdf["cik"].astype(str)
    returns_lookup = rdf.set_index(["cik", "filing_date"])[
        ["raw_return_30d", "ticker", "company"]
    ]

    # ---- Iterate val parquet, generate, parse, score, write
    val = pq.read_table(args.val_parquet).to_pylist()
    out_records = []
    print(f"[eval] {len(val)} val prompts × {args.n_rollouts} rollouts", file=sys.stderr)

    # Whether to strip the [START OF FILING]...[END OF FILING] body from the user
    # message before tokenizing. The paper-canonical cartridges setup is to strip
    # (see longhealth eval), BUT body-stripping changes the Q-tensor shape enough
    # to trigger a flex_attention compile failure ("NoValidChoicesError: No
    # choices to select. Provided reason: No choices exist for backend.") that
    # we have not yet root-caused. So default to OFF (body-included) for now —
    # apples-to-apples with how SFT/SDPO/etc are evaluated. To opt back in:
    # CARTRIDGES_FINANCE_STRIP_BODY=1 (currently broken; needs a fix in
    # cartridges/models/attention.py's flex_attention compile path).
    strip_body = os.environ.get("CARTRIDGES_FINANCE_STRIP_BODY", "0") == "1"
    _BODY_STRIP_RE = re.compile(r"\n*\[START OF FILING\].*?\[END OF FILING\]\n*", re.DOTALL)

    for idx, r in enumerate(val):
        prompt_msgs = r["prompt"]
        if hasattr(prompt_msgs, "tolist"):
            prompt_msgs = prompt_msgs.tolist()
        msgs_for_template = []
        for m in prompt_msgs:
            md = dict(m)
            if strip_body and md.get("role") == "user":
                md = {**md, "content": _BODY_STRIP_RE.sub("\n[Filing content provided via cartridge memory]\n",
                                                          md["content"])}
            msgs_for_template.append(md)
        # FlexQwen3 + cartridge requires the cartridges-native flex_generate
        # path, NOT HuggingFace's model.generate (TrainableCache isn't
        # subscriptable like DynamicCache → KeyError). Tokenize via chat
        # template directly (matches examples/cartridge_chat.py).
        input_ids_2d = tok.apply_chat_template(
            msgs_for_template,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(args.device)
        flat_input_ids = input_ids_2d.flatten()
        seq_ids = torch.zeros(flat_input_ids.shape[0], dtype=torch.long, device=args.device)
        position_ids = torch.arange(flat_input_ids.shape[0], device=args.device)

        # N rollouts: clear the cache's dynamic state between rollouts so each
        # one starts from the cartridge prefix only (TrainableCache.clear()
        # zeros _keys/_values but preserves the trained _init_keys/_init_values).
        rollout_texts = []
        for _ in range(args.n_rollouts):
            if cache is not None:
                cache.clear()
            out = flex_generate(
                model=model,
                tokenizer=tok,
                input_ids=flat_input_ids,
                seq_ids=seq_ids,
                position_ids=position_ids,
                cache=cache,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            rollout_texts.append(tok.decode(out.get(0, []), skip_special_tokens=True))

        rollout_labels = [_parse_label(t) for t in rollout_texts]
        gold = str(dict(r["reward_model"]).get("ground_truth", "")).lower()
        n_up = sum(1 for L in rollout_labels if L == "up")
        n_down = sum(1 for L in rollout_labels if L == "down")
        if n_up >= n_down and n_up > 0: pred_label = "up"
        elif n_down > 0: pred_label = "down"
        else: pred_label = ""

        ei = dict(r["extra_info"])
        cik = str(ei.get("cik", ""))
        filing_date = str(ei.get("filing_date", ""))

        if (cik, filing_date) in returns_lookup.index:
            rrow = returns_lookup.loc[(cik, filing_date)]
            raw_ret = float(rrow["raw_return_30d"]); ticker = str(rrow["ticker"]); company = str(rrow["company"])
        else:
            raw_ret = float("nan"); ticker = ""; company = str(ei.get("company", ""))

        score_mean = sum(1.0 for L in rollout_labels if L == gold) / max(1, len(rollout_labels))

        out_records.append({
            "equity": ticker if ticker else cik,
            "filing_date": filing_date,
            "raw_return_30d": raw_ret,
            "pred_label": pred_label,
            "gold_label": gold,
            "pred_score": (n_up - n_down) / max(1, len(rollout_labels)),
            "gold_score": _label_to_score(gold),
            "section": "",
            "doc_id": f"{cik}::{filing_date}",
            "method": "cartridges",
            "phase": args.phase_idx,
            "train_year": args.train_year,
            "eval_year": args.eval_year,
            "cartridge_path": str(args.cartridge) if args.cartridge else "",
            "score_mean_at_8": score_mean,
            "n_rollouts_up": n_up,
            "n_rollouts_down": n_down,
            "n_rollouts_invalid": len(rollout_labels) - n_up - n_down,
            "raw_response_0": rollout_texts[0] if rollout_texts else "",
            "all_rollout_labels": "|".join(rollout_labels),
            "cik": cik, "ticker": ticker, "company": company,
        })

        if (idx + 1) % 10 == 0:
            print(f"  {idx+1}/{len(val)} done", file=sys.stderr)

    df = pd.DataFrame(out_records)
    df.to_parquet(args.out, index=False)
    acc = float((df["pred_label"] == df["gold_label"]).mean())
    score_mean = float(df["score_mean_at_8"].mean())
    print(f"[eval] wrote {args.out}  rows={len(df)}  acc={acc:.3f}  score_mean@8={score_mean:.3f}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
