#!/usr/bin/env python3
"""Eval one (chain_phase, eval_set) cell of the cartridges TemporalWiki matrix.

Loads the cartridge for `phase i` (trained on cumulative_ts{i} articles) onto
Qwen3-8B and runs it on val_ts{j}.parquet (drift slices) or val_stable.parquet.
Scoring uses SDPO/SFT/GRPO's exact temporalwiki scorer (verbatim copy at
/workspace/home/nayan/_sync/sdpo_scorers/temporalwiki.py — F1 ≥ 0.5 over
article+punctuation-stripped tokens) so cartridge numbers are apples-to-apples
with the existing twiki-cl_*_orderT_nothink_s42 baselines.

Writes a parquet with consistent columns (phase, train_slice, eval_set,
gold_text, pred_text, hit_majority, score_mean_acc_at_8, ...).
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import torch

# Disable inductor max-autotune before cartridges imports — flex_attention's
# autotune choice list races on parallel compile (NoValidChoicesError seen
# on cartridges_finance prod). Force ATEN backend defensively.
import torch._inductor.config as _ic
_ic.max_autotune = False
_ic.max_autotune_gemm = False
_ic.max_autotune_gemm_backends = "ATEN"
from transformers import AutoTokenizer

from cartridges.cache import TrainableCache
from cartridges.models import FlexQwen3ForCausalLM
from cartridges.generation import flex_generate

# Bypass cartridges' torch.compile wrapper around flex_attention. Short
# temporalwiki prompts (~150 tokens) trigger a recompile that hits
# `NoValidChoicesError: No choices exist for backend` because the dynamic-
# shape choice list is empty for these dimensions. Replacing the compiled
# wrapper with the eager flex_attention sidesteps the autotune entirely;
# eager is fast enough for our short eval prompts.
import torch.nn.attention.flex_attention as _fa
import cartridges.models.attention as _ca
_ca.flex_attention_generate = _fa.flex_attention
_ca.flex_attention_train = _fa.flex_attention

# SDPO's exact scorer for temporalwiki_*. The `temporalwiki.compute_score(sol,
# gold)` returns dict {score, acc, pred, ...} where `score` is the raw F1 and
# `acc` is binary (F1 >= 0.5). Per SDPO/run_sequential.py, all temporalwiki
# data sources (drift_s1/s2/s3/stable) route to the same scorer; the slice tag
# is the metric-axis label.
sys.path.insert(0, "/workspace/home/nayan/_sync")
from sdpo_scorers import temporalwiki as twiki_scorer  # noqa: E402


_NORMALIZE_FOR_VOTE = (re.compile(r"\b(a|an|the)\b", re.IGNORECASE), str.maketrans("", "", "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"), re.compile(r"\s+"))


def _normalize(s: str) -> str:
    """Same normalization as twiki_scorer._normalize, surfaced for majority-voting."""
    arts, punct, ws = _NORMALIZE_FOR_VOTE
    s = (s if isinstance(s, str) else str(s)).lower()
    s = arts.sub(" ", s)
    s = s.translate(punct)
    s = ws.sub(" ", s).strip()
    return s


def load_cartridge(p: Path, device: str = "cuda") -> TrainableCache:
    if p.is_dir():
        last = p / "cache_last.pt"
        if last.exists(): ckpt = last
        else:
            cands = sorted(p.glob("cache-step*.pt"),
                           key=lambda q: int(q.stem.replace("cache-step", "")))
            if not cands: raise FileNotFoundError(f"no cache-step*.pt or cache_last.pt under {p}")
            ckpt = cands[-1]
    else:
        ckpt = p
    cache = TrainableCache.from_pretrained(str(ckpt), device=device)
    return cache.to(device)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cartridge", type=Path, default=None,
                    help="Run dir of ONE cartridge to load. None = baseline zero-shot.")
    ap.add_argument("--eval-set", required=True,
                    help="drift_s1 | drift_s2 | drift_s3 | stable")
    ap.add_argument("--train-slice", required=True, help="ts1 | ts2 | ts3")
    ap.add_argument("--phase-idx", type=int, required=True)
    ap.add_argument("--val-parquet", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-rollouts", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[eval] loading model {args.model}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = FlexQwen3ForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device,
    ).eval()

    cache = None
    if args.cartridge is not None:
        print(f"[eval] loading cartridge: {args.cartridge}", file=sys.stderr)
        cache = load_cartridge(args.cartridge, device=args.device)

    val = pq.read_table(args.val_parquet).to_pylist()
    out_records = []
    print(f"[eval] {len(val)} val prompts × {args.n_rollouts} rollouts (eval_set={args.eval_set})",
          file=sys.stderr, flush=True)

    for idx, r in enumerate(val):
        prompt_msgs = r["prompt"]
        if hasattr(prompt_msgs, "tolist"): prompt_msgs = prompt_msgs.tolist()
        input_ids_2d = tok.apply_chat_template(
            [dict(m) for m in prompt_msgs],
            tokenize=True, add_generation_prompt=True,
            return_tensors="pt", enable_thinking=False,
        ).to(args.device)
        flat_input_ids = input_ids_2d.flatten()
        seq_ids = torch.zeros(flat_input_ids.shape[0], dtype=torch.long, device=args.device)
        position_ids = torch.arange(flat_input_ids.shape[0], device=args.device)

        rollout_texts = []
        for _ in range(args.n_rollouts):
            if cache is not None: cache.clear()
            out = flex_generate(
                model=model, tokenizer=tok,
                input_ids=flat_input_ids, seq_ids=seq_ids,
                position_ids=position_ids, cache=cache,
                max_new_tokens=args.max_tokens, temperature=args.temperature,
            )
            rollout_texts.append(tok.decode(out.get(0, []), skip_special_tokens=True))

        gold = str(dict(r["reward_model"]).get("ground_truth", "")).strip()

        # Score each rollout with SDPO's exact temporalwiki scorer.
        rollout_results = [twiki_scorer.compute_score(t, gold) for t in rollout_texts]
        rollout_preds = [str(rr.get("pred", "")) for rr in rollout_results]
        rollout_f1s = [float(rr.get("score", 0.0)) for rr in rollout_results]
        rollout_accs = [float(rr.get("acc", 0.0)) for rr in rollout_results]
        score_mean_f1 = sum(rollout_f1s) / max(1, len(rollout_f1s))
        score_mean_acc = sum(rollout_accs) / max(1, len(rollout_accs))

        # Majority vote on normalized predictions; tie-break by first occurrence.
        cnt = Counter([_normalize(p) for p in rollout_preds if p])
        pred_norm_majority = cnt.most_common(1)[0][0] if cnt else ""
        # Pick the raw form whose normalization equals the majority (for display).
        pred_text = next(
            (p for p in rollout_preds if _normalize(p) == pred_norm_majority),
            "",
        )
        # Re-score the majority pick to get its hit/F1 — same scorer.
        majority_result = twiki_scorer.compute_score(pred_text, gold)
        hit_majority = majority_result.get("acc", 0.0) >= 1.0
        majority_f1 = majority_result.get("score", 0.0)

        ei = dict(r["extra_info"]) if r.get("extra_info") is not None else {}
        out_records.append({
            "doc_id": str(ei.get("index", idx)),
            "subject": ei.get("subject", ""),
            "relation": ei.get("relation", ""),
            "pred_text": pred_text,
            "gold_text": gold,
            "majority_f1": majority_f1,
            "hit_majority": bool(hit_majority),
            "method": "cartridges",
            "phase": args.phase_idx,
            "train_slice": args.train_slice,
            "eval_set": args.eval_set,
            "data_source": str(r.get("data_source", "")),
            "cartridge_path": str(args.cartridge) if args.cartridge else "",
            "score_mean_f1_at_8": score_mean_f1,
            "score_mean_acc_at_8": score_mean_acc,
            "n_rollouts_correct": int(sum(int(a >= 1.0) for a in rollout_accs)),
            "raw_response_0": rollout_texts[0] if rollout_texts else "",
            "all_rollout_preds": "|".join(rollout_preds),
        })
        if (idx + 1) % 25 == 0:
            print(f"  {idx+1}/{len(val)} done", file=sys.stderr, flush=True)

    df = pd.DataFrame(out_records)
    df.to_parquet(args.out, index=False)
    acc_maj = float(df["hit_majority"].mean())
    f1_mean = float(df["score_mean_f1_at_8"].mean())
    acc_mean = float(df["score_mean_acc_at_8"].mean())
    print(
        f"[eval] wrote {args.out}  rows={len(df)}  acc_majority={acc_maj:.3f}  "
        f"f1_mean@8={f1_mean:.3f}  acc_mean@8={acc_mean:.3f}",
        file=sys.stderr, flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
