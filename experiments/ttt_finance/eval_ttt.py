#!/usr/bin/env python3
"""Eval one (chain_phase, eval_year) cell of the In-Place TTT finance matrix.

Loads the converted HF Qwen3-8B checkpoint (which is just a regular causal-LM
with TTT-shifted weights) and runs vanilla HF generate on val_y{eval_year}.parquet.
Scores each prompt with up/down, writes a parquet matching Anastasia's
10k_hf_analysis.py contract (same schema as cartridges/sft/sdpo predictions).

Usage:
    python eval_ttt.py \
        --hf-ckpt /workspace/.../train/y2015/hf \
        --eval-year 2017 --train-year 2015 --phase-idx 1 \
        --val-parquet /workspace/.../val_y2017.parquet \
        --returns-table /workspace/.../finance_returns_table.parquet \
        --out /workspace/.../ttt_predictions/phase1/y2017.parquet
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf-ckpt", type=Path, required=True,
                    help="Path to the converted HF checkpoint (output of merge_dcp_to_hf.py).")
    ap.add_argument("--eval-year", type=int, required=True)
    ap.add_argument("--val-parquet", type=Path, required=True)
    ap.add_argument("--returns-table", type=Path,
                    default=Path("/workspace/home/nayan/finance_data/finance_returns_table.parquet"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--phase-idx", type=int, required=True,
                    help="Which CL phase position this eval is for (1..6 for ordering F).")
    ap.add_argument("--train-year", type=int, required=True,
                    help="The year that this phase trained on (= 2014+phase).")
    ap.add_argument("--n-rollouts", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[eval] loading TTT-shifted Qwen3-8B from {args.hf_ckpt}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(args.hf_ckpt)
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_ckpt, torch_dtype=torch.bfloat16, device_map=args.device,
    )
    model.eval()

    # Returns lookup
    rdf = pd.read_parquet(args.returns_table)
    rdf["filing_date"] = rdf["filing_date"].astype(str)
    rdf["cik"] = rdf["cik"].astype(str)
    returns_lookup = rdf.set_index(["cik", "filing_date"])[
        ["raw_return_30d", "ticker", "company"]
    ]

    val = pq.read_table(args.val_parquet).to_pylist()
    out_records = []
    print(f"[eval] {len(val)} val prompts × {args.n_rollouts} rollouts", file=sys.stderr)

    for idx, r in enumerate(val):
        prompt_msgs = r["prompt"]
        if hasattr(prompt_msgs, "tolist"):
            prompt_msgs = prompt_msgs.tolist()
        text = tok.apply_chat_template(
            [dict(m) for m in prompt_msgs],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = tok(text, return_tensors="pt").to(args.device)

        # Share prefill across n rollouts via num_return_sequences
        with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model.generate(
                **inputs,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_tokens,
                num_return_sequences=args.n_rollouts,
                pad_token_id=tok.eos_token_id,
            )
        rollout_texts = []
        prompt_len = inputs["input_ids"].shape[1]
        for s in out:
            new_ids = s[prompt_len:]
            rollout_texts.append(tok.decode(new_ids, skip_special_tokens=True))

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
            "method": "ttt",
            "phase": args.phase_idx,
            "train_year": args.train_year,
            "eval_year": args.eval_year,
            "ckpt_path": str(args.hf_ckpt),
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
