#!/usr/bin/env python3
"""Eval one (chain_phase, eval_set) cell of the TTT TemporalWiki matrix.

Loads the converted HF Qwen3-8B checkpoint and runs vanilla HF generate on
val_ts{1,2,3}.parquet (drift slices) or val_stable.parquet. Each row is a
(subject, relation) pair; gold is a short string answer; we score with the
EXACT SDPO/SFT/GRPO temporalwiki scorer (F1 ≥ 0.5 over article+punctuation
stripped tokens) — apples-to-apples with the existing twiki-cl_*_orderT_nothink_s42
baselines.
"""
from __future__ import annotations

import argparse, re, sys
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# SDPO's exact temporalwiki scorer (verbatim file copy at /workspace/_sync/sdpo_scorers/).
sys.path.insert(0, "/workspace/home/nayan/_sync")
from sdpo_scorers import temporalwiki as twiki_scorer  # noqa: E402


_NORMALIZE = (re.compile(r"\b(a|an|the)\b", re.IGNORECASE),
              str.maketrans("", "", "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"),
              re.compile(r"\s+"))


def _normalize(s: str) -> str:
    """Mirrors twiki_scorer._normalize for majority-voting (the scorer's
    own normalize is private; we replicate it here for the tie-break)."""
    arts, punct, ws = _NORMALIZE
    s = (s if isinstance(s, str) else str(s)).lower()
    s = arts.sub(" ", s)
    s = s.translate(punct)
    s = ws.sub(" ", s).strip()
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-ckpt", type=Path, required=True)
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
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[eval] loading TTT-shifted Qwen3-8B from {args.hf_ckpt}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(args.hf_ckpt)
    model = AutoModelForCausalLM.from_pretrained(args.hf_ckpt,
        torch_dtype=torch.bfloat16, device_map=args.device).eval()

    val = pq.read_table(args.val_parquet).to_pylist()
    out_records = []
    print(f"[eval] {len(val)} val prompts × {args.n_rollouts} rollouts (eval_set={args.eval_set})",
          file=sys.stderr)

    for idx, r in enumerate(val):
        prompt_msgs = r["prompt"]
        if hasattr(prompt_msgs, "tolist"): prompt_msgs = prompt_msgs.tolist()
        text = tok.apply_chat_template([dict(m) for m in prompt_msgs],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tok(text, return_tensors="pt").to(args.device)

        with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model.generate(**inputs, do_sample=True,
                temperature=args.temperature, top_p=args.top_p,
                max_new_tokens=args.max_tokens, num_return_sequences=args.n_rollouts,
                pad_token_id=tok.eos_token_id)
        prompt_len = inputs["input_ids"].shape[1]
        rollout_texts = [tok.decode(s[prompt_len:], skip_special_tokens=True) for s in out]
        gold = str(dict(r["reward_model"]).get("ground_truth", "")).strip()

        # Score each rollout with SDPO's exact temporalwiki scorer.
        rollout_results = [twiki_scorer.compute_score(t, gold) for t in rollout_texts]
        rollout_preds = [str(rr.get("pred", "")) for rr in rollout_results]
        rollout_f1s = [float(rr.get("score", 0.0)) for rr in rollout_results]
        rollout_accs = [float(rr.get("acc", 0.0)) for rr in rollout_results]
        score_mean_f1 = sum(rollout_f1s) / max(1, len(rollout_f1s))
        score_mean_acc = sum(rollout_accs) / max(1, len(rollout_accs))

        cnt = Counter([_normalize(p) for p in rollout_preds if p])
        pred_norm_majority = cnt.most_common(1)[0][0] if cnt else ""
        pred_text = next(
            (p for p in rollout_preds if _normalize(p) == pred_norm_majority),
            "",
        )
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
            "method": "ttt",
            "phase": args.phase_idx,
            "train_slice": args.train_slice,
            "eval_set": args.eval_set,
            "data_source": str(r.get("data_source", "")),
            "ckpt_path": str(args.hf_ckpt),
            "score_mean_f1_at_8": score_mean_f1,
            "score_mean_acc_at_8": score_mean_acc,
            "n_rollouts_correct": int(sum(int(a >= 1.0) for a in rollout_accs)),
            "raw_response_0": rollout_texts[0] if rollout_texts else "",
            "all_rollouts": "|".join(rollout_preds),
        })
        if (idx + 1) % 25 == 0:
            print(f"  {idx+1}/{len(val)} done", file=sys.stderr, flush=True)

    df = pd.DataFrame(out_records)
    df.to_parquet(args.out, index=False)
    acc_maj = float(df["hit_majority"].mean())
    score_mean = float(df["score_mean_at_8"].mean())
    print(f"[eval] wrote {args.out}  rows={len(df)}  acc_majority={acc_maj:.3f}  score_mean@8={score_mean:.3f}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
