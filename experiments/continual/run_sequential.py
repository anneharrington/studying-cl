#!/usr/bin/env python3
"""Sequential continual-learning driver: run one method × one ordering × one seed.

For each ordering (A/B/C) we launch three back-to-back verl training jobs, one per task.
Task i+1 starts from task i's final HF checkpoint (fresh python process ⇒ optimizer reset,
which matches the paper's protocol). Verl's native `data.val_files: [bio, finqa, tooluse]`
gives us per-source validation metrics every `trainer.test_freq` steps, so the cross-task
forgetting/transfer matrix falls out of the wandb stream without a bespoke eval loop.

Usage:
    python experiments/continual/run_sequential.py \\
        --method sdft --ordering A --seed 42 \\
        --model Qwen/Qwen3-8B --enable-thinking false \\
        --n-gpus 4 --nnodes 1

See experiments/continual/README for sbatch/sweep wrappers.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Task/ordering registry
# ---------------------------------------------------------------------------

SDPO_ROOT = Path(__file__).resolve().parents[2]

TASK_DATA: Dict[str, Dict] = {
    # key -> (train_parquet, test_parquet, test_subsample_n)
    "bio":     {"train": "datasets/sciknoweval/biology/train.parquet",
                "test":  "datasets/sciknoweval/biology/test.parquet",
                "test_n": 50},   # keep full 50 test rows
    "finqa":   {"train": "datasets/finqa/train.parquet",
                "test":  "datasets/finqa/test.parquet",
                "test_n": 128},  # subsample 1125 -> 128 to bound eval cost
    "tooluse": {"train": "datasets/tooluse/train.parquet",
                "test":  "datasets/tooluse/test.parquet",
                "test_n": 68},   # keep full 68 test rows
}

# Temporal-drift task family (TemporalWiki). Set CL_TEMPORAL_DRIFT_DATA to the directory
# produced by the dataset-prep script; entries here pin the per-slice file names + eval
# sizes that surface in val metric keys (val-core/<data_source>/acc/mean@N).
TEMPORAL_DRIFT_DATA_DIR = Path(os.environ.get("CL_TEMPORAL_DRIFT_DATA", "data/temporalwiki_drift"))
TASK_DATA_TEMPORAL_DRIFT: Dict[str, Dict] = {
    # 4-date / 3-pair chronological chain (Nov 2025 → Feb 2026). 3 sequential CL phases.
    "ts1":    {"train": "train_s1.parquet",     "test": "val_s1.parquet",     "test_n": 50},
    "ts2":    {"train": "train_s2.parquet",     "test": "val_s2.parquet",     "test_n": 50},
    "ts3":    {"train": "train_s3.parquet",     "test": "val_s3.parquet",     "test_n": 50},
    "stable": {"train": None,                   "test": "val_stable.parquet", "test_n": 50},
}

# Finance task family (10-K forward sentiment, JanosAudran/financial-reports-sec).
# Set CL_FINANCE_DATA to the per-year-prepped directory; each year is one CL phase
# (2015..2020 = 6 sequential phases) and the parquets are expected pre-sized.
FINANCE_DATA_DIR = Path(os.environ.get("CL_FINANCE_DATA", "data/finance_yearly"))
TASK_DATA_FINANCE: Dict[str, Dict] = {
    f"y{y}": {"train": f"train_y{y}.parquet", "test": f"val_y{y}.parquet", "test_n": 50}
    for y in (2015, 2016, 2017, 2018, 2019, 2020)
}

# ORDERINGS values are variable-length lists of task keys. The "core" family uses 3-task
# orderings (A/B/C); the temporal family is a single 4-slice chronological chain (T);
# the finance family is a 6-year chronological chain (F).
ORDERINGS: Dict[str, Tuple[str, ...]] = {
    "A": ("bio", "finqa", "tooluse"),
    "B": ("tooluse", "finqa", "bio"),
    "C": ("finqa", "bio", "tooluse"),
    "T": ("ts1", "ts2", "ts3"),
    "F": ("y2015", "y2016", "y2017", "y2018", "y2019", "y2020"),
}

# task key -> data_source string used in verl's val metrics (val-core/<source>/acc/mean@N).
# Diverges for bio (dataset is "sciknoweval"); finqa/tooluse are identity. Mirrors the same
# mapping in analyze.py (SOURCE_TO_TASK), kept here so the best-ckpt tracker can pick the
# right metric key per training task without importing analyze.
TASK_TO_SOURCE: Dict[str, str] = {
    "bio":     "sciknoweval",
    "finqa":   "finqa",
    "tooluse": "tooluse",
    # TemporalWiki drift family — slice-specific data_source values so each cell M[i,j] of
    # the 4×4 matrix lands as its own val-core/temporalwiki_drift_sJ/acc/mean@N key.
    "ts1":     "temporalwiki_drift_s1",
    "ts2":     "temporalwiki_drift_s2",
    "ts3":     "temporalwiki_drift_s3",
    "ts4":     "temporalwiki_drift_s4",
    "stable":  "temporalwiki_stable",
    # Finance family — per-year data_source so M[i,j] of the 6×6 matrix lands as its own
    # val-core/finance_yr_YYYY/acc/mean@N key. Matches feedback/__init__.py dispatcher
    # branch (data_source.startswith("finance_yr_") -> finance.compute_score).
    "y2015":   "finance_yr_2015",
    "y2016":   "finance_yr_2016",
    "y2017":   "finance_yr_2017",
    "y2018":   "finance_yr_2018",
    "y2019":   "finance_yr_2019",
    "y2020":   "finance_yr_2020",
}

# Method -> verl config name (under verl/trainer/config/).
# Each config composes `ppo_trainer + user` with method-specific deltas. Selecting the
# config at this level (rather than monkey-patching sdpo.yaml via overrides) keeps us
# faithful to each method's upstream recipe:
#   - sdft:  sdft.yaml          paper-exact SDFT hyperparameters.
#   - sdpo:  sdpo.yaml          upstream SDPO (rollout.n=8, mini=32, alpha=0.5, lr=1e-5).
#   - grpo:  baseline_grpo.yaml upstream GRPO (mini=8, rollout.n=8).
#   - sft:   sft.yaml           plain SFT as a special case of SDPO (alpha=0 forward-KL
#                               with one-hot gold teacher + use_gold_response=true; the
#                               KL collapses to NLL on gold tokens). Routes through
#                               main_ppo so val cadence matches the other methods.
METHOD_CONFIG: Dict[str, str] = {
    "sdft": "sdft",
    "sdpo": "sdpo",
    "grpo": "baseline_grpo",
    "sft":  "sft",
}

# Rollout KV-cache ceiling, mirrored from each method YAML's top-level `max_model_len`.
# rollout.yaml defaults `max_model_len: null`, which makes vLLM fall back to the model's
# native context (40960 for Qwen3-8B). That inflated KV footprint OOMs val_before_train
# at the per-task data sizes used here. Pinning rollout.max_model_len to the same value
# the trainer uses keeps vLLM's KV pool sized to what the data actually needs.
METHOD_MAX_MODEL_LEN: Dict[str, int] = {
    "sdft": 10240,   # sdft.yaml:18
    "sdpo": 18944,   # sdpo.yaml:11
    "grpo": 10240,   # baseline_grpo.yaml:9
    "sft":  10240,   # sft.yaml:30
}

# SFT now flows through main_ppo via sft.yaml (alpha=0 + teacher=gold +
# use_gold_response=true). The legacy fsdp_sft_trainer helpers below remain for
# reference and stay importable via the constants here.
SFT_METHODS: set = set()
SFT_EVAL_CONFIG = "baseline_grpo"


def method_overrides(method: str, args) -> List[str]:
    """Method-specific Hydra overrides on top of the shared config.

    SDPO and GRPO follow the upstream recipes from
    experiments/generalization/run_{sdpo,baseline_grpo}_all.sh. Both pin
    optim.lr=1e-5 and lr_warmup_steps=10, which
    sdpo.yaml/baseline_grpo.yaml omit; without these overrides the actor.yaml
    defaults (lr=1e-6, no warmup) would silently apply and bias the comparison.
    """
    if method == "sdft":
        # TemporalWiki drift has 1-3 token golds and noisier labels (edit-conflict
        # answers): lr=1e-5 collapses fast. Lower to 3e-6 to keep cumulative
        # displacement under the empirical collapse threshold for that data.
        if args.ordering == "T":
            return ["actor_rollout_ref.actor.optim.lr=3e-6"]
        return []
    if method == "sdpo":
        if args.ordering == "T":
            return ["actor_rollout_ref.actor.optim.lr=3e-6",
                    "actor_rollout_ref.actor.optim.lr_warmup_steps=10"]
        # sdpo.yaml already pins lr=1e-5; only warmup needs adding for parity with the
        # upstream run scripts.
        return ["actor_rollout_ref.actor.optim.lr_warmup_steps=10"]
    if method == "grpo":
        if args.ordering == "T":
            return ["actor_rollout_ref.actor.optim.lr=3e-6",
                    "actor_rollout_ref.actor.optim.lr_warmup_steps=10"]
        # baseline_grpo.yaml omits lr, which would let actor.yaml's default 1e-6 win;
        # pin 1e-5 to match the upstream GRPO recipe.
        return [
            "actor_rollout_ref.actor.optim.lr=1e-5",
            "actor_rollout_ref.actor.optim.lr_warmup_steps=10",
        ]
    if method == "sft":
        # SFT trains on bare-answer gold (~5-30 tokens); per-token gradient signal is
        # ~10x larger than SDFT/SDPO (which train on ~200+ token rollout golds). At
        # lr=1e-5 the model peaks early then cliffs hard. Scaling lr by ~1/10 keeps
        # cumulative parameter displacement under the empirical collapse threshold.
        #
        # SFT_T_LR overrides for ordering=T (paper TemporalWiki default 1e-6).
        # SFT_LR overrides for any other ordering (paper default 1e-6).
        # Both are needed because LoRA-SFT wants LR ~5e-4 across all orderings,
        # while full-FT SFT keeps 1e-6.
        if args.ordering == "T":
            _lr = os.environ.get("SFT_T_LR", "1e-6")
        else:
            _lr = os.environ.get("SFT_LR", "1e-6")
        return [f"actor_rollout_ref.actor.optim.lr={_lr}"]
    raise ValueError(f"unknown method {method!r}")


def _lora_overrides(method: str) -> List[str]:
    """LoRA-SFT only. Returns the verl overrides that flip SFT from full FT to
    a rank-r LoRA adapter; no-op for any other method.

    Activated by USE_LORA=1 with method=sft. Defaults: rank=32, alpha=16,
    target_modules=all-linear (standard PEFT for Qwen3-8B). Knobs:
      LORA_RANK=N      LoRA rank
      LORA_ALPHA=N     LoRA alpha (scaling)
      LORA_TARGETS=... CSV or 'all-linear'

    We only enable LoRA for SFT because that's the canonical PEFT baseline
    (IMPRINTBENCH "SFT + LoRA"). LoRA-SDFT / LoRA-SDPO / LoRA-GRPO are not
    standard and mixing in a LoRA flag with USE_LORA=1 on those methods would
    be misleading — so we explicitly no-op there even if the env var is set.
    """
    if os.environ.get("USE_LORA") != "1":
        return []
    if method != "sft":
        return []
    rank = os.environ.get("LORA_RANK", "32")
    alpha = os.environ.get("LORA_ALPHA", "16")
    targets = os.environ.get("LORA_TARGETS", "all-linear")
    return [
        f"actor_rollout_ref.model.lora_rank={rank}",
        f"actor_rollout_ref.model.lora_alpha={alpha}",
        f"actor_rollout_ref.model.target_modules={targets}",
        # CRITICAL: default load_format=dummy makes fsdp_workers.py:670 set
        # self.base_sync_done=False at init. The LoRA sync at fsdp_workers.py:748
        # only fires when `peft_config is not None AND base_sync_done`, so with
        # dummy, vLLM never receives the LoRA adapter — it serves the base model
        # while the FSDP training pass uses base+adapter. This produces a near-
        # 100% rollout-vs-training probability mismatch (training/rollout_probs_diff
        # _mean ~ 0.99), garbage advantages, and val-acc collapse after task 1.
        # safetensors makes vLLM load base on init AND keeps base_sync_done=True
        # so the LoRA adapter actually syncs each step.
        "actor_rollout_ref.rollout.load_format=safetensors",
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def subsample_parquet(src: Path, dst: Path, n: int, seed: int) -> int:
    """Deterministic seeded subsample to at most `n` rows. Returns actual rows written."""
    table = pq.read_table(src)
    total = len(table)
    if total <= n:
        # No need to resample; just materialize a copy so manifest.json can cite a single path.
        pq.write_table(table, dst)
        return total
    import numpy as np
    rng = np.random.default_rng(seed)
    idx = rng.permutation(total)[:n]
    idx.sort()  # preserve sharded reads; non-fatal if unsorted but friendlier to arrow
    sub = table.take(idx)
    pq.write_table(sub, dst)
    return n


def stage_sft_messages(src: Path, dst: Path) -> int:
    """Build a `messages` column for verl's MultiTurnSFTDataset from a PPO-format parquet.

    PPO parquets carry `prompt` (list of {role,content} dicts: system+user or just user) and
    `reward_model.ground_truth` (gold answer). MultiTurnSFTDataset trains on every assistant
    turn, so we append a single assistant message containing the gold answer. The user/system
    turns inherit zero loss-mask, the assistant turn inherits ones — i.e. plain CE on gold.
    Writes a new parquet with the messages column added. Returns row count.
    """
    import numpy as np
    import pyarrow as pa
    table = pq.read_table(src)
    df = table.to_pandas()

    def to_messages(row):
        prompt = row["prompt"]
        # numpy ndarray of dicts -> python list
        if isinstance(prompt, np.ndarray):
            prompt = prompt.tolist()
        msgs = [dict(m) for m in prompt]
        rm = row["reward_model"]
        if isinstance(rm, dict):
            gt = rm.get("ground_truth", "")
        else:
            gt = ""
        msgs.append({"role": "assistant", "content": str(gt)})
        return msgs

    df["messages"] = df.apply(to_messages, axis=1)
    out = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(out, dst)
    return len(df)


def latest_global_step_dir(ckpt_dir: Path) -> Path:
    """Return <ckpt_dir>/global_step_N for the largest N; raise if none found."""
    candidates = sorted(
        [p for p in ckpt_dir.glob("global_step_*") if p.is_dir()],
        key=lambda p: int(p.name.rsplit("_", 1)[1]),
    )
    if not candidates:
        raise FileNotFoundError(f"no global_step_* checkpoint under {ckpt_dir}")
    return candidates[-1]


# ---------------------------------------------------------------------------
# Best-checkpoint tracking
#
# verl has no native save_best. We save at every eval (save_freq == test_freq) with a
# rolling window of 2 (max_actor_ckpt_to_keep=2), and a background thread polls the
# metrics JSONL: each time val-core/<source>/acc/mean@N sets a new peak, we `cp -al`
# that step's hf_model dir into <ckpt_dir>/best_hf/. Hardlinks keep the files alive
# even after verl prunes the old global_step_*/ entries, so peak disk stays ~2 ckpts.
# ---------------------------------------------------------------------------


def _hardlink_swap(src_hf: Path, dst_hf: Path) -> None:
    """Atomically replace dst_hf with a hardlink-tree of src_hf. Zero storage cost."""
    staging = dst_hf.with_name(dst_hf.name + "_staging")
    if staging.exists():
        shutil.rmtree(staging)
    # cp -al: archive mode + hardlink every file. Preserves perms/owner, skips copy I/O.
    subprocess.check_call(["cp", "-al", str(src_hf), str(staging)])
    if dst_hf.exists() or dst_hf.is_symlink():
        shutil.rmtree(dst_hf, ignore_errors=True)
    staging.rename(dst_hf)


def _track_best_worker(
    metrics_path: Path,
    ckpt_dir: Path,
    best_hf: Path,
    source: str,
    stop_event: threading.Event,
    state: Dict,
    poll_s: float = 5.0,
) -> None:
    """Background poll: read new JSONL lines, promote new-best ckpt via hardlink swap.

    `state` is mutated in place (best_acc, best_step) so the main thread can inspect it
    post-join. Skips step 0 (val_before_train has no ckpt). If a new-best eval fires
    before its ckpt is on disk, we retry on subsequent polls until it lands.
    """
    offset = 0
    pending_best: Optional[Tuple[int, float]] = None  # (step, acc) waiting on ckpt
    while not stop_event.is_set():
        try:
            if metrics_path.exists():
                size = metrics_path.stat().st_size
                if size > offset:
                    with open(metrics_path, "rb") as fh:
                        fh.seek(offset)
                        chunk = fh.read(size - offset)
                    offset = size
                    for raw in chunk.splitlines():
                        if not raw.strip():
                            continue
                        try:
                            evt = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        step = int(evt.get("step", 0))
                        if step == 0:
                            continue
                        acc = _extract_source_acc(evt.get("data", {}), source)
                        if acc is None:
                            continue
                        if acc > state["best_acc"]:
                            pending_best = (step, acc)
            # Try to materialize the pending best (ckpt may have landed after the metric).
            if pending_best is not None:
                step, acc = pending_best
                src_hf = ckpt_dir / f"global_step_{step}" / "actor" / "huggingface"
                if src_hf.exists() and any(src_hf.iterdir()):
                    _hardlink_swap(src_hf, best_hf)
                    # LoRA chain: also mirror the step's lora_adapter dir to
                    # best_adapter (sibling of best_hf) so the next task can
                    # load it via actor_rollout_ref.model.lora_adapter_path.
                    # Verl actually writes the adapter under <step>/actor/
                    # (the actor's local_path), NOT directly under <step>/.
                    # If LoRA wasn't engaged, this no-ops (src_adapter absent).
                    src_adapter = ckpt_dir / f"global_step_{step}" / "actor" / "lora_adapter"
                    if src_adapter.exists() and any(src_adapter.iterdir()):
                        _hardlink_swap(src_adapter, ckpt_dir / "best_adapter")
                    state["best_acc"] = acc
                    state["best_step"] = step
                    pending_best = None
        except Exception as exc:  # never let the tracker crash the run
            state.setdefault("errors", []).append(f"{dt.datetime.utcnow().isoformat()}Z  {exc!r}")
        if stop_event.wait(poll_s):
            break


def _extract_source_acc(data: Dict, source: str) -> Optional[float]:
    """Return the val-core/<source>/acc/mean@N value if present in a metrics record."""
    key_prefix = f"val-core/{source}/acc/mean@"
    for k, v in data.items():
        if k.startswith(key_prefix):
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
    return None


def _finalize_task_ckpts(ckpt_dir: Path, best_hf: Path, final_hf: Path) -> None:
    """End-of-task cleanup: hardlink the final ckpt, then strip all rolling global_step_*.

    After this runs, only best_hf/ and final_hf/ survive under ckpt_dir. If best was
    never promoted (e.g. zero-shot beat every training eval), fall back to final as best.
    """
    last_step_dir = latest_global_step_dir(ckpt_dir)
    final_src = last_step_dir / "actor" / "huggingface"
    _hardlink_swap(final_src, final_hf)
    if not best_hf.exists():
        _hardlink_swap(final_src, best_hf)
    # LoRA chain: preserve the final-step lora_adapter as final_adapter (sibling
    # of final_hf). Mirror to best_adapter if best was never promoted. Skipped
    # cleanly when LoRA wasn't engaged (src dir absent).
    final_adapter_src = last_step_dir / "actor" / "lora_adapter"
    if final_adapter_src.exists() and any(final_adapter_src.iterdir()):
        _hardlink_swap(final_adapter_src, ckpt_dir / "final_adapter")
        if not (ckpt_dir / "best_adapter").exists():
            _hardlink_swap(final_adapter_src, ckpt_dir / "best_adapter")
    for gs_dir in ckpt_dir.glob("global_step_*"):
        if gs_dir.is_dir():
            shutil.rmtree(gs_dir, ignore_errors=True)


def hf_ckpt_complete(hf_dir: Path) -> bool:
    """True if hf_dir has a fully materialized HF checkpoint (config + at least one shard)."""
    if not hf_dir.is_dir():
        return False
    if not (hf_dir / "config.json").exists():
        return False
    has_shard = any(hf_dir.glob("model*.safetensors")) or any(hf_dir.glob("pytorch_model*.bin"))
    return has_shard


def git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True,
        ).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Launch one task block
# ---------------------------------------------------------------------------

def launch_task_block(
    args,
    task_dir: Path,                # where ckpts + JSONL live (chain-scoped)
    metrics_path: Path,             # verl's FileLogger target for this task
    task_idx: int,                  # 1-based
    task_name: str,
    model_path: str,
    train_file: Path,
    val_files: List[Path],
    log_file: Path,
    experiment_name: str,
) -> Tuple[Path, Path]:
    """Launch verl for one task. Returns (best_hf, final_hf) HF ckpt paths.

    Best tracked online via a background thread that hardlinks the running-best
    global_step_* into <ckpt_dir>/best_hf/; see _track_best_worker for invariants.
    """
    config_name = METHOD_CONFIG[args.method]
    ckpt_dir = task_dir / f"task{task_idx}_{task_name}_ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_hf = ckpt_dir / "best_hf"
    final_hf = ckpt_dir / "final_hf"

    # Resume short-circuit: if a previous run already produced a complete final_hf for
    # this task, skip the launch and reuse the existing ckpts. We require both best_hf
    # and final_hf to be complete; if best_hf is missing (e.g. interrupted finalize),
    # mirror final_hf into best_hf so downstream chain_best can keep going.
    if args.resume and hf_ckpt_complete(final_hf):
        if not hf_ckpt_complete(best_hf):
            print(f"[resume] {task_dir.name}/task{task_idx}_{task_name}: best_hf missing, mirroring from final_hf")
            _hardlink_swap(final_hf, best_hf)
        print(f"\n=== [{task_dir.name}] task {task_idx}/{len(args.ordering_tasks)} {task_name}  RESUME-SKIP (final_hf exists) ===")
        return best_hf, final_hf

    # Derive step counts.
    # train_samples / batch -> 1-epoch step count; multiply by total_epochs for full schedule.
    # test_freq=1 -> eval + save at every step (densest learning curve, exact best-ckpt
    # granularity, ~1.5-2x walltime vs test_freq=3). --eval-every-prompts is retained as
    # a manifest field for provenance but no longer drives cadence.
    steps_per_epoch = max(1, (args.train_samples + args.train_batch - 1) // args.train_batch)
    total_steps = steps_per_epoch * args.total_epochs
    # test_freq controls the eval+save cadence inside verl. Default 1 = densest curve.
    # For initial smokes, --test-freq 0 means "only eval at the end" (we set it to total_steps
    # so verl fires exactly once at the final step; val_before_train=true still gives step 0).
    if args.test_freq == 0:
        test_freq = total_steps
    else:
        test_freq = args.test_freq

    # LoRA chain handoff (Option A): a single adapter accumulates across tasks.
    # For task >= 2:
    #  - previous task's ckpt_dir holds best_adapter/ (chain_best) or
    #    final_adapter/ (chain_final), sibling of the HF model dir we got
    #    as model_path. We load it via lora_adapter_path so verl continues
    #    training that adapter instead of starting fresh.
    #  - we OVERRIDE model_path to the ORIGINAL base (args.model) instead of
    #    the previous task's saved hf_model. Reason: when LoRA is on, the saved
    #    hf_model is in PEFT format (state-dict keys prefixed with `base_model.`)
    #    which vLLM can't load as plain Qwen3 (raises "no module 'base_model'").
    #    Pointing model.path at the original Qwen3-8B base gives vLLM a clean
    #    HF model to load, and the LoRA adapter is layered on via PEFT.
    lora_adapter_path = None
    if (
        args.method == "sft"
        and os.environ.get("USE_LORA") == "1"
        and task_idx > 1
    ):
        prev_model_dir = Path(model_path)  # = <prev_task_ckpts>/{best_hf,final_hf}
        adapter_name = "best_adapter" if prev_model_dir.name == "best_hf" else "final_adapter"
        candidate = prev_model_dir.parent / adapter_name
        if candidate.exists() and any(candidate.iterdir()):
            lora_adapter_path = str(candidate)
            # Redirect model_path back to the original base so vLLM can load
            # a clean Qwen3 model. The adapter carries the chain state.
            model_path = args.model

    overrides = [
        f"--config-name={config_name}",
        # -- data ----
        f"data.train_files=[{train_file}]",
        "data.val_files=[" + ",".join(str(p) for p in val_files) + "]",
        f"data.train_batch_size={args.train_batch}",
        f"data.apply_chat_template_kwargs.enable_thinking={str(args.enable_thinking).lower()}",
        # -- model ----
        f"actor_rollout_ref.model.path={model_path}",
        f"critic.model.path={model_path}",
        # LoRA chain handoff: for task >= 2, point at the previous task's
        # accumulating adapter. No-op for task 1 / non-LoRA / missing dir.
        *([f"actor_rollout_ref.model.lora_adapter_path={lora_adapter_path}"]
          if lora_adapter_path else []),
        # user.yaml's custom_reward_function.path hardcodes /users/$USER/... — override to our tree.
        f"custom_reward_function.path={SDPO_ROOT}/engine/verl/utils/reward_score/feedback/__init__.py",
        # -- rollout ----
        f"actor_rollout_ref.rollout.val_kwargs.n={args.val_n}",
        # Cap vLLM KV to the method's trainer-side max_model_len (see METHOD_MAX_MODEL_LEN).
        # Without this, vLLM infers Qwen3-8B's 40960 native ctx and KV during val_before_train
        # (246 prompts x val_kwargs.n rollouts) overruns the 80 GB H100.
        # Ordering F (finance 10-K) needs ~13.3k prompt + 256 response = ~13.6k; bump cap to
        # 16640 to give headroom for chat-template tokens. Other orderings keep their per-method
        # default since their prompts are <2k tokens.
        f"actor_rollout_ref.rollout.max_model_len="
        f"{16640 if args.ordering == 'F' else METHOD_MAX_MODEL_LEN[args.method]}",
        # Leave headroom for FSDP actor weights + optimizer on the same cards (rollout default
        # is 0.5; with 8-GPU FSDP of Qwen3-8B the actor side is ~12 GB/GPU, 0.45 is the safe
        # ceiling that still gives vLLM ~30 GB for weights+KV+graphs).
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.45",
        # -- trainer ----
        f"trainer.project_name=sdpo_seq",
        f"trainer.experiment_name={experiment_name}",
        f"trainer.group_name={args.run_tag}",
        f"trainer.default_local_dir={ckpt_dir}",
        f"trainer.total_training_steps={total_steps}",
        "trainer.total_epochs=999",  # cap is via total_training_steps, not epochs
        f"trainer.test_freq={test_freq}",
        f"trainer.save_freq={test_freq}",       # save at every eval so best-tracker has a ckpt per val point
        "trainer.val_before_train=true",
        f"trainer.n_gpus_per_node={args.n_gpus}",
        f"trainer.nnodes={args.nnodes}",
        "trainer.max_actor_ckpt_to_keep=2",      # rolling window: tracker runs every 5s, 2 is enough slack
        "trainer.resume_mode=disable",           # we control checkpointing externally per task; no mid-task auto-resume
        "trainer.logger=[wandb,console,file]",
        # Save only HF-format weights (~16 GB) instead of [model, hf_model, extra]
        # (~62 GB with FSDP shards + optim state). The next task only needs HF
        # weights for model.path handoff; we don't resume mid-task.
        "actor_rollout_ref.actor.checkpoint.save_contents=[hf_model]",
        # -- seed ----
        f"data.shuffle=true",
        # Override CSCS-specific user.yaml vars with absolute paths we control (replaces ${oc.env:USER/TASK}).
        f"vars.dir={SDPO_ROOT}",
        f"vars.task={task_name}",
        f"vars.ckpt_dir={task_dir}",
        f"vars.log_dir={task_dir}/logs",
        f"+trainer.seed={args.seed}",
    ]
    overrides.extend(method_overrides(args.method, args))
    overrides.extend(_lora_overrides(args.method))

    # Ordering-F (finance 10-K) prompt/response budget. The 50k-char filing excerpts
    # land around ~13.3k tokens, so a 16384 prompt cap fits with margin. Responses are
    # a single word (up/down), 256 is plenty. Must mirror rollout.max_model_len=16640.
    if args.ordering == "F":
        overrides.extend([
            "data.max_prompt_length=16384",
            "data.max_response_length=256",
        ])

    cmd = [
        sys.executable, "-m", "verl.trainer.main_ppo",
        *overrides,
    ]

    log_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n=== [{task_dir.name}] task {task_idx}/{len(args.ordering_tasks)} {task_name}  (steps={total_steps}, test_freq={test_freq}) ===")
    print("CMD:", " ".join(shlex.quote(c) for c in cmd))
    print(f"LOG:     {log_file}")
    print(f"METRICS: {metrics_path}")

    if args.dry_run:
        return (ckpt_dir / "best_hf", ckpt_dir / "final_hf")

    env = os.environ.copy()
    # Slurm sets both ROCR_VISIBLE_DEVICES (AMD) and CUDA_VISIBLE_DEVICES (NVIDIA) inside
    # the allocation; verl.single_controller rejects the pair. We only use NVIDIA GPUs, so
    # drop the AMD var before Ray inherits this env.
    env.pop("ROCR_VISIBLE_DEVICES", None)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("WANDB_DIR", str(task_dir / "wandb"))
    env.setdefault("USER", os.environ.get("USER", ""))
    env.setdefault("EXPERIMENT", experiment_name)
    env.setdefault("TASK", task_name)
    env["VERL_FILE_LOGGER_PATH"] = str(metrics_path)

    # Per-task RAY_TMPDIR so each task starts with a fresh Ray socket directory
    # and can't collide with leftover state from a prior task. We deliberately
    # avoid issuing active cleanup commands here (they interfere with vLLM/Ray's
    # own teardown).
    slurm_job = os.environ.get("SLURM_JOB_ID", "local")
    env["RAY_TMPDIR"] = f"/tmp/ray_{slurm_job}_task{task_idx}_{task_name}"

    # Start the best-ckpt tracker before launching verl so we never miss an early peak.
    source = TASK_TO_SOURCE[task_name]
    stop_event = threading.Event()
    state: Dict = {"best_acc": -float("inf"), "best_step": None}
    tracker = threading.Thread(
        target=_track_best_worker,
        args=(metrics_path, ckpt_dir, best_hf, source, stop_event, state),
        daemon=True,
    )
    tracker.start()

    try:
        with open(log_file, "w") as fh:
            proc = subprocess.run(
                cmd, cwd=SDPO_ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env,
            )
    finally:
        # Give the tracker ~2 polls to drain the tail of the JSONL and promote any trailing best.
        time.sleep(6)
        stop_event.set()
        tracker.join(timeout=30)

    if proc.returncode != 0:
        print(f"!! task {task_idx} ({task_name}) failed with code {proc.returncode}; see {log_file}")
        raise SystemExit(proc.returncode)

    _finalize_task_ckpts(ckpt_dir, best_hf, final_hf)
    best_step = state.get("best_step")
    if best_step is None:
        print(f"   best_hf  <- final (no training eval beat zero-shot)  final_hf={final_hf}")
    else:
        print(f"   best_hf  <- step {best_step} acc={state['best_acc']:.3f}   final_hf={final_hf}")
    if state.get("errors"):
        print(f"   tracker warnings: {len(state['errors'])}  last={state['errors'][-1]}")
    return best_hf, final_hf


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_run_dir(args) -> Path:
    # sweep_tag (optional) disambiguates successive reruns of the same (method,ordering,seed)
    # so wandb doesn't collapse them into "_1, _2" suffixes and run_dirs don't collide.
    base = (
        f"{args.method}_order{args.ordering}"
        f"_{'think' if args.enable_thinking else 'nothink'}"
        f"_s{args.seed}"
    )
    args.run_tag = f"{args.sweep_tag}_{base}" if args.sweep_tag else base
    run_dir = Path(args.out_root) / args.run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    return run_dir


def stage_data(args, run_dir: Path) -> Tuple[List[Path], List[Path]]:
    """Write per-run subsampled train and val parquets; return (train_files, val_files).

    With --resume, existing parquets are reused (subsample is deterministic given seed,
    but copying again wastes IO and could risk a partial write if interrupted).

    Dispatches by task family. The "core" family (orderings A/B/C) subsamples from the
    bio/finqa/tooluse source datasets. The "temporal_drift" family (ordering T) consumes
    parquets pre-staged by scripts/prep_temporalwiki_drift.py — they're already at the
    right size and per-row keys are aligned across slices, so we just copy/symlink them.
    """
    if args.ordering == "T":
        return _stage_data_temporal_drift(args, run_dir)
    if args.ordering == "F":
        return _stage_data_finance(args, run_dir)

    train_files: List[Path] = []
    for i, task in enumerate(args.ordering_tasks, start=1):
        src = SDPO_ROOT / TASK_DATA[task]["train"]
        dst = run_dir / f"task{i}_{task}_train.parquet"
        if args.resume and dst.exists():
            print(f"  train[{task}]: reuse existing {dst.name}")
        else:
            n = subsample_parquet(src, dst, args.train_samples, seed=args.seed + i)
            print(f"  train[{task}]: {n} rows -> {dst.name}")
        train_files.append(dst)

    val_files: List[Path] = []
    for task, meta in TASK_DATA.items():
        src = SDPO_ROOT / meta["test"]
        dst = run_dir / f"val_{task}.parquet"
        if args.resume and dst.exists():
            print(f"  val[{task}]:   reuse existing {dst.name}")
        else:
            n = subsample_parquet(src, dst, meta["test_n"], seed=args.seed + 1000)
            print(f"  val[{task}]:   {n} rows -> {dst.name}")
        val_files.append(dst)
    return train_files, val_files


def _stage_data_temporal_drift(args, run_dir: Path) -> Tuple[List[Path], List[Path]]:
    """Stage TemporalWiki drift parquets into the run dir.

    Source directory is `TEMPORAL_DRIFT_DATA_DIR` (set CL_TEMPORAL_DRIFT_DATA env
    var). Writes into run_dir:
        task{i}_{task}_train.parquet  for each chronological slice
        val_{task}.parquet            for every slice + stable
        temporal_manifest.json        (slice -> date-label mapping)
    """
    import shutil

    src_root = TEMPORAL_DRIFT_DATA_DIR
    if not src_root.exists():
        raise FileNotFoundError(
            f"Temporal-drift data not staged at {src_root}. "
            f"Set CL_TEMPORAL_DRIFT_DATA to a directory containing the per-slice parquets."
        )

    # Copy the prep manifest into the run for downstream plotting / labeling.
    prep_manifest_src = src_root / "manifest.json"
    if prep_manifest_src.exists():
        shutil.copy(prep_manifest_src, run_dir / "temporal_manifest.json")
        print(f"  temporal_manifest.json copied (slice→date labels for plotting)")

    train_files: List[Path] = []
    for i, task in enumerate(args.ordering_tasks, start=1):
        meta = TASK_DATA_TEMPORAL_DRIFT[task]
        src = src_root / meta["train"]
        dst = run_dir / f"task{i}_{task}_train.parquet"
        if args.resume and dst.exists():
            print(f"  train[{task}]: reuse existing {dst.name}")
        else:
            shutil.copy(src, dst)
            print(f"  train[{task}]: {meta['train']} -> {dst.name}")
        train_files.append(dst)

    val_files: List[Path] = []
    for task, meta in TASK_DATA_TEMPORAL_DRIFT.items():
        src = src_root / meta["test"]
        dst = run_dir / f"val_{task}.parquet"
        if args.resume and dst.exists():
            print(f"  val[{task}]:   reuse existing {dst.name}")
        else:
            shutil.copy(src, dst)
            print(f"  val[{task}]:   {meta['test']} -> {dst.name}")
        val_files.append(dst)
    return train_files, val_files


def _stage_data_finance(args, run_dir: Path) -> Tuple[List[Path], List[Path]]:
    """Stage finance per-year parquets into the run dir.

    Source directory is `FINANCE_DATA_DIR` (set CL_FINANCE_DATA env var). Writes:
        run_dir/task{i}_y{YYYY}_train.parquet  for each chronological year
        run_dir/val_y{YYYY}.parquet            for each year
        run_dir/finance_manifest.json          (year metadata, label up-fractions)
    """
    src_root = FINANCE_DATA_DIR
    if not src_root.exists():
        raise FileNotFoundError(
            f"Finance data not staged at {src_root}. "
            f"Set CL_FINANCE_DATA to a directory containing the per-year parquets."
        )

    prep_manifest_src = src_root / "manifest.json"
    if prep_manifest_src.exists():
        shutil.copy(prep_manifest_src, run_dir / "finance_manifest.json")
        print(f"  finance_manifest.json copied (year metadata for plotting)")

    train_files: List[Path] = []
    for i, task in enumerate(args.ordering_tasks, start=1):
        meta = TASK_DATA_FINANCE[task]
        src = src_root / meta["train"]
        dst = run_dir / f"task{i}_{task}_train.parquet"
        if args.resume and dst.exists():
            print(f"  train[{task}]: reuse existing {dst.name}")
        else:
            shutil.copy(src, dst)
            print(f"  train[{task}]: {meta['train']} -> {dst.name}")
        train_files.append(dst)

    val_files: List[Path] = []
    for task, meta in TASK_DATA_FINANCE.items():
        src = src_root / meta["test"]
        dst = run_dir / f"val_{task}.parquet"
        if args.resume and dst.exists():
            print(f"  val[{task}]:   reuse existing {dst.name}")
        else:
            shutil.copy(src, dst)
            print(f"  val[{task}]:   {meta['test']} -> {dst.name}")
        val_files.append(dst)
    return train_files, val_files


def stage_sft_train_files(args, run_dir: Path, train_files: List[Path]) -> List[Path]:
    """For SFT: build per-task parquets with a `messages` column appended (system+user+assistant).

    These are siblings of the PPO-format train_files (e.g. taskN_<task>_train_sft.parquet).
    Returned in the same order as train_files so callers can index by task position.
    """
    out: List[Path] = []
    for i, (task, src) in enumerate(zip(args.ordering_tasks, train_files), start=1):
        dst = run_dir / f"task{i}_{task}_train_sft.parquet"
        if args.resume and dst.exists():
            print(f"  sft-train[{task}]: reuse existing {dst.name}")
        else:
            n = stage_sft_messages(src, dst)
            print(f"  sft-train[{task}]: {n} rows -> {dst.name}")
        out.append(dst)
    return out


# ---------------------------------------------------------------------------
# SFT launch (verl FSDP SFT trainer) + eval-only pass
# ---------------------------------------------------------------------------

def launch_sft_task_block(
    args,
    task_dir: Path,
    task_idx: int,
    task_name: str,
    model_path: str,
    sft_train_file: Path,
    log_file: Path,
    experiment_name: str,
) -> Path:
    """Train one SFT task via verl.trainer.fsdp_sft_trainer (torchrun).

    Returns the HF checkpoint path produced at end-of-training. SFT trainer saves to
    <ckpt_dir>/global_step_<N>/huggingface/, so we promote that to <ckpt_dir>/final_hf
    via a hardlink swap (consistent with PPO's <step>/actor/huggingface convention).

    No best-ckpt tracking: SFT has loss-only val (no rollouts), so "best" is just the
    final-step checkpoint. We post-hoc mirror final_hf -> best_hf so analyze.py works.
    """
    ckpt_dir = task_dir / f"task{task_idx}_{task_name}_ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final_hf = ckpt_dir / "final_hf"
    best_hf = ckpt_dir / "best_hf"

    if args.resume and hf_ckpt_complete(final_hf):
        print(f"\n=== [{task_dir.name}] task {task_idx}/{len(args.ordering_tasks)} {task_name}  RESUME-SKIP (final_hf exists) ===")
        if not hf_ckpt_complete(best_hf):
            _hardlink_swap(final_hf, best_hf)
        return final_hf

    steps_per_epoch = max(1, (args.train_samples + args.train_batch - 1) // args.train_batch)
    total_steps = steps_per_epoch * args.total_epochs

    # micro_batch_size_per_gpu: with train_batch=32 / 8 GPUs = 4 examples per GPU per step
    # at grad_accum=1. Match SFT default of 4. max_length 2048 covers the longest finqa
    # contexts after truncation.
    micro = max(1, args.train_batch // args.n_gpus)

    overrides = [
        f"data.train_files=[{sft_train_file}]",
        f"data.val_files=[{sft_train_file}]",  # loss-only val; reuse train as cheap stand-in
        f"data.train_batch_size={args.train_batch}",
        f"data.micro_batch_size_per_gpu={micro}",  # field exists in sft_trainer.yaml, no `+` needed
        "data.multiturn.enable=true",
        "data.multiturn.messages_key=messages",
        f"data.max_length={args.sft_max_length}",
        "data.truncation=right",
        # Qwen3's chat template emits different tokens for the *last* assistant turn vs intermediate
        # ones (even with enable_thinking=false), so per-message-tokenize-then-concat ≠ one-shot
        # tokenize. The actual training tokens we use are the concatenated ones (correct for SFT
        # loss); the assertion just guards against silently-misaligned templates. Bypass it.
        "+data.ignore_input_ids_mismatch=true",
        f"+data.apply_chat_template_kwargs.enable_thinking={str(args.enable_thinking).lower()}",
        f"model.partial_pretrain={model_path}",
        # Match SDFT/SDPO/GRPO learning rate so cross-method comparison has zero LR confound.
        "optim.lr=1e-5",
        f"trainer.project_name=sdpo_seq",
        f"trainer.experiment_name={experiment_name}",
        f"+trainer.group_name={args.run_tag}",
        f"trainer.default_local_dir={ckpt_dir}",
        # CRITICAL: SFT trainer's LR scheduler uses self.total_steps = steps_per_epoch * total_epochs
        # (fsdp_sft_trainer.py:351), NOT total_training_steps. Setting total_epochs=999 caps the
        # train loop via total_training_steps but stretches warmup to ~1500 steps, which means
        # 32 steps never escape warmup -> effective lr ~50x too small. Set total_epochs to match
        # what the user actually wants (matches PPO sweep's --total-epochs).
        f"trainer.total_epochs={args.total_epochs}",
        "trainer.test_freq=-1",      # skip loss-val during training
        f"trainer.save_freq={total_steps}",  # save once at end (is_last_step also forces save)
        f"trainer.n_gpus_per_node={args.n_gpus}",
        f"trainer.nnodes={args.nnodes}",
        f"trainer.seed={args.seed}",  # sft_trainer.yaml already has trainer.seed:1, no `+` prefix
        "trainer.resume_mode=disable",  # we control checkpointing externally per task
        # save_contents includes hf_model so the next task can use model.partial_pretrain;
        # trimmed to hf_model only (see run_method_ppo override for rationale).
        "trainer.checkpoint.save_contents=[hf_model]",
        "trainer.logger=[console,file]",  # wandb attribution is via env (WANDB_*)
    ]

    cmd = [
        "torchrun",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={args.n_gpus}",
        "-m", "verl.trainer.fsdp_sft_trainer",
        f"--config-name={METHOD_CONFIG['sft']}",
        *overrides,
    ]

    log_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n=== [{task_dir.name}] task {task_idx}/{len(args.ordering_tasks)} {task_name}  SFT (steps={total_steps}) ===")
    print("CMD:", " ".join(shlex.quote(c) for c in cmd))
    print(f"LOG:     {log_file}")

    if args.dry_run:
        return final_hf

    env = os.environ.copy()
    env.pop("ROCR_VISIBLE_DEVICES", None)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("WANDB_DIR", str(task_dir / "wandb"))
    env.setdefault("USER", os.environ.get("USER", ""))

    with open(log_file, "w") as fh:
        proc = subprocess.run(cmd, cwd=SDPO_ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env)
    if proc.returncode != 0:
        print(f"!! sft task {task_idx} ({task_name}) failed code={proc.returncode}; see {log_file}")
        raise SystemExit(proc.returncode)

    # FSDP SFT trainer writes <ckpt_dir>/global_step_<N>/huggingface (no `actor/` intermediate).
    final_src = latest_global_step_dir(ckpt_dir) / "huggingface"
    if not hf_ckpt_complete(final_src):
        raise FileNotFoundError(f"sft hf ckpt missing: {final_src}")
    _hardlink_swap(final_src, final_hf)
    _hardlink_swap(final_src, best_hf)  # best == final for SFT (no rollout-based picker)
    for gs_dir in ckpt_dir.glob("global_step_*"):
        if gs_dir.is_dir():
            shutil.rmtree(gs_dir, ignore_errors=True)
    print(f"   final_hf={final_hf}  (best_hf mirrored from final)")
    return final_hf


def launch_eval_only_block(
    args,
    task_dir: Path,
    metrics_path: Path,
    model_path: str,
    val_files: List[Path],
    log_file: Path,
    experiment_name: str,
    label_step: int,
) -> None:
    """Run a single rollout-based val pass on `model_path` and write ONE row to metrics_path.

    Uses verl.trainer.main_ppo with --config-name=sdft (cheap KV ceiling, val_kwargs.n=8) plus
    `total_training_steps=0 trainer.val_only=true trainer.val_before_train=true`. The trainer
    runs validation, logs once at global_steps=0, then exits. We then rewrite that single row
    with `step=label_step` and *append* it to metrics_path (so multiple eval passes for the
    same task can coexist — analyze.py picks first/last by step order).
    """
    # Per-eval-pass output JSONL.
    eval_jsonl = task_dir / "logs" / f"_eval_{experiment_name}.jsonl"
    eval_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if eval_jsonl.exists():
        eval_jsonl.unlink()

    overrides = [
        f"--config-name={SFT_EVAL_CONFIG}",
        # data: train_files is required by the loader but unused (we run val_only)
        f"data.train_files=[{val_files[0]}]",  # any parquet; never iterated
        "data.val_files=[" + ",".join(str(p) for p in val_files) + "]",
        f"data.train_batch_size={args.train_batch}",
        f"data.apply_chat_template_kwargs.enable_thinking={str(args.enable_thinking).lower()}",
        f"actor_rollout_ref.model.path={model_path}",
        f"critic.model.path={model_path}",
        f"custom_reward_function.path={SDPO_ROOT}/engine/verl/utils/reward_score/feedback/__init__.py",
        f"actor_rollout_ref.rollout.val_kwargs.n={args.val_n}",
        f"actor_rollout_ref.rollout.max_model_len={METHOD_MAX_MODEL_LEN['sdft']}",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.45",
        f"trainer.project_name=sdpo_seq",
        f"trainer.experiment_name={experiment_name}",
        f"trainer.group_name={args.run_tag}",
        f"trainer.default_local_dir={task_dir}/_eval_scratch",
        # total_training_steps>0 so cosine LR scheduler init doesn't divide by zero. The
        # training loop never runs because val_only=true returns right after the first val.
        "trainer.total_training_steps=1",
        "trainer.total_epochs=999",
        "trainer.test_freq=1",
        "trainer.save_freq=-1",
        "trainer.val_before_train=true",
        "trainer.val_only=true",  # field already in ppo_trainer.yaml; no `+` prefix
        f"trainer.n_gpus_per_node={args.n_gpus}",
        f"trainer.nnodes={args.nnodes}",
        "trainer.logger=[console,file]",
        "actor_rollout_ref.actor.checkpoint.save_contents=[model]",
        f"vars.dir={SDPO_ROOT}",
        f"vars.task=eval",
        f"vars.ckpt_dir={task_dir}",
        f"vars.log_dir={task_dir}/logs",
        f"+trainer.seed={args.seed}",
    ]

    cmd = [sys.executable, "-m", "verl.trainer.main_ppo", *overrides]
    log_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n--- [{task_dir.name}] EVAL_ONLY {experiment_name}  step={label_step} ---")
    print("CMD:", " ".join(shlex.quote(c) for c in cmd))
    print(f"LOG:     {log_file}")
    print(f"EVAL OUT:{eval_jsonl}")

    if args.dry_run:
        return

    env = os.environ.copy()
    env.pop("ROCR_VISIBLE_DEVICES", None)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("WANDB_DIR", str(task_dir / "wandb"))
    env.setdefault("USER", os.environ.get("USER", ""))
    env["VERL_FILE_LOGGER_PATH"] = str(eval_jsonl)

    with open(log_file, "w") as fh:
        proc = subprocess.run(cmd, cwd=SDPO_ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env)
    if proc.returncode != 0:
        print(f"!! eval_only {experiment_name} failed code={proc.returncode}; see {log_file}")
        raise SystemExit(proc.returncode)

    # Read the single row written by FileLogger and append to metrics_path with relabeled step.
    if not eval_jsonl.exists():
        print(f"!! eval_only produced no output JSONL: {eval_jsonl}")
        raise SystemExit(1)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_jsonl, "rb") as src, open(metrics_path, "ab") as dst:
        for raw in src:
            if not raw.strip():
                continue
            evt = json.loads(raw)
            evt["step"] = label_step
            dst.write(json.dumps(evt).encode() + b"\n")
    # Cleanup scratch ckpt dir (val_only never writes weights, but the dir gets created).
    shutil.rmtree(task_dir / "_eval_scratch", ignore_errors=True)
    print(f"   appended 1 eval row (step={label_step}) to {metrics_path}")


def _base_manifest(args, train_files: List[Path], val_files: List[Path]) -> Dict:
    """Fields shared by the master manifest and each per-chain pseudo-manifest."""
    return {
        "run_tag": args.run_tag,
        "sweep_tag": args.sweep_tag,
        "method": args.method,
        "ordering": args.ordering,
        "ordering_tasks": list(args.ordering_tasks),
        "model": args.model,
        "enable_thinking": args.enable_thinking,
        "seed": args.seed,
        "train_samples_per_task": args.train_samples,
        "total_epochs": args.total_epochs,
        "eval_every_prompts": args.eval_every_prompts,
        "train_batch_size": args.train_batch,
        "val_rollouts_per_prompt": args.val_n,
        "n_gpus_per_node": args.n_gpus,
        "nnodes": args.nnodes,
        "train_files": [str(p) for p in train_files],
        "val_files": [str(p) for p in val_files],
        "sdpo_commit": git_commit(SDPO_ROOT),
        "started_at": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "cli": " ".join(shlex.quote(a) for a in sys.argv),
    }


def write_manifest(args, run_dir: Path, train_files, val_files) -> None:
    manifest = _base_manifest(args, train_files, val_files)
    manifest["chains"] = ["chain_best", "chain_final"]
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    print(f"  manifest -> {path}")


def write_chain_manifest(args, chain_dir: Path, chain_name: str,
                          train_files: List[Path], val_files: List[Path]) -> None:
    """Per-chain pseudo-run manifest; lets analyze.py treat each chain dir as a run."""
    manifest = _base_manifest(args, train_files, val_files)
    manifest["chain"] = chain_name  # "best" or "final"
    manifest["run_tag"] = f"{args.run_tag}/{chain_name}"
    path = chain_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))


def _share_task1_artifacts(chain_best: Path, chain_final: Path, t1: str, t1_metrics: Path,
                            args) -> Tuple[Path, Path]:
    """Hardlink chain_best/task1 ckpts into chain_final and copy the JSONL. Idempotent under --resume."""
    cb_task1 = chain_best / f"task1_{t1}_ckpts"
    cf_task1 = chain_final / f"task1_{t1}_ckpts"
    if not args.dry_run:
        if cf_task1.exists():
            # Already shared (resume case). Verify the hf payload is intact; if not, redo.
            if not (hf_ckpt_complete(cf_task1 / "final_hf") and hf_ckpt_complete(cf_task1 / "best_hf")):
                shutil.rmtree(cf_task1, ignore_errors=True)
                subprocess.check_call(["cp", "-al", str(cb_task1), str(cf_task1)])
        else:
            subprocess.check_call(["cp", "-al", str(cb_task1), str(cf_task1)])
        cf_metrics = chain_final / "metrics" / f"task1_{t1}.jsonl"
        if not cf_metrics.exists() and t1_metrics.exists():
            shutil.copy2(t1_metrics, cf_metrics)
    return cf_task1 / "best_hf", cf_task1 / "final_hf"


def run_method_ppo(args, run_dir: Path, train_files: List[Path], val_files: List[Path]) -> None:
    """Original two-chain PPO pipeline (sdft / sdpo / grpo)."""
    chain_best = run_dir / "chain_best"
    chain_final = run_dir / "chain_final"
    for d in (chain_best, chain_final):
        (d / "metrics").mkdir(parents=True, exist_ok=True)
        (d / "logs").mkdir(parents=True, exist_ok=True)
        write_chain_manifest(args, d, d.name.split("_", 1)[1], train_files, val_files)

    tasks = list(args.ordering_tasks)
    t1 = tasks[0]
    t1_metrics = chain_best / "metrics" / f"task1_{t1}.jsonl"

    # ---- Task 1: shared ------------------------------------------------------
    best_hf_1, final_hf_1 = launch_task_block(
        args=args, task_dir=chain_best, metrics_path=t1_metrics,
        task_idx=1, task_name=t1, model_path=args.model,
        train_file=train_files[0], val_files=val_files,
        log_file=chain_best / "logs" / f"task1_{t1}.log",
        experiment_name=f"{args.run_tag}_task1_{t1}",
    )
    best_hf_1_f, final_hf_1_f = _share_task1_artifacts(chain_best, chain_final, t1, t1_metrics, args)

    # ---- Chain-best: tasks 2..T from running best ---------------------------
    if args.skip_chain_best:
        print("[skip-chain-best] chain_best loop skipped (only chain_final produced).")
    else:
        best_hf = best_hf_1
        for i, task in enumerate(tasks[1:], start=2):
            best_hf, _ = launch_task_block(
                args=args, task_dir=chain_best,
                metrics_path=chain_best / "metrics" / f"task{i}_{task}.jsonl",
                task_idx=i, task_name=task, model_path=str(best_hf),
                train_file=train_files[i - 1], val_files=val_files,
                log_file=chain_best / "logs" / f"task{i}_{task}.log",
                experiment_name=f"{args.run_tag}_best_task{i}_{task}",
            )

    # ---- Chain-final: tasks 2..T from running final -------------------------
    if args.skip_chain_final:
        print("[skip-chain-final] chain_final loop skipped (only chain_best produced).")
        return
    final_hf = final_hf_1_f
    for i, task in enumerate(tasks[1:], start=2):
        _, final_hf = launch_task_block(
            args=args, task_dir=chain_final,
            metrics_path=chain_final / "metrics" / f"task{i}_{task}.jsonl",
            task_idx=i, task_name=task, model_path=str(final_hf),
            train_file=train_files[i - 1], val_files=val_files,
            log_file=chain_final / "logs" / f"task{i}_{task}.log",
            experiment_name=f"{args.run_tag}_final_task{i}_{task}",
        )


def run_method_sft(args, run_dir: Path, train_files: List[Path], val_files: List[Path]) -> None:
    """Legacy SFT pipeline (kept for reference; the active SFT path is run_method_ppo
    via sft.yaml).

    Emits the same on-disk layout as the PPO-method runs:
      <run_dir>/
        manifest.json, DONE
        task<i>_<t>_train.parquet, task<i>_<t>_train_sft.parquet, val_*.parquet
        chain_{best,final}/
          manifest.json
          metrics/task<i>_<t>.jsonl                  val rollout rows
          logs/task<i>_<t>.log                       SFT trainer stdout
          logs/_eval_zeroshot_task1_<t1>.log         zero-shot eval-only stdout
          logs/_eval_post_task<i>_<t>.log            post-task eval-only stdout
          logs/_eval_<run_tag>_<experiment_name>.jsonl
          task<i>_<t>_ckpts/{best_hf, final_hf, latest_checkpointed_iteration.txt}

    SFT has no rollout-based per-step val picker, so chain_best == chain_final
    (mirrored via cp -al at the end). The FSDP SFT trainer writes its HF weights at
    <step>/huggingface (no `actor/` intermediate); _finalize handles that path.

    Eval-only passes (main_ppo with trainer.val_only=true) populate the per-task
    metrics JSONLs at: step 0 of task1.jsonl <- zero-shot, step i of task<i>.jsonl
    <- after task i, for i = 1..T.
    """
    chain_final = run_dir / "chain_final"
    chain_best = run_dir / "chain_best"
    for d in (chain_final, chain_best):
        (d / "metrics").mkdir(parents=True, exist_ok=True)
        (d / "logs").mkdir(parents=True, exist_ok=True)
        write_chain_manifest(args, d, d.name.split("_", 1)[1], train_files, val_files)

    sft_train_files = stage_sft_train_files(args, run_dir, train_files)
    tasks = list(args.ordering_tasks)
    t1 = tasks[0]

    # ---- Pre-task-1 eval (zero-shot) ----------------------------------------
    t1_metrics = chain_final / "metrics" / f"task1_{t1}.jsonl"
    if not (args.resume and t1_metrics.exists() and t1_metrics.stat().st_size > 0):
        if t1_metrics.exists():
            t1_metrics.unlink()
        launch_eval_only_block(
            args=args, task_dir=chain_final, metrics_path=t1_metrics,
            model_path=args.model, val_files=val_files,
            log_file=chain_final / "logs" / f"_eval_zeroshot_task1_{t1}.log",
            experiment_name=f"{args.run_tag}_zeroshot",
            label_step=0,
        )
    else:
        print(f"[resume] {chain_final.name}/task1_{t1} pre-eval already in JSONL, skipping zero-shot eval")

    # ---- Train + post-eval, sequentially ------------------------------------
    cur_model = args.model
    for i, task in enumerate(tasks, start=1):
        ckpt_hf = launch_sft_task_block(
            args=args, task_dir=chain_final, task_idx=i, task_name=task,
            model_path=str(cur_model), sft_train_file=sft_train_files[i - 1],
            log_file=chain_final / "logs" / f"task{i}_{task}.log",
            experiment_name=f"{args.run_tag}_final_task{i}_{task}",
        )
        cur_model = ckpt_hf

        task_metrics = chain_final / "metrics" / f"task{i}_{task}.jsonl"
        # Has post-eval already been written for this task? We track by line count;
        # task1's JSONL has the zero-shot row plus the post-eval row (=2 rows). Other
        # tasks have just the post-eval row (=1 row).
        expected_min_lines = 2 if i == 1 else 1
        existing_lines = 0
        if task_metrics.exists():
            with open(task_metrics, "rb") as fh:
                existing_lines = sum(1 for _ in fh)
        if args.resume and existing_lines >= expected_min_lines:
            print(f"[resume] {chain_final.name}/task{i}_{task} post-eval already present ({existing_lines} rows), skipping")
            continue
        launch_eval_only_block(
            args=args, task_dir=chain_final, metrics_path=task_metrics,
            model_path=str(ckpt_hf), val_files=val_files,
            log_file=chain_final / "logs" / f"_eval_post_task{i}_{task}.log",
            experiment_name=f"{args.run_tag}_eval_after_task{i}_{task}",
            label_step=i,  # any monotone step suffices; analyze.py picks last per task
        )

    # ---- Mirror chain_final -> chain_best so analyze.py treats them uniformly ----
    if not args.dry_run:
        for sub in ("metrics", "logs"):
            src = chain_final / sub
            dst = chain_best / sub
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        for i, task in enumerate(tasks, start=1):
            cf_ckpts = chain_final / f"task{i}_{task}_ckpts"
            cb_ckpts = chain_best / f"task{i}_{task}_ckpts"
            if cb_ckpts.exists():
                shutil.rmtree(cb_ckpts)
            if cf_ckpts.exists():
                subprocess.check_call(["cp", "-al", str(cf_ckpts), str(cb_ckpts)])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=list(METHOD_CONFIG.keys()), required=True)
    p.add_argument("--ordering", choices=list(ORDERINGS.keys()), required=True)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--enable-thinking", type=lambda s: s.lower() == "true", default=False)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-samples", type=int, default=500, dest="train_samples")
    p.add_argument("--total-epochs", type=int, default=2, dest="total_epochs",
                   help="Epochs per task. Steps/task = total_epochs * ceil(train_samples/train_batch).")
    p.add_argument("--eval-every-prompts", type=int, default=100, dest="eval_every_prompts")
    p.add_argument("--test-freq", type=int, default=1, dest="test_freq",
                   help="verl trainer.test_freq (eval+save cadence). 1=every step (default, densest curve). "
                        "0=eval only at end (val_before_train still gives step 0). N>1=every N steps.")
    p.add_argument("--train-batch", type=int, default=32)
    p.add_argument("--val-n", type=int, default=8, dest="val_n", help="rollouts per val prompt")
    p.add_argument("--n-gpus", type=int, default=8, dest="n_gpus")
    p.add_argument("--nnodes", type=int, default=1)
    p.add_argument(
        "--out-root",
        default=os.environ.get("SDPO_RUN_ROOT", str(SDPO_ROOT / "runs")),
        dest="out_root",
        help="Where per-run output dirs live. Defaults to $SDPO_RUN_ROOT or <repo>/runs.",
    )
    p.add_argument("--sweep-tag", default="", dest="sweep_tag",
                   help="Optional prefix (e.g. 'sweep-0422-a3f7') prepended to run_tag so "
                        "reruns of the same (method,ordering,seed) don't collide in wandb/runs/.")
    p.add_argument("--resume", action="store_true",
                   help="Skip task launches whose final_hf already exists in run_dir; reuse that ckpt "
                        "as the starting point for the next task. Use after copying a partially-failed "
                        "run_dir into a fresh sweep_tag (see sweep_resume.sh).")
    p.add_argument("--skip-chain-final", action="store_true", dest="skip_chain_final",
                   help="PPO methods only: skip the chain_final loop (tasks 2..T started from running "
                        "final ckpt). Use when only the best-checkpoint chain is needed (faster).")
    p.add_argument("--skip-chain-best", action="store_true", dest="skip_chain_best",
                   help="PPO methods only: skip the chain_best loop (tasks 2..T started from running "
                        "best ckpt). Use when only the final-checkpoint chain is needed (continue-from-final "
                        "semantics, e.g. ordering F's temporal-drift sweep).")
    p.add_argument("--num-tasks", type=int, default=0, dest="num_tasks",
                   help="If >0, run only the first N tasks of the ordering (debugging / partial reruns). "
                        "Default 0 = all tasks in the ordering.")
    p.add_argument("--sft-max-length", type=int, default=4096, dest="sft_max_length",
                   help="SFT-only: max sequence length for token packing in fsdp_sft_trainer. "
                        "Bumped from 2048 to 4096 because tooluse prompts+gold (tool docs + JSON "
                        "actions) routinely exceed 2048; right-truncation was killing the gold "
                        "answer's tail and producing acc=0 on tooluse after SFT-on-tooluse.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    # Temporal-drift chain (ordering T) defaults to 1 epoch — short chain, fast iteration.
    # Pass --total-epochs explicitly to override.
    if args.ordering == "T" and "--total-epochs" not in sys.argv:
        args.total_epochs = 1
        print(f"=== ordering=T: forcing total_epochs=1 (override --total-epochs to change)")

    args.ordering_tasks = ORDERINGS[args.ordering]
    if args.num_tasks and args.num_tasks > 0:
        args.ordering_tasks = args.ordering_tasks[: args.num_tasks]
        print(f"=== num-tasks={args.num_tasks}: truncated ordering to {args.ordering_tasks}")
    run_dir = build_run_dir(args)
    print(f"=== run_tag: {args.run_tag}")
    print(f"=== run_dir: {run_dir}")
    print(f"=== ordering tasks: {args.ordering_tasks}")
    if args.resume:
        print(f"=== RESUME mode: skipping tasks with existing final_hf")

    train_files, val_files = stage_data(args, run_dir)
    write_manifest(args, run_dir, train_files, val_files)

    # Directory layout
    # --------------------------------------------------------------------------
    # <run_dir>/
    #   manifest.json
    #   chain_best/{manifest.json, metrics/, task<i>_<t>_ckpts/}
    #   chain_final/{manifest.json, metrics/, task<i>_<t>_ckpts/}
    # Task 1 trains ONCE into chain_best/task1_*_ckpts; we hardlink that tree (cp -al)
    # into chain_final and copy its JSONL across, so the two chains share task-1 cost but
    # remain self-contained for analyze.py.
    # --------------------------------------------------------------------------
    # SFT now goes through the same main_ppo path as sdft/sdpo/grpo via sft.yaml
    # (alpha=0 + teacher_regularization=gold + use_gold_response=true ⇒ NLL on gold).
    # Schema parity with PPO methods is automatic.
    #
    # Escape hatch: set SFT_LEGACY=1 to route SFT through the legacy
    # fsdp_sft_trainer path (pure NLL, no rollout, no ratio inflation from the
    # Qwen3 chat-template last-turn quirk). Use this when the main_ppo
    # rollout↔training PPL gap is corrupting the gradient and the loss won't
    # descend smoothly.
    if args.method == "sft" and os.environ.get("SFT_LEGACY") == "1":
        print(f"=== SFT_LEGACY=1: routing SFT through legacy fsdp_sft_trainer path")
        run_method_sft(args, run_dir, train_files, val_files)
    else:
        run_method_ppo(args, run_dir, train_files, val_files)

    # Finalize
    (run_dir / "DONE").write_text(dt.datetime.utcnow().isoformat(timespec="seconds") + "Z\n")
    print(f"\n=== run {args.run_tag} complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
