"""Shared parquet loader for verl/sdpo-format eval parquets.

Each parquet directory holds two files written by the training pipeline:
    <dir>/train.parquet   ←  task<N>_<task>_train.parquet  in the sweep
    <dir>/val.parquet     ←  val_<task>.parquet            in the sweep

Schema (uniform across tooluse / finqa / bio):
    prompt              list[{"role": "system"|"user", "content": str}]
    reward_model        {"ground_truth": str, "style": str}
    extra_info          dict (free-form per-task metadata)
    system, embedding,  unused here
    data_source, ability

This module pulls the LAST user message out of `prompt` as `question` and
`reward_model.ground_truth` (or `extra_info.answer`, depending on task) as
`answer`. The split semantics match what the user asked for:
    train.parquet → seeded shuffle, first  train_n rows = train,
                                     next  val_n  rows = val
                    (50/50 split of the parquet when train_n+val_n == len)
    val.parquet   → first eval_n rows = eval (or all if eval_n in (0, -1))

Returning lists of plain dicts keeps the existing GEPA / OE / ACE pipelines
happy — they wrap these into dspy.Examples themselves.
"""

from __future__ import annotations

import os
import random
from typing import Callable, List, Tuple


def _user_content(prompt_field) -> str:
    """Extract the last user message's content from a chat-shaped `prompt`."""
    if hasattr(prompt_field, "tolist") and not isinstance(prompt_field, str):
        prompt_field = prompt_field.tolist()
    if isinstance(prompt_field, (list, tuple)):
        user_msgs = [m for m in prompt_field
                     if isinstance(m, dict) and m.get("role") == "user"]
        if user_msgs:
            return user_msgs[-1].get("content", "")
        # Fall back to concatenating any content fields.
        return " ".join(
            str(m.get("content", "")) if isinstance(m, dict) else str(m)
            for m in prompt_field
        )
    if isinstance(prompt_field, str):
        return prompt_field
    return str(prompt_field)


def _ground_truth(reward_model) -> str:
    """Pull the gold answer out of the verl `reward_model` field."""
    if isinstance(reward_model, dict):
        return str(reward_model.get("ground_truth", ""))
    return str(reward_model)


def is_parquet_dir(path: str) -> bool:
    """True if `path` is a directory holding train.parquet / val.parquet."""
    return (
        os.path.isdir(path)
        and os.path.isfile(os.path.join(path, "train.parquet"))
        and os.path.isfile(os.path.join(path, "val.parquet"))
    )


def load_parquet_dir(
    path: str,
    train_n: int,
    val_n: int,
    seed: int,
    eval_n: int,
    to_example: Callable[[dict], dict],
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Read train.parquet + val.parquet from `path` and split into 3 sets.

    `to_example` maps a row dict (with `question` and `answer` keys already
    pulled from the chat prompt and reward_model) to whatever shape the task's
    downstream code expects (e.g. an extra `task_type` field for FinQA, or
    additional `golden_steps` for tooluse).
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "pyarrow + pandas are required for parquet loading. "
            "Run: uv add pyarrow"
        ) from e

    train_path = os.path.join(path, "train.parquet")
    val_path = os.path.join(path, "val.parquet")

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    def _row_to_base(row) -> dict:
        return {
            "question": _user_content(row.get("prompt")),
            "answer": _ground_truth(row.get("reward_model")),
            "_extra_info": row.get("extra_info"),
        }

    train_raw = [_row_to_base(r) for _, r in train_df.iterrows()]
    val_raw = [_row_to_base(r) for _, r in val_df.iterrows()]

    print(f"  Loaded {len(train_raw)} train + {len(val_raw)} val parquet rows from {path}")

    random.Random(seed).shuffle(train_raw)
    train_set = [to_example(it) for it in train_raw[:train_n]]
    if val_n < 0:
        val_set = [to_example(it) for it in train_raw[train_n:]]
    else:
        val_set = [to_example(it) for it in train_raw[train_n : train_n + val_n]]

    if eval_n in (0, None):
        eval_set: List[dict] = [to_example(it) for it in val_raw]
    elif eval_n < 0:
        eval_set = [to_example(it) for it in val_raw]
    else:
        eval_set = [to_example(it) for it in val_raw[:eval_n]]

    return train_set, val_set, eval_set
