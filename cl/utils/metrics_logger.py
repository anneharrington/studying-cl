"""Stage-level metrics logging for benchmark runs.

Logs accuracy scores and token usage to a JSONL file after each optimization
stage completes. Called from run scripts (not evaluators).

Enable via config:
    logging:
      detailed_metrics_log: true
"""

import json
import os
import time


def log_stage(log_file, stage, method, scores, token_usage,
              optimization_usage=None, eval_usage=None):
    """Log metrics at the end of an optimization stage.

    Args:
        log_file: Path to the JSONL log file.
        stage: Stage name (e.g. "baseline", "after_hotpotqa").
        method: Method name ("gepa" or "openevolve").
        scores: Dict of task scores (e.g. {"hotpotqa": 65.0, "ifeval": 80.0, "hover": 55.0}).
        token_usage: Dict with prompt_tokens, completion_tokens, total_tokens, api_calls.
            Cumulative totals up to this point.
        optimization_usage: Optional dict with token counts for optimization only (this stage).
        eval_usage: Optional dict with token counts for evaluation only (this stage).
    """
    entry = {
        "timestamp": time.time(),
        "method": method,
        "stage": stage,
        "scores": scores,
        "prompt_tokens": token_usage.get("prompt_tokens", 0),
        "completion_tokens": token_usage.get("completion_tokens", 0),
        "total_tokens": token_usage.get("total_tokens", 0),
        "api_calls": token_usage.get("api_calls", 0),
    }
    if optimization_usage is not None:
        entry["optimization_tokens"] = {
            "prompt_tokens": optimization_usage.get("prompt_tokens", 0),
            "completion_tokens": optimization_usage.get("completion_tokens", 0),
            "total_tokens": optimization_usage.get("total_tokens", 0),
            "api_calls": optimization_usage.get("api_calls", 0),
        }
    if eval_usage is not None:
        entry["eval_tokens"] = {
            "prompt_tokens": eval_usage.get("prompt_tokens", 0),
            "completion_tokens": eval_usage.get("completion_tokens", 0),
            "total_tokens": eval_usage.get("total_tokens", 0),
            "api_calls": eval_usage.get("api_calls", 0),
        }
    try:
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass


def read_metrics_log(log_file):
    """Read all entries from a metrics log file.

    Returns:
        List of dicts, one per logged stage.
    """
    entries = []
    if not log_file or not os.path.exists(log_file):
        return entries
    try:
        with open(log_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    except (OSError, json.JSONDecodeError):
        pass
    return entries
