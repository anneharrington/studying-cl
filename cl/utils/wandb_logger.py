"""Weights & Biases logging for benchmark runs.

Provides batched example-level logging during evaluation, plus stage-level
and task-level summaries. Thread-safe for use with parallel eval.

Usage:

    wlog = WandbLogger.from_config(cfg)  # reads cfg["wandb"]
    if wlog:
        wlog.log_stage_scores("after_hotpotqa", scores)

    # Inside _score_example / metric wrapper:
    if wlog:
        wlog.record_example("hotpotqa", score, phase="eval")

    # At end of run:
    if wlog:
        wlog.finish()

Config:

    wandb:
      enabled: true
      project: "adaptive-intelligence-benchmark"
      name: "gepa-sequential-qwen3-8b"   # optional, auto-generated if omitted
      entity: null             # optional team/user
      tags: ["gepa", "qwen"]   # optional
      log_batch_size: 50       # log avg every N examples
"""

import os
import threading
from collections import defaultdict

try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    _WANDB_AVAILABLE = False


# Map task names to their metric type for display
TASK_METRIC_LABELS = {
    "hotpotqa": "hotpotqa_f1",
    "ifeval": "ifeval_acc",
    "hover": "hover_acc",
}


def _labeled(task_name):
    """Return task name with metric type suffix."""
    return TASK_METRIC_LABELS.get(task_name, task_name)


def _auto_run_name(cfg):
    """Generate an informative default run name from the config."""
    parts = []

    # Method: gepa or openevolve (infer from config keys)
    if "gepa" in cfg:
        parts.append("gepa")
    elif "openevolve" in cfg:
        parts.append("openevolve")

    # Strategy: sequential, append, mixed, allorders (infer from output_dir)
    output_dir = cfg.get("output_dir", "")
    for strategy in ["mixed", "allorders", "append", "sequential"]:
        if strategy in output_dir:
            parts.append(strategy)
            break

    # Model: short name from task_lm
    model_cfg = cfg.get("model", {})
    task_lm = model_cfg.get("task_lm", "")
    # Extract last segment: "openai/qwen/qwen3-8b" -> "qwen3-8b"
    model_short = task_lm.rsplit("/", 1)[-1] if task_lm else ""
    if model_short:
        parts.append(model_short)

    # Dataset size
    tasks = cfg.get("tasks", [])
    if tasks:
        ds = tasks[0].get("dataset", {})
        train_n = ds.get("train_n", "")
        val_n = ds.get("val_n", "")
        if train_n and val_n:
            parts.append(f"t{train_n}-v{val_n}")

    return "-".join(parts) if parts else None


class WandbLogger:
    """Batched wandb logger for benchmark runs."""

    def __init__(self, project, entity=None, tags=None, log_batch_size=50,
                 config=None, name=None):
        if not _WANDB_AVAILABLE:
            raise ImportError("wandb is not installed. Run: pip install wandb")

        self._batch_size = log_batch_size
        self._buffers = defaultdict(list)  # key: (phase, task_name)
        self._counts = defaultdict(int)    # total examples seen per key
        self._lock = threading.Lock()

        wandb.init(
            project=project,
            entity=entity,
            tags=tags or [],
            config=config or {},
            name=name,
            reinit=True,
        )

    @classmethod
    def from_config(cls, cfg):
        """Create a WandbLogger from the run config, or return None if disabled.

        Args:
            cfg: Full run config dict. Reads cfg["wandb"] for settings.

        Returns:
            WandbLogger instance or None if wandb is disabled/not configured.
        """
        wandb_cfg = cfg.get("wandb", {})
        if not wandb_cfg.get("enabled", False):
            return None
        if not _WANDB_AVAILABLE:
            print("Warning: wandb.enabled=true but wandb is not installed. Skipping.")
            return None

        name = wandb_cfg.get("name") or _auto_run_name(cfg)

        return cls(
            project=wandb_cfg.get("project", "adaptive-intelligence-benchmark"),
            entity=wandb_cfg.get("entity"),
            tags=wandb_cfg.get("tags"),
            log_batch_size=wandb_cfg.get("log_batch_size", 50),
            config=cfg,
            name=name,
        )

    # ── Example-level (batched) ──────────────────────────────

    def record_example(self, task_name, score, phase="eval"):
        """Record a single example score. Logs batch avg every N examples.

        Args:
            task_name: e.g. "hotpotqa", "ifeval", "hover"
            score: float score for this example (0-1 scale for OpenEvolve, 0-100 for GEPA)
            phase: "eval", "opt", or "baseline" — used as wandb key prefix
        """
        key = (phase, task_name)
        with self._lock:
            self._buffers[key].append(score)
            self._counts[key] += 1
            if len(self._buffers[key]) >= self._batch_size:
                self._flush_buffer(key)

    def _flush_buffer(self, key):
        """Log and clear a buffer. Must be called with self._lock held."""
        buf = self._buffers[key]
        if not buf:
            return
        phase, task_name = key
        label = _labeled(task_name)
        avg = sum(buf) / len(buf)
        n = self._counts[key]
        wandb.log({
            f"{phase}/{label}_batch_avg": avg,
            f"{phase}/{label}_n": n,
        })
        buf.clear()

    def flush_task(self, task_name, phase="eval"):
        """Flush any remaining partial batch for a task. Call after task eval completes."""
        key = (phase, task_name)
        with self._lock:
            self._flush_buffer(key)

    # ── Metric wrapping (GEPA) ─────────────────────────────

    def wrap_metric(self, metric_fn, task_name, phase="eval"):
        """Wrap a DSPy metric function to record example scores.

        The wrapper extracts the numeric score from the metric return value
        and calls record_example. Works with GEPA metrics that return
        floats or dspy.Prediction-like objects with a score.

        Args:
            metric_fn: Original metric function (example, prediction, trace=None, ...)
            task_name: e.g. "hotpotqa"
            phase: "eval" or "opt"

        Returns:
            Wrapped metric function with the same signature.
        """
        wlog = self

        def _tracked_metric(example, prediction, trace=None, **kwargs):
            result = metric_fn(example, prediction, trace=trace, **kwargs)
            # Extract numeric score: could be float, int, or have a .score attr
            if isinstance(result, (int, float)):
                score = float(result)
            else:
                score = float(getattr(result, "score", result))
            wlog.record_example(task_name, score, phase=phase)
            return result

        return _tracked_metric

    # ── Task-level ───────────────────────────────────────────

    def log_task_score(self, task_name, score, stage):
        """Log a single task's final score within a stage.

        Called after each task completes in eval_all_tasks.
        """
        label = _labeled(task_name)
        wandb.log({
            f"task_score/{label}": score,
            "stage": stage,
        })

    # ── Stage-level ──────────────────────────────────────────

    def log_stage_scores(self, stage, scores, extra=None):
        """Log all task scores at a stage boundary.

        Args:
            stage: e.g. "baseline", "after_hotpotqa", "cycle_2"
            scores: dict of {task_name: score}
            extra: optional dict of additional metrics to log
        """
        payload = {f"stage_score/{_labeled(task)}": score for task, score in scores.items()}
        payload["stage"] = stage
        if extra:
            payload.update(extra)
        wandb.log(payload)

    def log_usage(self, usage, label="cumulative"):
        """Log token usage."""
        wandb.log({
            f"usage/{label}/prompt_tokens": usage.get("prompt_tokens", 0),
            f"usage/{label}/completion_tokens": usage.get("completion_tokens", 0),
            f"usage/{label}/total_tokens": usage.get("total_tokens", 0),
            f"usage/{label}/api_calls": usage.get("api_calls", 0),
        })

    def log_failures(self, failure_summary):
        """Log LM failure summary."""
        wandb.log({
            "failures/total": failure_summary.get("total", 0),
        })
        for stage, count in failure_summary.get("by_stage", {}).items():
            wandb.log({f"failures/by_stage/{stage}": count})

    # ── Lifecycle ────────────────────────────────────────────

    def finish(self):
        """Flush all buffers and finish the wandb run."""
        with self._lock:
            for key in list(self._buffers.keys()):
                self._flush_buffer(key)
        wandb.finish()
