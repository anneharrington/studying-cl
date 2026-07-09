"""Unified entry point for all benchmark runs.

Supports both traditional flat configs and composable configs.

Traditional (existing configs work unchanged):
    python scripts/run.py --method gepa --task hotpotqa --config configs/gepa_hotpotqa.yaml
    python scripts/run.py --method gepa --strategy sequential --config configs/gepa_sequential.yaml

Composable (new thin configs referencing model/task/method profiles):
    python scripts/run.py --method gepa --task hotpotqa --config configs/runs/single.yaml
    python scripts/run.py --method gepa --strategy sequential --config configs/runs/sequential.yaml

CLI overrides (work with both config styles):
    python scripts/run.py --method gepa --task hotpotqa --model qwen-3-8b --config configs/runs/single.yaml
    python scripts/run.py --method openevolve --strategy sequential --tasks hotpotqa ifeval hover sciknoweval --model gemini-flash-lite --config configs/runs/sequential.yaml

Strategies:
    --strategy sequential    Replace instructions at each stage (default sequential)
    --strategy append        Append previous optimized instructions with META_PROMPT
    --strategy mixed         Round-robin interleaving across tasks
    --allorders              Run all 6 task orderings (with sequential/append)
    --ordering 1,3,5         Run specific orderings (with --allorders)
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cl.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Run benchmark optimization")
    parser.add_argument("--method", required=True,
                        choices=["gepa", "openevolve", "openevolve-v2", "openevolve-v2-meta",
                                 "openevolve-v2-meta-single", "ace", "ace-minimal"],
                        help="Optimization method (openevolve-v2 uses delimited general instructions; "
                             "openevolve-v2-meta additionally evolves the meta_prompt between tasks; "
                             "openevolve-v2-meta-single resets [GI] every task so meta_prompt is the "
                             "only cross-task carrier, and measures position-vs-score on the current task; "
                             "ace wraps the reference ACE implementation in ./ace/; "
                             "ace-minimal is a from-scratch reimplementation kept as a fallback)")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--model", default=None,
                        help="Model profile name (e.g. gemini-flash-lite, qwen-3-8b)")
    parser.add_argument("--task", default=None,
                        help="Single-task mode: task name (e.g. hotpotqa, sciknoweval)")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Multi-task mode: task names (e.g. hotpotqa ifeval hover)")
    parser.add_argument("--strategy", default=None,
                        choices=["sequential", "append", "mixed"],
                        help="Multi-task strategy (omit for single-task)")
    parser.add_argument("--allorders", action="store_true",
                        help="Run all 6 task orderings (sequential/append only)")
    parser.add_argument("--ordering", default=None,
                        help="Comma-separated ordering indices 1-6 (with --allorders)")
    parser.add_argument("--output-dir", default=None,
                        help="Override output directory (timestamp+ID still appended)")
    parser.add_argument("--skip-baseline", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="Skip the pre-optimization baseline stage. Overrides "
                             "the config's skip_baseline; omit to use the config "
                             "value. Use --no-skip-baseline to force the baseline "
                             "on even if the config disables it.")
    parser.add_argument("--train-n", type=int, default=None,
                        help="Override train_n on every task in the run")
    parser.add_argument("--val-n", type=int, default=None,
                        help="Override val_n on every task in the run")
    parser.add_argument("--eval-n", type=int, default=None,
                        help="Override eval_n on every task in the run")
    parser.add_argument("--resume-from", type=int, default=None,
                        help="ACE only: 1-indexed task position to resume at. "
                             "Loads <output_dir>/_ace_checkpoint.json and skips "
                             "earlier phases. The output_dir must be the exact "
                             "timestamped+id'd directory from the original run "
                             "(e.g. results/seq_ace_finance_20260503_1745_a3f2).")
    args = parser.parse_args()

    # Load and assemble config (handles both traditional and composable)
    # Normalize method name for config loading (openevolve-v2 uses same config as openevolve)
    if args.method.startswith("openevolve"):
        config_method = "openevolve"
    else:
        config_method = args.method
    cfg = load_config(args.config, method=config_method, model=args.model,
                      tasks=args.tasks, task=args.task,
                      split_overrides={
                          "train_n": args.train_n,
                          "val_n": args.val_n,
                          "eval_n": args.eval_n,
                      })

    if args.output_dir:
        cfg["output_dir"] = args.output_dir

    # CLI override for skip_baseline — applies to every runner, which all read
    # cfg.get("skip_baseline", False). None means "not specified": leave the
    # config value untouched.
    if args.skip_baseline is not None:
        cfg["skip_baseline"] = args.skip_baseline

    # Parse ordering indices (validated by the runner against actual permutation count)
    ordering_indices = None
    if args.ordering is not None:
        ordering_indices = [int(x.strip()) for x in args.ordering.split(",")]

    if args.method == "gepa":
        from methods.prompt_based.runners.gepa_runner import run_single, run_sequential, run_allorders, run_mixed

        if args.task or cfg.get("task_name"):
            cfg.setdefault("task_name", args.task)
            run_single(cfg)
        elif args.strategy == "mixed":
            run_mixed(cfg)
        elif args.allorders:
            strategy = "append" if args.strategy == "append" else "replace"
            run_allorders(cfg, strategy=strategy, ordering_indices=ordering_indices)
        elif args.strategy in ("sequential", "append"):
            strategy = "append" if args.strategy == "append" else "replace"
            run_sequential(cfg, strategy=strategy)
        else:
            parser.error("Specify --task for single-task or --strategy for multi-task")

    elif args.method == "openevolve":
        from methods.prompt_based.runners.openevolve_runner import run_single, run_sequential, run_allorders, run_mixed

        if args.task or cfg.get("task_name"):
            cfg["_config_path"] = args.config
            run_single(cfg, task_name=args.task or cfg.get("task_name"))
        elif args.strategy == "mixed":
            run_mixed(cfg)
        elif args.allorders:
            strategy = "append" if args.strategy == "append" else "replace"
            run_allorders(cfg, strategy=strategy, ordering_indices=ordering_indices)
        elif args.strategy in ("sequential", "append"):
            strategy = "append" if args.strategy == "append" else "replace"
            run_sequential(cfg, strategy=strategy)
        else:
            parser.error("Specify --task for single-task or --strategy for multi-task")

    elif args.method == "openevolve-v2":
        from methods.prompt_based.runners.openevolve_runner_v2 import run_single, run_sequential, run_allorders, run_mixed

        if args.task or cfg.get("task_name"):
            cfg["_config_path"] = args.config
            run_single(cfg, task_name=args.task or cfg.get("task_name"))
        elif args.strategy == "mixed":
            run_mixed(cfg)
        elif args.allorders:
            strategy = "append" if args.strategy == "append" else "replace"
            run_allorders(cfg, strategy=strategy, ordering_indices=ordering_indices)
        elif args.strategy in ("sequential", "append"):
            strategy = "append" if args.strategy == "append" else "replace"
            run_sequential(cfg, strategy=strategy)
        else:
            parser.error("Specify --task for single-task or --strategy for multi-task")

    elif args.method == "openevolve-v2-meta":
        # Sequential entry points come from the meta runner; single/mixed reuse v2.
        from methods.prompt_based.runners.openevolve_runner_v2_meta import run_sequential, run_allorders
        from methods.prompt_based.runners.openevolve_runner_v2 import run_single, run_mixed

        if args.task or cfg.get("task_name"):
            cfg["_config_path"] = args.config
            run_single(cfg, task_name=args.task or cfg.get("task_name"))
        elif args.strategy == "mixed":
            run_mixed(cfg)
        elif args.allorders:
            strategy = "append" if args.strategy == "append" else "replace"
            run_allorders(cfg, strategy=strategy, ordering_indices=ordering_indices)
        elif args.strategy in ("sequential", "append"):
            strategy = "append" if args.strategy == "append" else "replace"
            run_sequential(cfg, strategy=strategy)
        else:
            parser.error("Specify --task for single-task or --strategy for multi-task")

    elif args.method == "openevolve-v2-meta-single":
        # [GI] is always reset; strategy arg is ignored for consistency with v2.
        from methods.prompt_based.runners.openevolve_runner_v2_meta_single import run_sequential, run_allorders

        if args.task or cfg.get("task_name"):
            parser.error("openevolve-v2-meta-single does not support single-task mode — "
                         "the whole point is measuring position-vs-score across a sequence. "
                         "Use --strategy sequential or --allorders.")
        elif args.strategy == "mixed":
            parser.error("openevolve-v2-meta-single does not support mixed (round-robin) mode.")
        elif args.allorders:
            run_allorders(cfg, ordering_indices=ordering_indices)
        elif args.strategy in ("sequential", "append"):
            run_sequential(cfg)
        else:
            parser.error("Specify --strategy sequential or --allorders for openevolve-v2-meta-single")

    elif args.method in ("ace", "ace-minimal"):
        # Both ACE variants are inherently sequential. Single-task / mixed are
        # not supported because their whole point is a cross-task playbook.
        if args.method == "ace":
            from methods.prompt_based.runners.ace_runner import run_sequential, run_allorders
        else:
            from methods.prompt_based.runners.ace_minimal_runner import run_sequential, run_allorders

        if args.task or cfg.get("task_name"):
            parser.error(f"{args.method} does not support single-task mode — it "
                         "requires a cross-task sequence. Use --strategy sequential "
                         "or --allorders.")
        elif args.strategy == "mixed":
            parser.error(f"{args.method} does not support mixed (round-robin) mode.")
        elif args.allorders:
            if args.resume_from is not None:
                parser.error("--resume-from is not supported with --allorders. "
                             "Resume requires a single ordering's checkpoint.")
            run_allorders(cfg, ordering_indices=ordering_indices)
        elif args.strategy in ("sequential", "append"):
            if args.resume_from is not None and args.method != "ace":
                parser.error("--resume-from is only supported for --method ace.")
            run_sequential(cfg, resume_from=args.resume_from)
        else:
            parser.error(f"Specify --strategy sequential or --allorders for {args.method}")


if __name__ == "__main__":
    main()
