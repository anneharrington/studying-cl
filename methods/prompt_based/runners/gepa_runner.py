"""Shared GEPA runner logic.

Provides functions for single-task, sequential, append, allorders, and mixed
strategies. Individual scripts become thin wrappers around these functions.
"""

import itertools
import json
import random
import time
from pathlib import Path

import dspy

from methods.prompt_based.gepa_method import configure_lms, run_gepa
from cl.tasks import get_gepa_tasks
from cl.utils.token_tracker import get_dspy_usage, usage_diff
from cl.utils.metrics_logger import log_stage, read_metrics_log
from cl.utils.failure_tracker import FailureTracker
from cl.utils.wandb_logger import WandbLogger
from cl.utils.plotting import plot_sequential, plot_allorders

# All 6 orderings of 3 tasks (used by allorders strategies)
ALL_ORDERINGS = [
    ("hotpotqa", "ifeval", "hover"),
    ("hotpotqa", "hover", "ifeval"),
    ("ifeval", "hotpotqa", "hover"),
    ("ifeval", "hover", "hotpotqa"),
    ("hover", "hotpotqa", "ifeval"),
    ("hover", "ifeval", "hotpotqa"),
]

from methods.prompt_based.runners.meta_prompt import get_meta_prompt

# Module-level state set by setup_run() so deeply-nested metric calls can reach it.
_predictions_log_path = None


# ---------------------------------------------------------------------------
#  Shared helpers
# ---------------------------------------------------------------------------

def _wrap_metric_with_meta(metric, meta_prompt):
    """Wrap a metric function to append meta_prompt to feedback strings.

    This lets GEPA's reflection LM see the multi-task retention guidance
    without polluting the task LM's instruction text.
    """
    if not meta_prompt:
        return metric

    def wrapped(example, prediction, trace=None, pred_name=None, pred_trace=None):
        result = metric(example, prediction, trace, pred_name, pred_trace)
        if hasattr(result, 'feedback') and result.feedback:
            result = type(result)(
                score=result.score,
                feedback=f"{result.feedback}\n\nNote: {meta_prompt}",
            )
        return result

    return wrapped


def _wrap_metric_with_predictions_log(metric, task_name, stage, path, instructions):
    """Wrap a DSPy metric to also append per-example records to the predictions log.

    DSPy calls the metric with (example, prediction, trace, pred_name, pred_trace).
    We render a question/gold summary from the example, extract a short string
    representation of the prediction, compute the numeric score, and append.
    """
    if not path or not stage or not task_name:
        return metric

    from cl.utils.predictions_logger import log_prediction, render_question, render_gold
    import threading
    _idx_lock = threading.Lock()
    _idx = {"n": 0}

    def wrapped(example, prediction, trace=None, pred_name=None, pred_trace=None):
        result = metric(example, prediction, trace, pred_name, pred_trace)
        # DSPy metrics may return either a float or a ScorePrediction-like object
        # with a .score attribute.
        score_val = getattr(result, "score", result)
        try:
            score_float = float(score_val)
        except (TypeError, ValueError):
            score_float = 0.0
        # DSPy Example is dict-like; render using our task-aware helpers.
        ex_dict = {k: example[k] for k in getattr(example, "keys", lambda: [])()} if example is not None else {}
        # Extract a short textual form of the prediction (DSPy Predictions stringify OK).
        pred_str = ""
        try:
            pred_str = str(prediction)
        except Exception:
            pred_str = "<unprintable prediction>"
        # Try to pluck a cleaner extracted answer if there's a conventional field.
        extracted = ""
        for k in ("answer", "label", "response", "output", "result"):
            if hasattr(prediction, k):
                extracted = str(getattr(prediction, k))
                break
        if not extracted:
            extracted = pred_str[:400]
        with _idx_lock:
            i = _idx["n"]; _idx["n"] += 1
        log_prediction(
            path=path,
            stage=stage,
            task=task_name,
            question=render_question(task_name, ex_dict),
            llm_response=pred_str[:4000],
            extracted=extracted,
            gold=render_gold(task_name, ex_dict),
            metric=score_float,
            index=i,
            general_instructions=instructions,
        )
        return result

    return wrapped


def _evaluate_at_n(program, devset, metric, num_threads, n_rollouts):
    """Evaluate `program` over `devset` with `n_rollouts` samples per example.

    Returns the mean over examples of the per-example mean score across
    rollouts. With n_rollouts=1 this is one call per example (matches RL's
    `mean@1`); with n_rollouts=N it's N calls per example before averaging
    (matches verl's `val_kwargs.n=N` → metric key `acc/mean@N`).

    Each rollout is a fresh program(**inputs) call, so the task LM's
    sampling temperature determines whether the N rollouts are diverse. For
    apples-to-apples with the SDPO bundle's mean@8, configure the task LM
    with temperature > 0.

    Returns score on the 0-100 scale (matches dspy.Evaluate's convention).
    """
    from concurrent.futures import ThreadPoolExecutor

    inputs_list = [{k: ex[k] for k in ex.inputs().keys()} for ex in devset]

    def _score_one(idx):
        ex = devset[idx]
        kwargs = inputs_list[idx]
        per_rollout = []
        for _ in range(n_rollouts):
            try:
                pred = program(**kwargs)
                result = metric(ex, pred)
            except Exception as e:
                # Match dspy.Evaluate's lenient behavior: a per-example failure
                # contributes 0 rather than aborting the whole eval.
                print(f"    eval rollout failed on example {idx}: {type(e).__name__}: {e}")
                per_rollout.append(0.0)
                continue
            score_val = getattr(result, "score", result)
            try:
                per_rollout.append(float(score_val))
            except (TypeError, ValueError):
                per_rollout.append(0.0)
        return sum(per_rollout) / len(per_rollout) if per_rollout else 0.0

    with ThreadPoolExecutor(max_workers=max(1, num_threads)) as pool:
        per_example = list(pool.map(_score_one, range(len(devset))))

    if not per_example:
        return 0.0
    return 100.0 * sum(per_example) / len(per_example)


def eval_instructions_on_task(instructions, build_fn, val_set, metric, num_threads,
                              wlog=None, task_name=None, phase="eval", stage=None,
                              n_rollouts=1):
    """Evaluate a set of instructions on a specific task's val set.

    n_rollouts: samples per example (default 1 = mean@1, matches dspy.Evaluate).
    n_rollouts > 1 routes through `_evaluate_at_n` for mean@N parity with the
    SDPO bundle's val-core/<source>/acc/mean@N metric.
    """
    eval_metric = metric
    if wlog and task_name:
        eval_metric = wlog.wrap_metric(eval_metric, task_name, phase=phase)
    if _predictions_log_path and stage and task_name:
        eval_metric = _wrap_metric_with_predictions_log(
            eval_metric, task_name, stage, _predictions_log_path, instructions
        )

    program = build_fn()
    program.predict.signature = program.predict.signature.with_instructions(instructions)

    if n_rollouts and n_rollouts > 1:
        score = _evaluate_at_n(program, val_set, eval_metric, num_threads, n_rollouts)
    else:
        evaluator = dspy.Evaluate(
            devset=val_set,
            metric=eval_metric,
            num_threads=num_threads,
            display_progress=True,
        )
        score = evaluator(program).get("score")

    if wlog and task_name:
        wlog.flush_task(task_name, phase=phase)

    return score


def eval_all_tasks(instructions, task_data, num_threads, wlog=None, stage=None,
                   n_rollouts=1):
    """Evaluate instructions on all tasks in parallel. Uses eval_set if available."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _eval_one(task_name, td):
        score = eval_instructions_on_task(
            instructions=instructions,
            build_fn=td["build_program"],
            val_set=td.get("eval_set", td["val_set"]),
            metric=td["metric"],
            num_threads=num_threads,
            wlog=wlog,
            task_name=task_name,
            stage=stage,
            n_rollouts=n_rollouts,
        )
        print(f"    {task_name}: {score:.2f}")
        if wlog and stage:
            wlog.log_task_score(task_name, score, stage)
        return task_name, score

    scores = {}
    with ThreadPoolExecutor(max_workers=len(task_data)) as pool:
        futures = {pool.submit(_eval_one, name, td): name for name, td in task_data.items()}
        for future in as_completed(futures):
            failed_name = futures[future]
            try:
                task_name, score = future.result()
                scores[task_name] = score
            except Exception as e:
                import traceback
                print(f"    eval_all_tasks failed on {failed_name}: {type(e).__name__}: {e}")
                traceback.print_exc()
                scores[failed_name] = 0.0

    if wlog and stage:
        wlog.log_stage_scores(stage, scores)

    return scores


def load_all_datasets(cfg, tasks_info):
    """Load datasets for all tasks specified in config. Returns task_data dict."""
    task_data = {}
    for task_cfg in cfg["tasks"]:
        name = task_cfg["name"]
        ds = task_cfg["dataset"]
        task_info = tasks_info[name]
        print(f"Loading {name} dataset from {ds['path']}...")
        eval_n = ds.get("eval_n", 0)
        loader_kwargs = dict(
            path=ds["path"],
            train_n=ds["train_n"],
            val_n=ds["val_n"],
            seed=ds["seed"],
            eval_n=eval_n,
        )
        # Optional pass-throughs for loaders that support them (e.g. sentiment10k
        # uses year_filter to scope each task entry to one filing year, and
        # max_context_chars to truncate filing text; temporalwiki uses
        # slice_filter to scope each task entry to one drift slice).
        for opt_key in ("year_filter", "max_context_chars", "slice_filter"):
            if opt_key in ds:
                loader_kwargs[opt_key] = ds[opt_key]
        splits = task_info["loader"](**loader_kwargs)
        train_set, val_set = splits[0], splits[1]
        # GEPA program-builder selection (priority order):
        #   1) use_system_prefix: keep GEPA's natural system-message slot,
        #      strip the DSPy [[ ## ]] field markers. Evolved instructions →
        #      bare system message; question → bare user message. Avoids the
        #      tooluse 0-baseline scaffold-conflict while letting GEPA improve.
        #   2) use_cot_user_prefix: ChainOfThought + custom adapter that moves
        #      signature.instructions from system → user message. Evolved slot
        #      matches ACE/OE; CoT field preserved so GEPA can still improve.
        #   3) use_raw_prefix_prompt: bypass DSPy adapter entirely, single
        #      user-message call structurally identical to OpenEvolve.
        #   4) use_oe_baseline_prompt: dspy.Predict (drops CoT) seeded with
        #      OE-equivalent instructions, but DSPy adapter still wraps in
        #      [[ ## ]] markers.
        #   5) (default) task_info["build_program"] = stock dspy.ChainOfThought.
        gepa_cfg = cfg.get("gepa", {})
        seed_mode = cfg.get("seed_mode", "default")
        if gepa_cfg.get("use_system_prefix"):
            from methods.prompt_based.gepa_system_prefix import build_program_system_prefix
            _build_program = lambda _name=name: build_program_system_prefix(_name)
        elif gepa_cfg.get("use_cot_user_prefix"):
            from methods.prompt_based.gepa_cot_user_prefix import build_program_cot_user_prefix
            _build_program = lambda _name=name: build_program_cot_user_prefix(_name)
        elif gepa_cfg.get("use_raw_prefix_prompt"):
            from methods.prompt_based.gepa_raw_prefix import build_program_raw_prefix
            _build_program = lambda _name=name, _mode=seed_mode: build_program_raw_prefix(_name, seed_mode=_mode)
        elif gepa_cfg.get("use_oe_baseline_prompt"):
            from methods.prompt_based.gepa_oe_prompt import build_program_oe_style
            _build_program = lambda _name=name: build_program_oe_style(_name)
        else:
            _build_program = task_info["build_program"]

        td = {
            "train_set": train_set,
            "val_set": val_set,
            "metric": task_info["metric"],
            "build_program": _build_program,
        }
        size_str = f"  {name}: {len(train_set)} train, {len(val_set)} val"
        if len(splits) == 3:
            td["eval_set"] = splits[2]
            size_str += f", {len(splits[2])} eval"
        print(size_str)
        task_data[name] = td
    return task_data


def setup_run(cfg):
    """Common setup: configure LMs, trackers, output dir. Returns (reflection_lm, failure_tracker, wlog, output_dir, metrics_log_path)."""
    import os
    from cl.config import ensure_unique_output_dir
    ensure_unique_output_dir(cfg)

    api_key_env = cfg["model"].get("api_key_env", "PORTKEY_API_KEY")
    # SageMaker/Bedrock routes use AWS auth (boto3 credential chain or
    # AWS_BEARER_TOKEN_BEDROCK), not the harness's API key env. Skip the env
    # var check for them. Any other model still requires the configured key.
    _task_lm_str = str(cfg["model"]["task_lm"])
    is_aws = _task_lm_str.startswith("sagemaker:") or _task_lm_str.startswith("bedrock:")
    if not is_aws and api_key_env not in os.environ:
        print(f"Error: {api_key_env} environment variable not set")
        import sys
        sys.exit(1)
    # Split-provider reflector: if the model profile points the reflection LM
    # at its own key env var, that var must be set too.
    reflection_api_key_env = cfg["model"].get("reflection_api_key_env")
    if reflection_api_key_env and reflection_api_key_env not in os.environ:
        print(f"Error: {reflection_api_key_env} environment variable not set")
        import sys
        sys.exit(1)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_log_enabled = cfg.get("logging", {}).get("detailed_metrics_log", False)
    metrics_log_path = str(output_dir / "metrics_log.jsonl") if metrics_log_enabled else None
    if metrics_log_enabled:
        print(f"Metrics logging enabled: {metrics_log_path}")

    predictions_log_enabled = cfg.get("logging", {}).get("predictions_log", True)
    global _predictions_log_path
    _predictions_log_path = str(output_dir / "predictions.jsonl") if predictions_log_enabled else None
    if predictions_log_enabled:
        print(f"Predictions logging enabled: {_predictions_log_path}")

    wlog = WandbLogger.from_config(cfg)

    failure_tracker = FailureTracker()
    failure_tracker.start()

    _thinking = cfg["model"].get("thinking")
    reflection_lm = configure_lms(
        task_model=cfg["model"]["task_lm"],
        reflection_model=cfg["model"]["reflection_lm"],
        api_base=cfg["model"].get("api_base"),
        api_key_env=api_key_env,
        task_temperature=cfg["model"].get("task_temperature", 0.7),
        task_max_tokens=cfg["model"].get("task_max_tokens", 8192),
        reflection_temperature=cfg["model"].get("reflection_temperature", 1.0),
        reflection_max_tokens=cfg["model"].get("reflection_max_tokens", 32000),
        task_thinking=cfg["model"].get("task_thinking", _thinking),
        reflection_thinking=cfg["model"].get("reflection_thinking", _thinking),
        extra_body=cfg["model"].get("extra_body"),
        aws_region_name=cfg["model"].get(
            "aws_region_name", os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        ),
        reflection_api_key_env=cfg["model"].get("reflection_api_key_env"),
        reflection_api_base=cfg["model"].get("reflection_api_base"),
    )

    return reflection_lm, failure_tracker, wlog, output_dir, metrics_log_path


def finalize_run(failure_tracker, reflection_lm, wlog, elapsed, output_dir,
                 results, metrics_log_path, plot_fn=None):
    """Common teardown: print summary, save results, generate plot, finalize wandb."""
    failure_tracker.stop()
    token_usage = get_dspy_usage(dspy.settings.lm, reflection_lm)
    results["usage"] = token_usage
    failure_summary = failure_tracker.summary()
    results["lm_failures"] = failure_summary
    results["runtime_seconds"] = round(elapsed, 1)
    results["runtime_hours"] = round(elapsed / 3600, 2)

    if metrics_log_path:
        results["metrics_log"] = read_metrics_log(metrics_log_path)
        print(f"  Metrics log: {len(results['metrics_log'])} entries")

    print(f"\nToken usage: {json.dumps(token_usage, indent=2)}")
    if failure_summary["total"] > 0:
        print(f"LM failures: {json.dumps(failure_summary, indent=2)}")

    elapsed_min = elapsed / 60
    print(f"\nTotal runtime: {elapsed_min:.1f} minutes ({elapsed:.0f}s)")

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    if plot_fn:
        plot_fn(results, output_dir)

    if wlog:
        wlog.log_usage(token_usage)
        wlog.log_failures(failure_summary)
        wlog.finish()


def print_results_table(stages, all_scores, task_names):
    """Print a formatted results table."""
    header = f"{'Stage':<20}" + "".join(f" {n:>12}" for n in task_names)
    print(f"\n{header}")
    print("-" * (20 + 13 * len(task_names)))
    for i, stage in enumerate(stages):
        row = f"{stage:<20}"
        for name in task_names:
            row += f" {all_scores[name][i]:>12.2f}"
        print(row)


def _gepa_optimize(td, current_instructions, reflection_lm, cfg, meta_prompt=None):
    """Run GEPA optimization on a single task. Returns optimized program."""
    program = td["build_program"]()
    program.predict.signature = program.predict.signature.with_instructions(current_instructions)

    # Wrap metric to include meta_prompt in feedback for reflection LM
    metric = _wrap_metric_with_meta(td["metric"], meta_prompt)

    return run_gepa(
        program=program,
        trainset=td["train_set"],
        valset=td["val_set"],
        metric=metric,
        reflection_lm=reflection_lm,
        budget=cfg["gepa"].get("budget", "light"),
        max_metric_calls=cfg["gepa"].get("max_metric_calls"),
        num_threads=cfg["gepa"]["num_threads"],
        reflection_minibatch_size=cfg["gepa"].get("reflection_minibatch_size", 3),
    )


# ---------------------------------------------------------------------------
#  Single-task runner
# ---------------------------------------------------------------------------

def run_single(cfg):
    """Run GEPA on a single task (configs with 'dataset' key, not 'tasks')."""
    from cl.config import ensure_unique_output_dir

    start_time = time.time()

    task_name = cfg.get("task_name")
    if not task_name:
        # Infer from output_dir name (before uniquifying)
        task_name = Path(cfg["output_dir"]).name.replace("gepa_", "")

    ensure_unique_output_dir(cfg)
    tasks_info = get_gepa_tasks([task_name])
    task_info = tasks_info[task_name]

    # Load dataset
    ds = cfg["dataset"]
    eval_n = ds.get("eval_n", 0)
    print(f"Loading dataset from {ds['path']}...")
    splits = task_info["loader"](
        path=ds["path"],
        train_n=ds["train_n"],
        val_n=ds["val_n"],
        seed=ds["seed"],
        eval_n=eval_n,
    )
    train_set, val_set = splits[0], splits[1]
    eval_set = splits[2] if len(splits) == 3 else None
    size_str = f"  Train: {len(train_set)}, Val: {len(val_set)}"
    if eval_set is not None:
        size_str += f", Eval: {len(eval_set)}"
    print(size_str)

    import os, sys
    api_key_env = cfg["model"].get("api_key_env", "PORTKEY_API_KEY")
    _task_lm_str = str(cfg["model"]["task_lm"])
    is_aws = _task_lm_str.startswith("sagemaker:") or _task_lm_str.startswith("bedrock:")
    if not is_aws and api_key_env not in os.environ:
        print(f"Error: {api_key_env} environment variable not set")
        sys.exit(1)
    reflection_api_key_env = cfg["model"].get("reflection_api_key_env")
    if reflection_api_key_env and reflection_api_key_env not in os.environ:
        print(f"Error: {reflection_api_key_env} environment variable not set")
        sys.exit(1)

    _thinking = cfg["model"].get("thinking")
    reflection_lm = configure_lms(
        task_model=cfg["model"]["task_lm"],
        reflection_model=cfg["model"]["reflection_lm"],
        api_base=cfg["model"].get("api_base"),
        api_key_env=api_key_env,
        task_temperature=cfg["model"].get("task_temperature", 0.7),
        task_max_tokens=cfg["model"].get("task_max_tokens", 8192),
        reflection_temperature=cfg["model"].get("reflection_temperature", 1.0),
        reflection_max_tokens=cfg["model"].get("reflection_max_tokens", 32000),
        task_thinking=cfg["model"].get("task_thinking", _thinking),
        reflection_thinking=cfg["model"].get("reflection_thinking", _thinking),
        extra_body=cfg["model"].get("extra_body"),
        aws_region_name=cfg["model"].get(
            "aws_region_name", os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        ),
        reflection_api_key_env=cfg["model"].get("reflection_api_key_env"),
        reflection_api_base=cfg["model"].get("reflection_api_base"),
    )

    gepa_cfg_single = cfg.get("gepa", {})
    seed_mode_single = cfg.get("seed_mode", "default")
    if gepa_cfg_single.get("use_system_prefix"):
        from methods.prompt_based.gepa_system_prefix import build_program_system_prefix
        program = build_program_system_prefix(task_name)
    elif gepa_cfg_single.get("use_cot_user_prefix"):
        from methods.prompt_based.gepa_cot_user_prefix import build_program_cot_user_prefix
        program = build_program_cot_user_prefix(task_name)
    elif gepa_cfg_single.get("use_raw_prefix_prompt"):
        from methods.prompt_based.gepa_raw_prefix import build_program_raw_prefix
        program = build_program_raw_prefix(task_name, seed_mode=seed_mode_single)
    elif gepa_cfg_single.get("use_oe_baseline_prompt"):
        from methods.prompt_based.gepa_oe_prompt import build_program_oe_style
        program = build_program_oe_style(task_name)
    else:
        program = task_info["build_program"]()
    skip_baseline = cfg.get("skip_baseline", False)
    num_threads = cfg["gepa"]["num_threads"]

    _baseline_metric = task_info["metric"]
    if _predictions_log_path:
        _baseline_metric = _wrap_metric_with_predictions_log(
            _baseline_metric, task_name, "baseline", _predictions_log_path,
            instructions=program.predict.signature.instructions,
        )
    # Score baseline on eval_set when available, so baseline and optimized are
    # measured on the same held-out 50 rows. (Falls back to val_set if no
    # eval_set was configured — keeps the old behavior in single-split runs.)
    _baseline_devset = eval_set if eval_set is not None else val_set
    baseline_eval = dspy.Evaluate(
        devset=_baseline_devset,
        metric=_baseline_metric,
        num_threads=num_threads,
        display_progress=True,
    )

    baseline_score = None
    if not skip_baseline:
        print(f"\n--- Baseline evaluation (unoptimized) on {'eval_set' if eval_set is not None else 'val_set'} ---")
        baseline_score = baseline_eval(program).get('score')
        print(f"Baseline: {baseline_score:.2f}")
    else:
        print("\nSkipping baseline evaluation")

    print("\n--- Running GEPA optimization ---")
    optimized_program = run_gepa(
        program=program,
        trainset=train_set,
        valset=val_set,
        metric=task_info["metric"],
        reflection_lm=reflection_lm,
        budget=cfg["gepa"].get("budget", "light"),
        max_metric_calls=cfg["gepa"].get("max_metric_calls"),
        num_threads=num_threads,
        reflection_minibatch_size=cfg["gepa"].get("reflection_minibatch_size", 3),
    )

    print("\n--- Optimized evaluation ---")
    _opt_metric = task_info["metric"]
    if _predictions_log_path:
        _opt_metric = _wrap_metric_with_predictions_log(
            _opt_metric, task_name, "optimized", _predictions_log_path,
            instructions=optimized_program.predict.signature.instructions,
        )
    if eval_set is not None:
        final_eval = dspy.Evaluate(
            devset=eval_set,
            metric=_opt_metric,
            num_threads=num_threads,
            display_progress=True,
        )
        optimized_score = final_eval(optimized_program).get('score')
    else:
        _opt_baseline_eval = dspy.Evaluate(
            devset=val_set,
            metric=_opt_metric,
            num_threads=num_threads,
            display_progress=True,
        )
        optimized_score = _opt_baseline_eval(optimized_program).get('score')
    print(f"Optimized: {optimized_score:.2f}")

    token_usage = get_dspy_usage(dspy.settings.lm, reflection_lm)
    print(f"\nToken usage: {json.dumps(token_usage, indent=2)}")

    print(f"\n{'='*40}")
    if baseline_score is not None:
        print(f"Baseline:    {baseline_score:.2f}")
    print(f"Optimized:   {optimized_score:.2f}")
    if baseline_score is not None:
        print(f"Improvement: {optimized_score - baseline_score:+.2f}")
    print(f"{'='*40}")

    elapsed = time.time() - start_time
    print(f"\nTotal runtime: {elapsed / 60:.1f} minutes ({elapsed:.0f}s)")

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "runtime_seconds": round(elapsed, 1),
        "runtime_hours": round(elapsed / 3600, 2),
        "baseline_score": baseline_score,
        "optimized_score": optimized_score,
        "usage": token_usage,
        "config": cfg,
    }
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    program_path = output_dir / "optimized_program.json"
    optimized_program.save(str(program_path))
    print(f"Optimized program saved to {program_path}")


# ---------------------------------------------------------------------------
#  Sequential runner (replace + append)
# ---------------------------------------------------------------------------

def _run_sequential_ordering(ordering, task_data, reflection_lm, cfg,
                             metrics_log_path, strategy="replace",
                             failure_tracker=None, wlog=None, eval_num_threads=None):
    """Run one sequential ordering (shared by sequential and allorders).

    strategy: "replace" — previous instructions fully replace defaults
              "append" — previous instructions are appended to defaults with META_PROMPT
    """
    num_threads = cfg["gepa"]["num_threads"]
    if eval_num_threads is None:
        eval_num_threads = cfg["gepa"].get("eval_num_threads", num_threads)
    # mean@N rollouts at the held-out eval pass. Default 1 (mean@1) matches
    # the historical behavior; set `gepa.eval_n_rollouts: 8` in the run cfg
    # for parity with the SDPO bundle's val-core/.../acc/mean@8.
    n_rollouts = int(cfg.get("gepa", {}).get("eval_n_rollouts", 1))
    task_names = list(task_data.keys())
    skip_baseline = cfg.get("skip_baseline", False)

    stages = [] if skip_baseline else ["baseline"]
    stages += [f"after_{name}" for name in ordering]
    all_scores = {name: [] for name in task_names}
    all_instructions = {}

    # Get default instructions from first task in ordering (used as starting point for optimization)
    default_program = task_data[ordering[0]]["build_program"]()
    default_instructions = default_program.predict.signature.instructions

    _last_usage = get_dspy_usage(dspy.settings.lm, reflection_lm)
    order_prefix = "_".join(ordering)

    # --- BASELINE ---
    if not skip_baseline:
        print(f"\n  {'─'*50}")
        print(f"  Baseline (each task with its own default instructions)")
        print(f"  {'─'*50}")

        if failure_tracker:
            failure_tracker.set_stage(f"{order_prefix}/baseline")

        # Evaluate each task with its own default instructions
        baseline_scores = {}
        for tn in task_names:
            task_program = task_data[tn]["build_program"]()
            task_default = task_program.predict.signature.instructions
            score = eval_instructions_on_task(
                instructions=task_default,
                build_fn=task_data[tn]["build_program"],
                val_set=task_data[tn].get("eval_set", task_data[tn]["val_set"]),
                metric=task_data[tn]["metric"],
                num_threads=eval_num_threads,
                wlog=wlog,
                task_name=tn,
                stage="baseline",
                n_rollouts=n_rollouts,
            )
            baseline_scores[tn] = score
            print(f"    {tn}: {score:.2f}")
        all_instructions["baseline"] = "(task-specific defaults)"

        if wlog:
            wlog.log_stage_scores("baseline", baseline_scores)
        for name in task_names:
            all_scores[name].append(baseline_scores[name])
        _last_usage = get_dspy_usage(dspy.settings.lm, reflection_lm)
    else:
        print("  Skipping baseline evaluation")

    # --- SEQUENTIAL OPTIMIZATION ---
    current_instructions = default_instructions
    previous_optimized = None
    meta_prompt = get_meta_prompt(cfg)

    for idx, task_name in enumerate(ordering):
        if failure_tracker:
            failure_tracker.set_stage(f"{order_prefix}/after_{task_name}")

        print(f"\n  {'─'*50}")
        print(f"  Optimize on {task_name}")
        print(f"  {'─'*50}")

        td = task_data[task_name]

        if strategy == "append":
            program = td["build_program"]()
            task_default = program.predict.signature.instructions
            if idx == 0:
                current_instructions = task_default
            else:
                current_instructions = (
                    f"{task_default}\n\n"
                    f"Previous task optimized instructions:\n{previous_optimized}"
                )
            print(f"  Initial instructions ({len(current_instructions)} chars)")

        print(f"  Running GEPA on {task_name}...")
        optimized_program = _gepa_optimize(td, current_instructions, reflection_lm, cfg,
                                           meta_prompt=meta_prompt)

        current_instructions = optimized_program.predict.signature.instructions
        previous_optimized = current_instructions
        all_instructions[f"after_{task_name}"] = current_instructions
        print(f"  Optimized instructions ({len(current_instructions)} chars)")

        _pre_eval_usage = get_dspy_usage(dspy.settings.lm, reflection_lm)

        print(f"\n  Evaluating on all tasks after {task_name} optimization:")
        stage_scores = eval_all_tasks(current_instructions, task_data, eval_num_threads,
                                      wlog=wlog, stage=f"after_{task_name}",
                                      n_rollouts=n_rollouts)
        for name in task_names:
            all_scores[name].append(stage_scores[name])

        if metrics_log_path:
            _post_eval_usage = get_dspy_usage(dspy.settings.lm, reflection_lm)
            log_stage(
                metrics_log_path, f"after_{task_name}", "gepa", stage_scores,
                _post_eval_usage,
                optimization_usage=usage_diff(_pre_eval_usage, _last_usage),
                eval_usage=usage_diff(_post_eval_usage, _pre_eval_usage),
            )
            _last_usage = _post_eval_usage

    return {
        "ordering": list(ordering),
        "stages": stages,
        "scores": all_scores,
        "instructions": all_instructions,
    }


def run_sequential(cfg, strategy="replace"):
    """Run sequential optimization: optimize on each task in order.

    strategy: "replace" or "append"

    Tasks with `optimize: false` in the cfg are loaded into task_data (so
    `eval_all_tasks` scores them every stage) but excluded from the
    optimization ordering. Used for "stable knowledge" probes like
    temporalwiki_stable that should be measured but never trained on.
    """
    import sys
    start_time = time.time()
    task_names = [t["name"] for t in cfg["tasks"]]
    tasks_info = get_gepa_tasks(task_names)

    reflection_lm, failure_tracker, wlog, output_dir, metrics_log_path = setup_run(cfg)
    task_data = load_all_datasets(cfg, tasks_info)

    optimization_order = [t["name"] for t in cfg["tasks"] if t.get("optimize", True)]
    if not optimization_order:
        sys.exit("error: no tasks to optimize on (all marked optimize: false)")
    eval_only = [n for n in task_names if n not in optimization_order]
    if eval_only:
        print(f"Eval-only tasks (skipped in optimization ordering): {eval_only}")
    result = _run_sequential_ordering(
        ordering=optimization_order,
        task_data=task_data,
        reflection_lm=reflection_lm,
        cfg=cfg,
        metrics_log_path=metrics_log_path,
        strategy=strategy,
        failure_tracker=failure_tracker,
        wlog=wlog,
    )

    elapsed = time.time() - start_time
    print_results_table(result["stages"], result["scores"], task_names)

    results = {
        "strategy": strategy,
        "stages": result["stages"],
        "scores": result["scores"],
        "instructions": result["instructions"],
        "config": cfg,
    }

    def _plot(results, output_dir):
        plot_path = output_dir / "sequential_performance.png"
        plot_sequential(results, str(plot_path))

    finalize_run(failure_tracker, reflection_lm, wlog, elapsed, output_dir,
                 results, metrics_log_path, plot_fn=_plot)


def run_allorders(cfg, strategy="replace", ordering_indices=None):
    """Run sequential optimization across all task orderings.

    Generates all permutations of the task list from the config.
    strategy: "replace" or "append"
    ordering_indices: list of 1-based indices to run, or None for all.
    """
    start_time = time.time()
    task_names = [t["name"] for t in cfg["tasks"]]
    tasks_info = get_gepa_tasks(task_names)

    all_orderings = list(itertools.permutations(task_names))
    n_orderings = len(all_orderings)

    if ordering_indices is None:
        ordering_indices = list(range(1, n_orderings + 1))
    else:
        for idx in ordering_indices:
            if idx < 1 or idx > n_orderings:
                print(f"Error: ordering index {idx} out of range (1-{n_orderings})")
                import sys; sys.exit(1)

    reflection_lm, failure_tracker, wlog, output_dir, metrics_log_path = setup_run(cfg)
    task_data = load_all_datasets(cfg, tasks_info)

    orderings_to_run = [(i, all_orderings[i - 1]) for i in ordering_indices]
    print(f"\nRunning {len(orderings_to_run)}/{n_orderings} ordering(s): {[i for i, _ in orderings_to_run]}")

    all_order_results = {}
    for i, ordering in orderings_to_run:
        order_label = f"order_{i}"
        order_str = " → ".join(ordering)
        print(f"\n{'='*60}")
        print(f"ORDER {i}/{n_orderings}: {order_str}")
        print(f"{'='*60}")

        result = _run_sequential_ordering(
            ordering=ordering,
            task_data=task_data,
            reflection_lm=reflection_lm,
            cfg=cfg,
            metrics_log_path=metrics_log_path,
            strategy=strategy,
            failure_tracker=failure_tracker,
            wlog=wlog,
        )
        all_order_results[order_label] = result

        order_dir = output_dir / f"order_{i}_{'_'.join(ordering)}"
        order_dir.mkdir(parents=True, exist_ok=True)
        with open(order_dir / "results.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Order {i} results saved to {order_dir}/results.json")

    elapsed = time.time() - start_time

    # Print combined results
    print(f"\n{'='*60}")
    print("COMBINED RESULTS")
    print(f"{'='*60}")
    for i, ordering in orderings_to_run:
        result = all_order_results[f"order_{i}"]
        print(f"\nOrder {i}: {' → '.join(ordering)}")
        print_results_table(result["stages"], result["scores"], task_names)

    results = {
        "orderings": all_order_results,
        "config": cfg,
    }

    def _plot(results, output_dir):
        if len(orderings_to_run) == n_orderings:
            plot_path = output_dir / "allorders_performance.png"
            plot_allorders(results["orderings"], str(plot_path))
        else:
            print(f"\nRan {len(orderings_to_run)}/6 orderings. Use scripts/merge_allorders.py to combine and plot.")

    finalize_run(failure_tracker, reflection_lm, wlog, elapsed, output_dir,
                 results, metrics_log_path, plot_fn=_plot)


# ---------------------------------------------------------------------------
#  Mixed (round-robin) runner
# ---------------------------------------------------------------------------

def _split_into_minibatches(data, n_batches):
    """Split a list into n roughly equal mini-batches."""
    batch_size = len(data) // n_batches
    remainder = len(data) % n_batches
    batches = []
    start = 0
    for i in range(n_batches):
        end = start + batch_size + (1 if i < remainder else 0)
        batches.append(data[start:end])
        start = end
    return batches


def run_mixed(cfg):
    """Run mixed (round-robin) task interleaving optimization."""
    start_time = time.time()
    task_names = [t["name"] for t in cfg["tasks"]]
    tasks_info = get_gepa_tasks(task_names)

    reflection_lm, failure_tracker, wlog, output_dir, metrics_log_path = setup_run(cfg)
    task_data = load_all_datasets(cfg, tasks_info)

    num_threads = cfg["gepa"]["num_threads"]
    eval_num_threads = cfg["gepa"].get("eval_num_threads", num_threads)

    mixed_cfg = cfg["mixed"]
    rounds_per_task = mixed_cfg["rounds_per_task"]
    max_metric_calls_per_round = mixed_cfg.get("max_metric_calls_per_round", 300)
    seed = mixed_cfg.get("seed", 42)

    # Split training data into mini-batches
    task_batches = {}
    for name in task_names:
        task_batches[name] = _split_into_minibatches(
            task_data[name]["train_set"], rounds_per_task
        )
        print(f"  {name}: {rounds_per_task} mini-batches, "
              f"sizes: {[len(b) for b in task_batches[name]]}")

    # Build schedule
    rng = random.Random(seed)
    task_order = list(task_names)
    rng.shuffle(task_order)
    total_rounds = rounds_per_task * len(task_names)
    schedule = task_order * rounds_per_task
    checkpoint_interval = len(task_names)

    print(f"\nSchedule ({total_rounds} rounds, eval every {checkpoint_interval}):")
    print(f"  Task order: {' → '.join(task_order)}")

    task_round_idx = {name: 0 for name in task_names}

    # Get default instructions
    default_program = task_data[task_names[0]]["build_program"]()
    current_instructions = default_program.predict.signature.instructions

    skip_baseline = cfg.get("skip_baseline", False)
    all_scores = {name: [] for name in task_names}
    all_instructions = {}
    stages = []

    _last_usage = get_dspy_usage(dspy.settings.lm, reflection_lm)

    if not skip_baseline:
        print(f"\n{'='*60}")
        print("Baseline (default instructions)")
        print(f"{'='*60}")
        stages.append("baseline")
        failure_tracker.set_stage("baseline")
        all_instructions["baseline"] = current_instructions
        baseline_scores = eval_all_tasks(current_instructions, task_data, eval_num_threads,
                                         wlog=wlog, stage="baseline")
        for name in task_names:
            all_scores[name].append(baseline_scores[name])
        _last_usage = get_dspy_usage(dspy.settings.lm, reflection_lm)
    else:
        print("\nSkipping baseline evaluation")

    # --- MIXED ROUND-ROBIN ---
    for round_num in range(total_rounds):
        task_name = schedule[round_num]
        cycle = round_num // len(task_names) + 1
        failure_tracker.set_stage(f"cycle_{cycle}/{task_name}")
        batch_idx = task_round_idx[task_name]
        mini_batch = task_batches[task_name][batch_idx]
        task_round_idx[task_name] += 1

        print(f"\n{'─'*50}")
        print(f"  Round {round_num + 1}/{total_rounds} "
              f"(cycle {cycle}, {task_name}, batch {batch_idx + 1}/{rounds_per_task}, "
              f"{len(mini_batch)} samples)")
        print(f"{'─'*50}")

        td = task_data[task_name]
        program = td["build_program"]()
        program.predict.signature = program.predict.signature.with_instructions(
            current_instructions
        )

        optimized_program = run_gepa(
            program=program,
            trainset=mini_batch,
            valset=td["val_set"],
            metric=td["metric"],
            reflection_lm=reflection_lm,
            max_metric_calls=max_metric_calls_per_round,
            num_threads=num_threads,
            reflection_minibatch_size=cfg["gepa"].get("reflection_minibatch_size", 3),
        )

        current_instructions = optimized_program.predict.signature.instructions
        print(f"  Instructions updated ({len(current_instructions)} chars)")

        # Evaluate at checkpoint
        if (round_num + 1) % checkpoint_interval == 0:
            checkpoint_label = f"cycle_{cycle}"
            stages.append(checkpoint_label)
            all_instructions[checkpoint_label] = current_instructions

            _pre_eval_usage = get_dspy_usage(dspy.settings.lm, reflection_lm)

            print(f"\n  *** Checkpoint after cycle {cycle} ***")
            checkpoint_scores = eval_all_tasks(
                current_instructions, task_data, eval_num_threads,
                wlog=wlog, stage=checkpoint_label,
            )
            for name in task_names:
                all_scores[name].append(checkpoint_scores[name])

            if metrics_log_path:
                _post_eval_usage = get_dspy_usage(dspy.settings.lm, reflection_lm)
                log_stage(
                    metrics_log_path, checkpoint_label, "gepa_mixed",
                    checkpoint_scores, _post_eval_usage,
                    optimization_usage=usage_diff(_pre_eval_usage, _last_usage),
                    eval_usage=usage_diff(_post_eval_usage, _pre_eval_usage),
                )
                _last_usage = _post_eval_usage

    elapsed = time.time() - start_time
    print_results_table(stages, all_scores, task_names)

    results = {
        "stages": stages,
        "scores": all_scores,
        "instructions": all_instructions,
        "schedule": schedule,
        "task_order": task_order,
        "rounds_per_task": rounds_per_task,
        "config": cfg,
    }

    def _plot(results, output_dir):
        plot_path = output_dir / "mixed_performance.png"
        plot_sequential(results, str(plot_path))

    finalize_run(failure_tracker, reflection_lm, wlog, elapsed, output_dir,
                 results, metrics_log_path, plot_fn=_plot)
