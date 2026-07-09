"""OpenEvolve runner v2 — uses delimited [GENERAL_INSTRUCTIONS] blocks.

Replaces the fragile clean_evolved_text() stripping approach with explicit
delimiters that separate the evolvable general reasoning strategy from the
fixed task-specific template. This makes cross-task transfer reliable.

Use via: python scripts/run.py --method openevolve-v2 ...

All other functionality (single-task, mixed, scoring, eval) is inherited
from the original runner — only the sequential ordering logic changes.
"""

import itertools
import json
import os
import re
import sys
import time
from pathlib import Path

import openevolve
from openai import OpenAI

# Re-use everything from the original runner except the sequential logic
from methods.prompt_based.runners.openevolve_runner import (
    _call_llm,
    _extract_answer,
    _extract_label,
    _extract_mcq_answer,
    _score_example,
    score_on_task,
    eval_all_tasks,
    write_task_config,
    strip_placeholders,
    load_all_datasets_raw,
    setup_run,
    print_results_table,
    run_single as _v1_run_single,
    run_mixed,
    apply_thinking_to_oe_dict,
    _failure_tracker,
    _wandb_logger,
)
from methods.prompt_based.runners.meta_prompt import get_meta_prompt
from cl.tasks import TASK_REGISTRY, get_openevolve_tasks
from cl.utils.token_tracker import usage_diff
from cl.utils.metrics_logger import log_stage, read_metrics_log
from cl.utils.plotting import plot_sequential, plot_allorders

CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "configs"

DEFAULT_SYSTEM_MESSAGE_PATH = CONFIGS_DIR / "methods" / "openevolve_system_message.txt"


def _build_system_message(cfg, meta_prompt):
    """Build the evolution LM system message with meta_prompt injected.

    Sources (in priority order):
    1. cfg["openevolve"]["prompt"]["system_message"] if it's raw text (not a template name)
    2. Default template from configs/methods/openevolve_system_message.txt
    """
    # Check if config already has a custom system message that's raw text
    oe_prompt_cfg = cfg.get("openevolve", {}).get("prompt", {})
    custom_msg = oe_prompt_cfg.get("system_message")
    if custom_msg and len(custom_msg) > 50:  # raw text, not a template name
        if meta_prompt and "{meta_prompt}" in custom_msg:
            return custom_msg.format(meta_prompt=f"\nIMPORTANT: {meta_prompt}")
        return custom_msg

    # Load default template
    if DEFAULT_SYSTEM_MESSAGE_PATH.exists():
        template = DEFAULT_SYSTEM_MESSAGE_PATH.read_text().strip()
    else:
        template = (
            "You are an expert prompt engineer. Preserve [GENERAL_INSTRUCTIONS] "
            "delimiters and all {placeholders} exactly. Focus improvements on the "
            "text inside the [GENERAL_INSTRUCTIONS] block.\n\n{meta_prompt}"
        )

    meta_section = f"\nIMPORTANT: {meta_prompt}" if meta_prompt else ""
    return template.format(meta_prompt=meta_section)


# ---------------------------------------------------------------------------
#  Delimited prompt helpers
# ---------------------------------------------------------------------------

GENERAL_START = "[GENERAL_INSTRUCTIONS]"
GENERAL_END = "[/GENERAL_INSTRUCTIONS]"


def build_delimited_prompt(general_instructions, task_name):
    """Build a prompt with delimited general instructions + fixed task template.

    Output format:
        [GENERAL_INSTRUCTIONS]
        <general reasoning strategy — this part gets evolved and carried>
        [/GENERAL_INSTRUCTIONS]

        <task-specific template with {placeholders} — fixed per task>
    """
    template = TASK_REGISTRY[task_name].get("template", "")
    return (
        f"{GENERAL_START}\n"
        f"{general_instructions}\n"
        f"{GENERAL_END}\n\n"
        f"{template}"
    )


def run_single(cfg, task_name=None):
    """openevolve-v2 single-task entry point.

    Materializes a delimited initial prompt (the v1 registry's initial_prompt.txt
    lacks [GENERAL_INSTRUCTIONS] markers that the v2 evaluators now require) and
    delegates to the v1 run_single flow via the _initial_prompt_override hook.
    """
    from cl.config import ensure_unique_output_dir

    if task_name is None:
        task_name = cfg.get("task_name")
    if task_name is None:
        raise ValueError("openevolve-v2 run_single requires a task_name")

    ensure_unique_output_dir(cfg)
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    default_general = TASK_REGISTRY[task_name].get("default_instruction", "")
    delimited = build_delimited_prompt(default_general, task_name)
    prompt_file = output_dir / f"initial_prompt_v2_{task_name}.txt"
    prompt_file.write_text(delimited)
    cfg["_initial_prompt_override"] = str(prompt_file)

    return _v1_run_single(cfg, task_name=task_name)


def extract_general_instructions(evolved_text):
    """Extract the general instructions from between delimiters.

    Falls back to clean_evolved_text() if delimiters are missing
    (e.g. evolution mutated them away).
    """
    match = re.search(
        rf"{re.escape(GENERAL_START)}\s*\n(.*?)\n\s*{re.escape(GENERAL_END)}",
        evolved_text,
        re.DOTALL,
    )
    if match:
        return match.group(1).strip()

    # Fallback: try partial matches (e.g. only opening tag survived)
    match = re.search(
        rf"{re.escape(GENERAL_START)}\s*\n(.*?)$",
        evolved_text,
        re.DOTALL,
    )
    if match:
        # Strip any remaining task template content heuristically
        text = match.group(1).strip()
        # Remove everything after common template markers
        for marker in ["\nContext:", "\nClaim:", "\nQuestion:", "\n{", "Action:", "####"]:
            idx = text.find(marker)
            if idx > 0:
                text = text[:idx].strip()
        if text:
            print(f"  Warning: only opening delimiter found, extracted {len(text)} chars")
            return text

    # Last resort: return the whole text stripped of obvious template parts
    print(f"  Warning: no delimiters found in evolved text, using full text as fallback")
    from methods.prompt_based.runners.openevolve_runner import clean_evolved_text
    return clean_evolved_text(evolved_text)


# ---------------------------------------------------------------------------
#  Sequential runner (replace + append) with delimiters
# ---------------------------------------------------------------------------

def _run_sequential_ordering(ordering, task_data, task_cfgs, client, model, cfg,
                             output_dir, metrics_log_path, tracker, strategy="replace",
                             eval_num_threads=8, order_label=None):
    """Run one sequential ordering using delimited prompts."""
    tasks_info = get_openevolve_tasks(list(task_data.keys()))
    task_names = list(task_data.keys())

    # Create per-ordering subdirectory
    if order_label:
        ordering_dir = output_dir / f"{order_label}_{'_'.join(ordering)}"
    else:
        ordering_dir = output_dir
    ordering_dir.mkdir(parents=True, exist_ok=True)
    skip_baseline = cfg.get("skip_baseline", False)

    stages = [] if skip_baseline else ["baseline"]
    stages += [f"after_{name}" for name in ordering]
    all_scores = {name: [] for name in task_names}
    all_instructions = {}
    all_prompts = {}

    _last_usage = tracker.get_usage()

    # --- BASELINE ---
    if not skip_baseline:
        print(f"\n  {'─'*50}")
        print(f"  Baseline (each task with its own default instructions)")
        print(f"  {'─'*50}")

        baseline_scores = {}
        for tn in task_names:
            default_instruction = TASK_REGISTRY[tn].get("default_instruction", "")
            print(f"    Evaluating {tn} with its default instructions...")
            score = score_on_task(default_instruction, tn, task_data[tn]["eval_set"],
                                 client, model, num_threads=eval_num_threads,
                                 stage="baseline")
            baseline_scores[tn] = score
            print(f"    {tn}: {score:.2f}")

        for name in task_names:
            all_scores[name].append(baseline_scores[name])
        all_instructions["baseline"] = "(task-specific defaults)"
        _last_usage = tracker.get_usage()
    else:
        print("  Skipping baseline evaluation")

    # --- SEQUENTIAL OPTIMIZATION ---
    current_general = None
    previous_general = None
    meta_prompt = get_meta_prompt(cfg)

    # Load and configure system message for evolution LM
    system_message = _build_system_message(cfg, meta_prompt)
    if system_message:
        print(f"  Evolution LM system message: {len(system_message)} chars")

    for idx, task_name in enumerate(ordering):
        from methods.prompt_based.runners.openevolve_runner import _failure_tracker
        if _failure_tracker:
            _failure_tracker.set_stage(f"after_{task_name}")

        print(f"\n  {'─'*50}")
        print(f"  Optimize on {task_name}")
        print(f"  {'─'*50}")

        task_default = TASK_REGISTRY[task_name].get("default_instruction", "")

        # Build general instructions (META_PROMPT no longer in task prompt —
        # it's in the evolution LM system message and GEPA metric feedback instead)
        if strategy == "append":
            if idx == 0:
                general = task_default
            else:
                general = (
                    f"{task_default}\n\n"
                    f"Previous task optimized instructions:\n{strip_placeholders(previous_general)}"
                )
        else:  # replace
            if current_general is None:
                general = task_default
            else:
                general = current_general

        # Build delimited prompt: [GENERAL_INSTRUCTIONS]...[/GENERAL_INSTRUCTIONS] + template
        initial_prompt_text = build_delimited_prompt(general, task_name)

        prompt_file = ordering_dir / f"initial_prompt_{task_name}.txt"
        with open(prompt_file, "w") as f:
            f.write(initial_prompt_text)
        print(f"  Initial prompt ({len(initial_prompt_text)} chars, general section: {len(general)} chars)")

        # Write task-specific evaluator config
        task_cfg = next(tc for tc in cfg["tasks"] if tc["name"] == task_name)
        task_config_path = write_task_config(task_cfg, cfg["model"],
                                             ordering_dir / f"config_{task_name}.yaml",
                                             eval_num_threads=eval_num_threads)
        os.environ["BENCHMARK_CONFIG"] = task_config_path

        oe_dict = cfg["openevolve"].copy()
        # Inject system message into OpenEvolve prompt config
        if system_message:
            oe_dict.setdefault("prompt", {})
            oe_dict["prompt"]["system_message"] = system_message
        apply_thinking_to_oe_dict(oe_dict, cfg)
        oe_config = openevolve.Config.from_dict(oe_dict)

        print(f"  Running OpenEvolve on {task_name}...")
        stage_output_dir = str(ordering_dir / f"evolution_{task_name}")
        result = openevolve.run_evolution(
            initial_program=str(prompt_file),
            evaluator=tasks_info[task_name]["evaluator"],
            config=oe_config,
            output_dir=stage_output_dir,
        )
        print(f"  Evolution complete. Best train score: {result.best_score:.4f}")

        # Extract general instructions from evolved prompt
        evolved_text = result.best_code
        all_prompts[f"after_{task_name}"] = evolved_text
        current_general = extract_general_instructions(evolved_text)
        previous_general = current_general
        all_instructions[f"after_{task_name}"] = current_general
        print(f"  Extracted general instructions ({len(current_general)} chars)")

        _pre_eval_usage = tracker.get_usage()

        # Cross-task evaluation: build each task's prompt using extracted general + task template
        print(f"\n  Evaluating on all tasks after {task_name} optimization:")
        stage_scores = eval_all_tasks(current_general, task_data, client, model,
                                      num_threads=eval_num_threads, stage=f"after_{task_name}")
        for name in task_names:
            all_scores[name].append(stage_scores[name])

        if metrics_log_path:
            _post_eval_usage = tracker.get_usage()
            log_stage(
                metrics_log_path, f"after_{task_name}", "openevolve_v2",
                stage_scores, _post_eval_usage,
                optimization_usage=usage_diff(_pre_eval_usage, _last_usage),
                eval_usage=usage_diff(_post_eval_usage, _pre_eval_usage),
            )
            _last_usage = _post_eval_usage

    return {
        "ordering": list(ordering),
        "stages": stages,
        "scores": all_scores,
        "instructions": all_instructions,
        "evolved_prompts": all_prompts,
    }


# ---------------------------------------------------------------------------
#  Public API — same signatures as openevolve_runner.py
# ---------------------------------------------------------------------------

def run_sequential(cfg, strategy="replace"):
    """Run sequential optimization with delimited prompts."""
    start_time = time.time()
    task_names = [t["name"] for t in cfg["tasks"]]

    client, model, tracker, failure_tracker, wlog, output_dir, metrics_log_path, eval_num_threads = setup_run(cfg)
    task_data = load_all_datasets_raw(cfg, get_openevolve_tasks(task_names), drop_val=True)
    task_cfgs = {tc["name"]: tc for tc in cfg["tasks"]}

    with tracker.track_to_file():
        result = _run_sequential_ordering(
            ordering=task_names,
            task_data=task_data,
            task_cfgs=task_cfgs,
            client=client,
            model=model,
            cfg=cfg,
            output_dir=output_dir,
            metrics_log_path=metrics_log_path,
            tracker=tracker,
            strategy=strategy,
            eval_num_threads=eval_num_threads,
        )

    elapsed = time.time() - start_time
    print_results_table(result["stages"], result["scores"], task_names)

    token_usage = tracker.get_usage()
    failure_summary = failure_tracker.summary() if failure_tracker else {"total": 0}

    print(f"\nToken usage: {json.dumps(token_usage, indent=2)}")
    if failure_summary["total"] > 0:
        print(f"LM failures: {json.dumps(failure_summary, indent=2)}")
    print(f"\nTotal runtime: {elapsed / 60:.1f} minutes ({elapsed:.0f}s)")

    results = {
        "runtime_seconds": round(elapsed, 1),
        "runtime_hours": round(elapsed / 3600, 2),
        "stages": result["stages"],
        "scores": result["scores"],
        "instructions": result["instructions"],
        "evolved_prompts": result.get("evolved_prompts", {}),
        "usage": token_usage,
        "lm_failures": failure_summary,
        "config": cfg,
    }
    if metrics_log_path:
        results["metrics_log"] = read_metrics_log(metrics_log_path)

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    plot_path = output_dir / "sequential_performance.png"
    plot_sequential(results, str(plot_path))

    if wlog:
        wlog.log_usage(token_usage)
        wlog.log_failures(failure_summary)
        wlog.finish()


def run_allorders(cfg, strategy="replace", ordering_indices=None):
    """Run sequential optimization across all task orderings with delimited prompts."""
    start_time = time.time()
    task_names = [t["name"] for t in cfg["tasks"]]

    all_orderings = list(itertools.permutations(task_names))
    n_orderings = len(all_orderings)

    if ordering_indices is None:
        ordering_indices = list(range(1, n_orderings + 1))
    else:
        for idx in ordering_indices:
            if idx < 1 or idx > n_orderings:
                print(f"Error: ordering index {idx} out of range (1-{n_orderings})")
                sys.exit(1)

    client, model, tracker, failure_tracker, wlog, output_dir, metrics_log_path, eval_num_threads = setup_run(cfg)
    task_data = load_all_datasets_raw(cfg, get_openevolve_tasks(task_names), drop_val=True)
    task_cfgs = {tc["name"]: tc for tc in cfg["tasks"]}

    orderings_to_run = [(i, all_orderings[i - 1]) for i in ordering_indices]
    print(f"\nRunning {len(orderings_to_run)}/{n_orderings} ordering(s): {[i for i, _ in orderings_to_run]}")

    all_order_results = {}

    with tracker.track_to_file():
        for i, ordering in orderings_to_run:
            order_label = f"order_{i}"
            order_str = " → ".join(ordering)
            print(f"\n{'='*60}")
            print(f"ORDER {i}/{n_orderings}: {order_str}")
            print(f"{'='*60}")

            result = _run_sequential_ordering(
                ordering=ordering,
                task_data=task_data,
                task_cfgs=task_cfgs,
                client=client,
                model=model,
                cfg=cfg,
                output_dir=output_dir,
                metrics_log_path=metrics_log_path,
                tracker=tracker,
                strategy=strategy,
                eval_num_threads=eval_num_threads,
                order_label=order_label,
            )
            all_order_results[order_label] = result

            order_dir = output_dir / f"{order_label}_{'_'.join(ordering)}"
            with open(order_dir / "results.json", "w") as f:
                json.dump(result, f, indent=2)
            print(f"  Order {i} results saved to {order_dir}/results.json")

    elapsed = time.time() - start_time
    token_usage = tracker.get_usage()
    failure_summary = failure_tracker.summary() if failure_tracker else {"total": 0}

    print(f"\n{'='*60}")
    print("COMBINED RESULTS")
    print(f"{'='*60}")
    for i, ordering in orderings_to_run:
        result = all_order_results[f"order_{i}"]
        print(f"\nOrder {i}: {' → '.join(ordering)}")
        print_results_table(result["stages"], result["scores"], task_names)

    print(f"\nToken usage: {json.dumps(token_usage, indent=2)}")
    print(f"\nTotal runtime: {elapsed / 60:.1f} minutes ({elapsed:.0f}s)")

    results = {
        "runtime_seconds": round(elapsed, 1),
        "runtime_hours": round(elapsed / 3600, 2),
        "orderings": all_order_results,
        "usage": token_usage,
        "lm_failures": failure_summary,
        "config": cfg,
    }
    if metrics_log_path:
        results["metrics_log"] = read_metrics_log(metrics_log_path)

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    if len(orderings_to_run) == n_orderings:
        plot_path = output_dir / "allorders_performance.png"
        plot_allorders(results["orderings"], str(plot_path))
    else:
        print(f"\nRan {len(orderings_to_run)}/{n_orderings} orderings. Use scripts/merge_allorders.py to combine and plot.")

    if wlog:
        wlog.log_usage(token_usage)
        wlog.log_failures(failure_summary)
        wlog.finish()
