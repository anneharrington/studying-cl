"""OpenEvolve runner v2-meta-single — single-task meta-prompt evolution.

Experiment design: measure whether `meta_prompt` accumulation across tasks
improves optimization of *the task currently being optimized*, isolating the
`meta_prompt` as the sole cross-task information carrier.

Key differences from v2-meta:
  1. `[GENERAL_INSTRUCTIONS]` is RESET to the task's default at the start of
     every task — it never carries across tasks.
  2. After OpenEvolve finishes, only the CURRENT task is evaluated (single-task
     post-opt val score). No cross-task eval.
  3. The meta-evolve LLM sees only the trajectory of prior (task_name, score)
     pairs + its current meta_prompt. It does NOT see the evolved [GI] block
     (avoiding verbatim-copy leakage), nor any cross-task score matrix.
  4. Results include a position×task matrix suitable for comparing "task X at
     position 1 (cold meta) vs. task X at position 3 (warm meta)".

Static-meta control arm: set `meta_prompt.evolution.enabled: false` to keep
the meta_prompt frozen across tasks — this gives the positional reference
against which the evolving-meta arm is compared.

Use via: python scripts/run.py --method openevolve-v2-meta-single ...
"""

import itertools
import json
import os
import sys
import time
from pathlib import Path

import openevolve

from methods.prompt_based.runners.openevolve_runner import (
    _call_llm,
    score_on_task,
    write_task_config,
    load_all_datasets_raw,
    setup_run,
    print_results_table,
    apply_thinking_to_oe_dict,
)
from methods.prompt_based.runners.openevolve_runner_v2 import (
    _build_system_message,
    build_delimited_prompt,
    extract_general_instructions,
)
from methods.prompt_based.runners.openevolve_runner_v2_meta import _get_meta_evolution_cfg
from methods.prompt_based.runners.meta_prompt import get_meta_prompt
from cl.tasks import TASK_REGISTRY, get_openevolve_tasks
from cl.utils.token_tracker import usage_diff
from cl.utils.metrics_logger import log_stage, read_metrics_log
from cl.utils.plotting import plot_meta_single, plot_meta_single_allorders


# ---------------------------------------------------------------------------
#  Meta-prompt evolution (single-task framing)
# ---------------------------------------------------------------------------

_META_EVOLVE_HEADER_SINGLE = """You are tuning a meta-strategy that guides an LLM evolving task-specific prompts. Each task is optimized independently — the [GENERAL_INSTRUCTIONS] block is RESET to the task default at the start of every task. Your meta-strategy is the ONLY channel by which lessons from one task's optimization can influence the next.

A new optimization round just finished. Here is what happened:

Task just optimized: {task_name}

Optimization trajectory so far (chronological, task → post-opt val score on that task):
{trajectory}

Your CURRENT meta-strategy:
\"\"\"
{current_meta_prompt}
\"\"\"

You are NOT shown the evolved prompts themselves — only scores. This is intentional: you must abstract generalizable reasoning skills rather than copy task-specific phrasing. The goal is for your meta-strategy to help the evolution LLM converge faster and higher on the NEXT task, whatever it turns out to be."""

_META_EVOLVE_CONSTRAINTS_SINGLE = """Constraints:
- Keep it concise (under 250 words). It is injected into a system message every iteration; bloat is costly.
- It is GUIDANCE, not a prompt template — do NOT include placeholders like {{context}} or {{prompt}}.
- Abstract concrete techniques into task-agnostic heuristics. Do NOT embed task-specific vocabulary (dataset names, field names, output keywords).
- Speak in the imperative voice to the evolution LLM ("When optimizing X, prefer Y...").
- Grow into a compact skills catalogue over rounds, not a kitchen-sink transcript.

Return ONLY the updated meta-strategy text, no preamble, no markdown fences."""


META_EVOLVE_TEMPLATE_SINGLE_MINIMAL = (
    _META_EVOLVE_HEADER_SINGLE
    + "\n\nRevise the meta-strategy so the next round's evolution (on a task you have not seen yet) does better.\n\n"
    + _META_EVOLVE_CONSTRAINTS_SINGLE
)


META_EVOLVE_TEMPLATE_SINGLE_GUIDED = (
    _META_EVOLVE_HEADER_SINGLE
    + """

Revise the meta-strategy so the next round's evolution (on a task you have not seen yet) does better. The revised meta-strategy should help the evolution LLM:
1. Recognize what TYPE of task is in front of it (reading comprehension, math, multi-step reasoning, format-following, tool use, etc.) and match its reasoning pattern to the type.
2. Apply generalizable skills (step-by-step decomposition, self-verification, structured intermediate reasoning, careful reading of the prompt) that have proved useful on prior rounds.
3. Avoid over-fitting the meta-strategy to the last task seen — each new task deserves a fresh structural frame, not a recycled one.

"""
    + _META_EVOLVE_CONSTRAINTS_SINGLE
)


def _select_template_single(template_style):
    if template_style == "guided":
        return META_EVOLVE_TEMPLATE_SINGLE_GUIDED
    if template_style == "minimal":
        return META_EVOLVE_TEMPLATE_SINGLE_MINIMAL
    raise ValueError(
        f"Unknown meta_prompt.evolution.template_style: {template_style!r} "
        f"(expected 'minimal' or 'guided')"
    )


def _format_trajectory(trajectory):
    """Render trajectory list as a compact numbered listing."""
    if not trajectory:
        return "(none yet — this was the first round)"
    lines = []
    for i, entry in enumerate(trajectory, start=1):
        lines.append(f"  {i}. {entry['task']} → {entry['score']:.2f}")
    return "\n".join(lines)


def evolve_meta_prompt_single(current_meta_prompt, task_name, trajectory,
                              client, model, template_style="minimal",
                              max_chars=4000):
    """Rewrite the meta_prompt from just the (task, score) trajectory.

    Deliberately does NOT expose the evolved [GI] block — the meta-LLM must
    abstract strategy from scores alone, preventing verbatim copying of
    task-specific phrasing into the meta_prompt.
    """
    template = _select_template_single(template_style)
    prompt = template.format(
        task_name=task_name,
        trajectory=_format_trajectory(trajectory),
        current_meta_prompt=current_meta_prompt or "(none yet)",
    )

    response = _call_llm(client, model, prompt)
    new_mp = (response or "").strip()
    if new_mp.startswith("```"):
        lines = new_mp.splitlines()
        new_mp = "\n".join(l for l in lines if not l.startswith("```")).strip()

    if not new_mp:
        print("    [META] Empty response from meta-evolve LLM; keeping current meta_prompt")
        return current_meta_prompt
    if len(new_mp) > max_chars:
        print(f"    [META] Meta-prompt too long ({len(new_mp)} chars); truncating to {max_chars}")
        new_mp = new_mp[:max_chars]
    print(f"    [META] meta_prompt updated ({len(current_meta_prompt or '')} -> {len(new_mp)} chars)")
    return new_mp


# ---------------------------------------------------------------------------
#  Sequential runner with single-task meta-prompt evolution
# ---------------------------------------------------------------------------

def _run_sequential_ordering_meta_single(ordering, task_data, task_cfgs,
                                         client, model, cfg, output_dir,
                                         metrics_log_path, tracker,
                                         eval_num_threads=8, order_label=None):
    """Sequential run with [GI] reset between tasks and trajectory-only meta feedback."""
    tasks_info = get_openevolve_tasks(list(task_data.keys()))
    task_names = list(task_data.keys())

    if order_label:
        ordering_dir = output_dir / f"{order_label}_{'_'.join(ordering)}"
    else:
        ordering_dir = output_dir
    ordering_dir.mkdir(parents=True, exist_ok=True)
    skip_baseline = cfg.get("skip_baseline", False)

    position_scores = []  # [{task, position, train_best_score, post_opt_val_score}]
    trajectory = []       # [{task, score}] — feeds meta-evolve
    meta_prompt_history = []

    _last_usage = tracker.get_usage()

    # --- BASELINE (each task with its default; a "cold" per-task reference) ---
    baseline_scores = {}
    if not skip_baseline:
        print(f"\n  {'─'*50}")
        print(f"  Baseline — each task scored with its own default instructions")
        print(f"  {'─'*50}")
        for tn in task_names:
            default_instruction = TASK_REGISTRY[tn].get("default_instruction", "")
            print(f"    Evaluating {tn} with its default instructions...")
            baseline_scores[tn] = score_on_task(
                default_instruction, tn, task_data[tn]["eval_set"],
                client, model, num_threads=eval_num_threads,
                stage="baseline",
            )
            print(f"    {tn}: {baseline_scores[tn]:.2f}")
        _last_usage = tracker.get_usage()
    else:
        print("  Skipping baseline evaluation")

    # --- SEQUENTIAL OPTIMIZATION (GI reset every task) ---
    meta_prompt = get_meta_prompt(cfg)
    meta_evolve_enabled, meta_evolve_model_override, meta_template_style = _get_meta_evolution_cfg(cfg)
    meta_evolve_model = meta_evolve_model_override or model
    if meta_evolve_enabled:
        _select_template_single(meta_template_style)
    print(f"  Meta-prompt evolution: enabled={meta_evolve_enabled}, "
          f"template_style={meta_template_style}, model={meta_evolve_model}")
    meta_prompt_history.append({"stage": "initial", "meta_prompt": meta_prompt})

    for idx, task_name in enumerate(ordering):
        from methods.prompt_based.runners.openevolve_runner import _failure_tracker
        if _failure_tracker:
            _failure_tracker.set_stage(f"pos{idx+1}_{task_name}")

        print(f"\n  {'─'*50}")
        print(f"  Position {idx+1}/{len(ordering)}: optimize on {task_name} "
              f"(GI reset, meta_prompt carries)")
        print(f"  {'─'*50}")

        system_message = _build_system_message(cfg, meta_prompt)
        if system_message:
            print(f"  Evolution LM system message: {len(system_message)} chars "
                  f"(meta_prompt: {len(meta_prompt or '')} chars)")

        # GI ALWAYS reset to task default — this is the core experimental
        # control: meta_prompt is the only cross-task information carrier.
        general = TASK_REGISTRY[task_name].get("default_instruction", "")
        initial_prompt_text = build_delimited_prompt(general, task_name)

        prompt_file = ordering_dir / f"initial_prompt_pos{idx+1}_{task_name}.txt"
        with open(prompt_file, "w") as f:
            f.write(initial_prompt_text)
        print(f"  Initial prompt ({len(initial_prompt_text)} chars, "
              f"general section: {len(general)} chars)")

        task_cfg = next(tc for tc in cfg["tasks"] if tc["name"] == task_name)
        task_config_path = write_task_config(
            task_cfg, cfg["model"],
            ordering_dir / f"config_pos{idx+1}_{task_name}.yaml",
            eval_num_threads=eval_num_threads,
        )
        os.environ["BENCHMARK_CONFIG"] = task_config_path

        oe_dict = cfg["openevolve"].copy()
        if system_message:
            oe_dict.setdefault("prompt", {})
            oe_dict["prompt"]["system_message"] = system_message
        apply_thinking_to_oe_dict(oe_dict, cfg)
        oe_config = openevolve.Config.from_dict(oe_dict)

        print(f"  Running OpenEvolve on {task_name}...")
        stage_output_dir = str(ordering_dir / f"evolution_pos{idx+1}_{task_name}")
        result = openevolve.run_evolution(
            initial_program=str(prompt_file),
            evaluator=tasks_info[task_name]["evaluator"],
            config=oe_config,
            output_dir=stage_output_dir,
        )
        print(f"  Evolution complete. Best train score: {result.best_score:.4f}")

        evolved_text = result.best_code
        evolved_general = extract_general_instructions(evolved_text)

        _pre_eval_usage = tracker.get_usage()

        # Single-task post-opt eval on the SAME split baseline was measured
        # on (eval_set), so post-opt and baseline are directly comparable.
        print(f"  Evaluating evolved prompt on {task_name} eval set...")
        post_opt_val = score_on_task(
            evolved_general, task_name, task_data[task_name]["eval_set"],
            client, model, num_threads=eval_num_threads,
            stage=f"after_pos{idx+1}_{task_name}",
        )
        print(f"  {task_name} post-opt eval score: {post_opt_val:.2f} "
              f"(cold baseline: {baseline_scores.get(task_name, float('nan')):.2f})")

        position_scores.append({
            "task": task_name,
            "position": idx + 1,
            "train_best_score": float(result.best_score),
            "post_opt_val_score": float(post_opt_val),
            "cold_baseline": float(baseline_scores.get(task_name, float("nan"))),
            "evolved_general": evolved_general,
        })
        trajectory.append({"task": task_name, "score": float(post_opt_val)})

        if metrics_log_path:
            _post_eval_usage = tracker.get_usage()
            log_stage(
                metrics_log_path, f"pos{idx+1}_{task_name}", "openevolve_v2_meta_single",
                {task_name: post_opt_val}, _post_eval_usage,
                optimization_usage=usage_diff(_pre_eval_usage, _last_usage),
                eval_usage=usage_diff(_post_eval_usage, _pre_eval_usage),
            )
            _last_usage = _post_eval_usage

        # --- META-PROMPT EVOLUTION (trajectory only — no [GI] leak) ---
        is_last = (idx == len(ordering) - 1)
        if meta_evolve_enabled and not is_last:
            print(f"\n  [META] Evolving meta_prompt after {task_name} (position {idx+1})...")
            meta_prompt = evolve_meta_prompt_single(
                current_meta_prompt=meta_prompt,
                task_name=task_name,
                trajectory=trajectory,
                client=client,
                model=meta_evolve_model,
                template_style=meta_template_style,
            )
            meta_prompt_history.append({
                "stage": f"after_pos{idx+1}_{task_name}",
                "meta_prompt": meta_prompt,
            })

    return {
        "ordering": list(ordering),
        "baseline_scores": baseline_scores,
        "position_scores": position_scores,
        "trajectory": trajectory,
        "meta_prompt_history": meta_prompt_history,
    }


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def run_sequential(cfg, strategy=None):
    """Sequential run over a single ordering (taken from cfg['tasks'])."""
    start_time = time.time()
    task_names = [t["name"] for t in cfg["tasks"]]

    (client, model, tracker, failure_tracker, wlog,
     output_dir, metrics_log_path, eval_num_threads) = setup_run(cfg)
    task_data = load_all_datasets_raw(cfg, get_openevolve_tasks(task_names), drop_val=True)
    task_cfgs = {tc["name"]: tc for tc in cfg["tasks"]}

    with tracker.track_to_file():
        result = _run_sequential_ordering_meta_single(
            ordering=task_names,
            task_data=task_data,
            task_cfgs=task_cfgs,
            client=client,
            model=model,
            cfg=cfg,
            output_dir=output_dir,
            metrics_log_path=metrics_log_path,
            tracker=tracker,
            eval_num_threads=eval_num_threads,
        )

    elapsed = time.time() - start_time
    token_usage = tracker.get_usage()
    failure_summary = failure_tracker.summary() if failure_tracker else {"total": 0}

    print(f"\n  Position scores:")
    for entry in result["position_scores"]:
        print(f"    pos{entry['position']} {entry['task']:<16} "
              f"train={entry['train_best_score']:.2f}  "
              f"val={entry['post_opt_val_score']:.2f}  "
              f"cold={entry['cold_baseline']:.2f}")

    print(f"\nToken usage: {json.dumps(token_usage, indent=2)}")
    if failure_summary["total"] > 0:
        print(f"LM failures: {json.dumps(failure_summary, indent=2)}")
    print(f"\nTotal runtime: {elapsed / 60:.1f} minutes ({elapsed:.0f}s)")

    results = {
        "method": "openevolve-v2-meta-single",
        "runtime_seconds": round(elapsed, 1),
        "runtime_hours": round(elapsed / 3600, 2),
        "ordering": result["ordering"],
        "baseline_scores": result["baseline_scores"],
        "position_scores": result["position_scores"],
        "trajectory": result["trajectory"],
        "meta_prompt_history": result["meta_prompt_history"],
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

    plot_path = output_dir / "meta_single_performance.png"
    plot_meta_single(results, str(plot_path))

    if wlog:
        wlog.log_usage(token_usage)
        wlog.log_failures(failure_summary)
        wlog.finish()


def run_allorders(cfg, strategy=None, ordering_indices=None):
    """Run all permutations (or a subset) of the task list."""
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

    (client, model, tracker, failure_tracker, wlog,
     output_dir, metrics_log_path, eval_num_threads) = setup_run(cfg)
    task_data = load_all_datasets_raw(cfg, get_openevolve_tasks(task_names), drop_val=True)
    task_cfgs = {tc["name"]: tc for tc in cfg["tasks"]}

    orderings_to_run = [(i, all_orderings[i - 1]) for i in ordering_indices]
    print(f"\nRunning {len(orderings_to_run)}/{n_orderings} ordering(s): "
          f"{[i for i, _ in orderings_to_run]}")

    all_order_results = {}

    with tracker.track_to_file():
        for i, ordering in orderings_to_run:
            order_label = f"order_{i}"
            print(f"\n{'='*60}")
            print(f"ORDER {i}/{n_orderings}: {' → '.join(ordering)}")
            print(f"{'='*60}")

            result = _run_sequential_ordering_meta_single(
                ordering=ordering,
                task_data=task_data,
                task_cfgs=task_cfgs,
                client=client,
                model=model,
                cfg=cfg,
                output_dir=output_dir,
                metrics_log_path=metrics_log_path,
                tracker=tracker,
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
    print("COMBINED RESULTS — position × task")
    print(f"{'='*60}")
    _print_position_task_table(all_order_results, task_names)

    print(f"\nToken usage: {json.dumps(token_usage, indent=2)}")
    print(f"\nTotal runtime: {elapsed / 60:.1f} minutes ({elapsed:.0f}s)")

    results = {
        "method": "openevolve-v2-meta-single",
        "runtime_seconds": round(elapsed, 1),
        "runtime_hours": round(elapsed / 3600, 2),
        "task_names": task_names,
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

    plot_path = output_dir / "meta_single_allorders.png"
    plot_meta_single_allorders(results, str(plot_path))

    if wlog:
        wlog.log_usage(token_usage)
        wlog.log_failures(failure_summary)
        wlog.finish()


def _print_position_task_table(all_order_results, task_names):
    """Print mean post-opt val score by (task, position), aggregated across orderings."""
    by_cell = {tn: {} for tn in task_names}  # by_cell[task][position] = [scores]
    cold = {}
    for order_key, result in all_order_results.items():
        for entry in result["position_scores"]:
            by_cell[entry["task"]].setdefault(entry["position"], []).append(
                entry["post_opt_val_score"]
            )
        for tn, s in result.get("baseline_scores", {}).items():
            cold[tn] = s

    positions = sorted({p for cells in by_cell.values() for p in cells})
    header = f"{'task':<16} {'cold':>8}" + "".join(f" {'pos'+str(p):>10}" for p in positions)
    print(header)
    print("-" * len(header))
    for tn in task_names:
        row = f"{tn:<16} {cold.get(tn, float('nan')):>8.2f}"
        for p in positions:
            scores = by_cell[tn].get(p, [])
            if scores:
                mean = sum(scores) / len(scores)
                row += f" {mean:>6.2f}({len(scores)})"
            else:
                row += f" {'—':>10}"
        print(row)
