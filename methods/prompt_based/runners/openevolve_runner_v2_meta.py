"""OpenEvolve runner v2-meta — evolves the meta_prompt itself between tasks.

Same delimited-prompt mechanics as v2, but adds an outer loop: after each
task's optimization+evaluation, an LLM call rewrites the meta_prompt based on
cross-task score deltas. The updated meta_prompt is then injected into the
evolution LM's system message for the next task.

Hypothesis: a meta_prompt that adapts to observed task-type interference will
guide the inner evolution loop toward more transferable general instructions
than a static meta_prompt.

Use via: python scripts/run.py --method openevolve-v2-meta ...

All non-sequential entry points (single, mixed) are inherited from v2.
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
    eval_all_tasks,
    write_task_config,
    strip_placeholders,
    load_all_datasets_raw,
    setup_run,
    print_results_table,
    run_single,
    run_mixed,
    apply_thinking_to_oe_dict,
)
from methods.prompt_based.runners.openevolve_runner_v2 import (
    _build_system_message,
    build_delimited_prompt,
    extract_general_instructions,
)
from methods.prompt_based.runners.meta_prompt import get_meta_prompt
from cl.tasks import TASK_REGISTRY, get_openevolve_tasks
from cl.utils.token_tracker import usage_diff
from cl.utils.metrics_logger import log_stage, read_metrics_log
from cl.utils.plotting import plot_sequential, plot_allorders


# ---------------------------------------------------------------------------
#  Meta-prompt evolution
# ---------------------------------------------------------------------------

# Two templates for the meta-evolve LLM call. Both expose the same data; only
# the "guided" version supplies opinionated guidance about HOW to use it.
# Use config `meta_prompt.evolution.template_style: minimal | guided`.

_META_EVOLVE_HEADER = """You are tuning a meta-strategy used to guide an LLM that evolves prompts for a benchmark. The benchmark runs multiple different tasks in sequence, and the same general-instructions block is carried across tasks. Your meta-strategy is fed to the prompt-evolution LLM as system-message guidance.

A new optimization round just finished. Here is what happened:

Task just optimized: {task_name}
Baseline scores (before any optimization): {baseline_scores}
Scores BEFORE this round (after previous task's optimization): {scores_before}
Scores AFTER this round: {scores_after}

The general-instructions block that was evolved during this round:
\"\"\"
{current_general}
\"\"\"

Your CURRENT meta-strategy:
\"\"\"
{current_meta_prompt}
\"\"\"

The evolved general-instructions block is reused across all the tasks above. The goal is for it to perform well on all of them, not just the one just optimized."""

_META_EVOLVE_CONSTRAINTS = """Constraints:
- Keep it concise (under 250 words). It is injected into a system message every iteration; bloat is costly.
- It is GUIDANCE, not a prompt template — do NOT include placeholders like {{context}} or {{prompt}}.
- Speak in the imperative voice to the evolution LLM ("When optimizing X, prefer Y...").
- It is allowed (and encouraged) to grow into a small skills/strategies catalogue over multiple rounds.

Return ONLY the updated meta-strategy text, no preamble, no markdown fences."""


META_EVOLVE_TEMPLATE_MINIMAL = (
    _META_EVOLVE_HEADER
    + "\n\nRevise the meta-strategy so that the next round's evolution does better.\n\n"
    + _META_EVOLVE_CONSTRAINTS
)


META_EVOLVE_TEMPLATE_GUIDED = (
    _META_EVOLVE_HEADER
    + """

Revise the meta-strategy so that the next round's evolution does better. The revised meta-strategy should help the evolution LLM:
1. Recognize what TYPE of task it is optimizing, and what reasoning patterns suit that type
2. Identify which parts of prior general-instructions tend to TRANSFER across task types vs. which parts are task-specific (e.g. output-format directives that contaminate other tasks)
3. Flag specific failure modes you observe in the score deltas above (regressions on tasks not currently being optimized are signals of cross-task interference)
4. Encourage the evolution LLM to structure the [GENERAL_INSTRUCTIONS] block as conditional, task-type-aware guidance rather than a single flat directive. The block must work well across all task types it has seen, not just the current one. Suggest patterns like "If the task involves X, do Y; if it involves Z, do W."

"""
    + _META_EVOLVE_CONSTRAINTS
)


def _select_template(template_style):
    if template_style == "guided":
        return META_EVOLVE_TEMPLATE_GUIDED
    if template_style == "minimal":
        return META_EVOLVE_TEMPLATE_MINIMAL
    raise ValueError(
        f"Unknown meta_prompt.evolution.template_style: {template_style!r} "
        f"(expected 'minimal' or 'guided')"
    )


def evolve_meta_prompt(current_meta_prompt, task_name, baseline_scores,
                        scores_before, scores_after, current_general,
                        client, model, template_style="minimal", max_chars=4000):
    """Call the LLM to rewrite the meta_prompt based on cross-task score deltas.

    template_style:
        "minimal" — exposes data only; lets the meta-LLM discover what's useful
        "guided"  — adds explicit bullets about task-type awareness, transfer,
                    and conditional structure inside [GENERAL_INSTRUCTIONS]

    Returns the new meta_prompt string. On failure (empty/oversize), returns
    the current_meta_prompt unchanged so the run continues.
    """
    def _fmt_scores(s):
        return ", ".join(f"{k}={v:.1f}" for k, v in sorted(s.items()))

    template = _select_template(template_style)
    prompt = template.format(
        task_name=task_name,
        baseline_scores=_fmt_scores(baseline_scores) if baseline_scores else "(skipped)",
        scores_before=_fmt_scores(scores_before),
        scores_after=_fmt_scores(scores_after),
        current_general=(current_general or "")[:max_chars],
        current_meta_prompt=current_meta_prompt or "(none yet)",
    )

    response = _call_llm(client, model, prompt)
    new_mp = (response or "").strip()
    # Strip accidental code fences
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


def _get_meta_evolution_cfg(cfg):
    """Return (enabled: bool, model: Optional[str], template_style: str).

    Config:
        meta_prompt:
          evolution:
            enabled: true               # default true under v2-meta
            model: "..."                # optional override; falls back to task_lm
            template_style: "minimal"   # "minimal" (default) or "guided"
    """
    mp_cfg = cfg.get("meta_prompt")
    if not isinstance(mp_cfg, dict):
        return True, None, "minimal"
    evo_cfg = mp_cfg.get("evolution", {}) or {}
    return (
        bool(evo_cfg.get("enabled", True)),
        evo_cfg.get("model"),
        evo_cfg.get("template_style", "minimal"),
    )


# ---------------------------------------------------------------------------
#  Sequential runner with meta-prompt evolution
# ---------------------------------------------------------------------------

def _run_sequential_ordering_meta(ordering, task_data, task_cfgs, client, model, cfg,
                                   output_dir, metrics_log_path, tracker, strategy="replace",
                                   eval_num_threads=8, order_label=None):
    """Sequential run with delimited prompts AND meta-prompt evolution between tasks."""
    tasks_info = get_openevolve_tasks(list(task_data.keys()))
    task_names = list(task_data.keys())

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
    meta_prompt_history = []  # list of {stage, meta_prompt}

    _last_usage = tracker.get_usage()

    # --- BASELINE ---
    baseline_scores = None
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

    # --- SEQUENTIAL OPTIMIZATION + META EVOLUTION ---
    current_general = None
    previous_general = None
    meta_prompt = get_meta_prompt(cfg)
    meta_evolve_enabled, meta_evolve_model_override, meta_template_style = _get_meta_evolution_cfg(cfg)
    meta_evolve_model = meta_evolve_model_override or model
    # Validate template style up front so misconfig fails fast (not after first task).
    if meta_evolve_enabled:
        _select_template(meta_template_style)
    print(f"  Meta-prompt evolution: enabled={meta_evolve_enabled}, "
          f"template_style={meta_template_style}, model={meta_evolve_model}")
    meta_prompt_history.append({"stage": "initial", "meta_prompt": meta_prompt})

    # Track the previous stage's scores (initialized from baseline)
    prev_stage_scores = dict(baseline_scores) if baseline_scores else {n: 0.0 for n in task_names}

    for idx, task_name in enumerate(ordering):
        from methods.prompt_based.runners.openevolve_runner import _failure_tracker
        if _failure_tracker:
            _failure_tracker.set_stage(f"after_{task_name}")

        print(f"\n  {'─'*50}")
        print(f"  Optimize on {task_name}")
        print(f"  {'─'*50}")

        # Rebuild system message from the (possibly updated) meta_prompt for THIS round
        system_message = _build_system_message(cfg, meta_prompt)
        if system_message:
            print(f"  Evolution LM system message: {len(system_message)} chars (meta_prompt: {len(meta_prompt or '')} chars)")

        task_default = TASK_REGISTRY[task_name].get("default_instruction", "")

        if strategy == "append":
            if idx == 0:
                general = task_default
            else:
                general = (
                    f"{task_default}\n\n"
                    f"Previous task optimized instructions:\n{strip_placeholders(previous_general)}"
                )
        else:  # replace
            general = task_default if current_general is None else current_general

        initial_prompt_text = build_delimited_prompt(general, task_name)

        prompt_file = ordering_dir / f"initial_prompt_{task_name}.txt"
        with open(prompt_file, "w") as f:
            f.write(initial_prompt_text)
        print(f"  Initial prompt ({len(initial_prompt_text)} chars, general section: {len(general)} chars)")

        task_cfg = next(tc for tc in cfg["tasks"] if tc["name"] == task_name)
        task_config_path = write_task_config(task_cfg, cfg["model"],
                                             ordering_dir / f"config_{task_name}.yaml",
                                             eval_num_threads=eval_num_threads)
        os.environ["BENCHMARK_CONFIG"] = task_config_path

        oe_dict = cfg["openevolve"].copy()
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

        evolved_text = result.best_code
        all_prompts[f"after_{task_name}"] = evolved_text
        current_general = extract_general_instructions(evolved_text)
        previous_general = current_general
        all_instructions[f"after_{task_name}"] = current_general
        print(f"  Extracted general instructions ({len(current_general)} chars)")

        _pre_eval_usage = tracker.get_usage()

        print(f"\n  Evaluating on all tasks after {task_name} optimization:")
        stage_scores = eval_all_tasks(current_general, task_data, client, model,
                                      num_threads=eval_num_threads, stage=f"after_{task_name}")
        for name in task_names:
            all_scores[name].append(stage_scores[name])

        if metrics_log_path:
            _post_eval_usage = tracker.get_usage()
            log_stage(
                metrics_log_path, f"after_{task_name}", "openevolve_v2_meta",
                stage_scores, _post_eval_usage,
                optimization_usage=usage_diff(_pre_eval_usage, _last_usage),
                eval_usage=usage_diff(_post_eval_usage, _pre_eval_usage),
            )
            _last_usage = _post_eval_usage

        # --- META-PROMPT EVOLUTION ---
        # Skip meta-update on the LAST task (next iteration won't use it).
        is_last = (idx == len(ordering) - 1)
        if meta_evolve_enabled and not is_last:
            print(f"\n  [META] Evolving meta_prompt after {task_name}...")
            meta_prompt = evolve_meta_prompt(
                current_meta_prompt=meta_prompt,
                task_name=task_name,
                baseline_scores=baseline_scores,
                scores_before=prev_stage_scores,
                scores_after=stage_scores,
                current_general=current_general,
                client=client,
                model=meta_evolve_model,
                template_style=meta_template_style,
            )
            meta_prompt_history.append({
                "stage": f"after_{task_name}",
                "meta_prompt": meta_prompt,
            })

        prev_stage_scores = dict(stage_scores)

    return {
        "ordering": list(ordering),
        "stages": stages,
        "scores": all_scores,
        "instructions": all_instructions,
        "evolved_prompts": all_prompts,
        "meta_prompt_history": meta_prompt_history,
    }


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def run_sequential(cfg, strategy="replace"):
    start_time = time.time()
    task_names = [t["name"] for t in cfg["tasks"]]

    client, model, tracker, failure_tracker, wlog, output_dir, metrics_log_path, eval_num_threads = setup_run(cfg)
    task_data = load_all_datasets_raw(cfg, get_openevolve_tasks(task_names), drop_val=True)
    task_cfgs = {tc["name"]: tc for tc in cfg["tasks"]}

    with tracker.track_to_file():
        result = _run_sequential_ordering_meta(
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
        "meta_prompt_history": result.get("meta_prompt_history", []),
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

            result = _run_sequential_ordering_meta(
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
