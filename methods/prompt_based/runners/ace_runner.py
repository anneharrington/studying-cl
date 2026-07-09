"""Wrapper runner that uses the ACE reference implementation in ./ace/.

A single `ACE` instance persists its playbook (a markdown string with
section headers + bullet points) across tasks, giving us the cross-task
sequential setting for free. For each task we call `ace.run(mode='offline',
...)` with train + val samples (offline mode tracks the best playbook
against val); we skip ACE's own test-time eval and instead score the
playbook on every task via our usual `eval_all_tasks` helper — that keeps
the cross-task matrix comparable with the gepa / openevolve / v2-meta
runners.

See `methods/prompt_based/runners/ace_minimal_runner.py` for the from-scratch reimplementation
used as a fallback / sanity check under `--method ace-minimal`.
"""

import itertools
import json
import sys
import time
from pathlib import Path

# The ACE repo is not pip-packaged — modules like `logger` / `playbook_utils`
# are top-level and assume the repo root is on sys.path. Insert it at the
# FRONT of sys.path so local edits (e.g. the user's openrouter patch in
# ace/utils.py) always win over any pip-installed `ace` that may happen to
# live in site-packages.
# ace_runner.py lives at methods/prompt_based/runners/, the vendored ACE repo
# at methods/prompt_based/ace/ — two levels up, then into `ace`.
_ACE_ROOT = (Path(__file__).resolve().parent.parent / "ace").resolve()
if str(_ACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ACE_ROOT))

import ace as _ace_pkg  # noqa: E402
_resolved_ace_dir = Path(_ace_pkg.__file__).resolve().parent.parent
if _resolved_ace_dir != _ACE_ROOT:
    raise ImportError(
        f"Expected to load ACE from {_ACE_ROOT} but got {_resolved_ace_dir}. "
        "A different `ace` package is shadowing the local repo. Uninstall "
        "it (`pip uninstall ace`) or ensure sys.path is not being reordered."
    )
from ace import ACE  # noqa: E402

from methods.prompt_based.runners.ace_processors import get_processor
from methods.prompt_based.runners.openevolve_runner import (
    eval_all_tasks,
    load_all_datasets_raw,
    print_results_table,
    score_on_task,
    setup_run,
)
from cl.tasks import TASK_REGISTRY, get_openevolve_tasks
from cl.utils.metrics_logger import log_stage, read_metrics_log
from cl.utils.plotting import plot_allorders, plot_sequential


# ---------------------------------------------------------------------------
#  ACE instance + config plumbing
# ---------------------------------------------------------------------------

def _build_ace_instance(cfg):
    """Create an ACE instance wired to the user's model profile."""
    import os
    ace_cfg = cfg.get("ace", {}) or {}
    model_name = cfg["model"]["task_lm"]
    _thinking = cfg["model"].get("thinking")
    enable_thinking = cfg["model"].get("task_thinking", _thinking)
    if enable_thinking is None:
        enable_thinking = True

    # Auto-route SageMaker model strings ("sagemaker:<endpoint>") to the
    # boto3-backed shim and Portkey strings ("portkey:<route>") to the Portkey
    # SDK shim, regardless of what cfg["ace"]["api_provider"] says. ACE's
    # initialize_clients reads SAGEMAKER_ENDPOINT / AWS_DEFAULT_REGION /
    # PORTKEY_ROUTE / ACE_NO_THINK from env, so we set those here before
    # constructing ACE.
    if str(model_name).startswith("sagemaker:"):
        api_provider = "sagemaker"
        endpoint = model_name.split(":", 1)[1]
        os.environ["SAGEMAKER_ENDPOINT"] = endpoint
        os.environ.setdefault(
            "AWS_DEFAULT_REGION",
            cfg["model"].get("aws_region_name", "us-east-1"),
        )
        os.environ["ACE_NO_THINK"] = "1" if not enable_thinking else "0"
    elif str(model_name).startswith("portkey:"):
        api_provider = "portkey"
        route = model_name.split(":", 1)[1]
        os.environ["PORTKEY_ROUTE"] = route
        os.environ["ACE_NO_THINK"] = "1" if not enable_thinking else "0"
    elif str(model_name).startswith("bedrock:"):
        # Bedrock Converse via boto3. AWS_BEARER_TOKEN_BEDROCK in env is
        # picked up by boto3 automatically — no IAM role/profile needed.
        api_provider = "bedrock"
        model_id = model_name.split(":", 1)[1]
        os.environ["BEDROCK_MODEL_ID"] = model_id
        os.environ["BEDROCK_REGION"] = cfg["model"].get(
            "aws_region_name", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        )
        os.environ["ACE_NO_THINK"] = "1" if not enable_thinking else "0"
    else:
        api_provider = ace_cfg.get("api_provider", "openrouter")

    # --- Optional split-provider reflector ---
    # When ace.reflector_api_provider is set, the reflector runs on a different
    # provider than the generator/curator (e.g. task LM via Portkey->Fireworks,
    # reflector direct to OpenAI GPT-5.2). The spec is handed to ACE's
    # initialize_clients through ACE_REFLECTOR_* env vars (mirrors the
    # PORTKEY_ROUTE convention) and ACE is told which api_provider the reflector
    # reports via reflector_api_provider. When unset, everything below is a
    # no-op and the reflector stays on the task provider (original behaviour).
    for _k in ("ACE_REFLECTOR_PROVIDER", "ACE_REFLECTOR_PORTKEY_ROUTE",
               "ACE_REFLECTOR_API_KEY_ENV", "ACE_REFLECTOR_API_BASE",
               "ACE_REFLECTOR_NO_THINK"):
        os.environ.pop(_k, None)

    reflector_api_provider = ace_cfg.get("reflector_api_provider")
    reflector_model = ace_cfg.get("reflector_model") or model_name

    if reflector_api_provider:
        if str(reflector_api_provider) == "portkey":
            route = str(reflector_model)
            if route.startswith("portkey:"):
                route = route.split(":", 1)[1]
            if not route or route == model_name:
                raise ValueError(
                    "ace.reflector_model must be a 'portkey:<route>' string "
                    "distinct from the task LM when "
                    "ace.reflector_api_provider='portkey'."
                )
            os.environ["ACE_REFLECTOR_PROVIDER"] = "portkey"
            os.environ["ACE_REFLECTOR_PORTKEY_ROUTE"] = route
            # Reasoning reflectors (GPT-5.2 etc.) keep thinking enabled.
            os.environ["ACE_REFLECTOR_NO_THINK"] = "0"
            reflector_model = route
        else:
            key_env = ace_cfg.get("reflector_api_key_env")
            if not key_env:
                raise ValueError(
                    "ace.reflector_api_key_env must be set when "
                    "ace.reflector_api_provider is a non-portkey provider."
                )
            os.environ["ACE_REFLECTOR_PROVIDER"] = str(reflector_api_provider)
            os.environ["ACE_REFLECTOR_API_KEY_ENV"] = str(key_env)
            base = ace_cfg.get("reflector_api_base")
            if base:
                os.environ["ACE_REFLECTOR_API_BASE"] = str(base)
            # reflector_model stays the bare model name (e.g. "gpt-5.2") —
            # it's sent verbatim by the dedicated reflector client.

    return ACE(
        api_provider=api_provider,
        generator_model=model_name,
        reflector_model=reflector_model,
        curator_model=ace_cfg.get("curator_model") or model_name,
        max_tokens=ace_cfg.get("max_tokens", 4096),
        enable_thinking=enable_thinking,
        extra_body=cfg["model"].get("extra_body"),
        reflector_api_provider=reflector_api_provider,
        reflector_max_tokens=ace_cfg.get("reflector_max_tokens"),
        # The analyzer is constructed inside ACE.__init__, so the flag MUST
        # be passed at construction time — not just in the per-run config
        # dict consumed by ace.run(). _build_ace_config() also forwards these
        # fields, but that path only controls the post-init behavior; without
        # this kwarg here, ACE() defaults use_bulletpoint_analyzer=False and
        # self.bulletpoint_analyzer stays None regardless of the config.
        use_bulletpoint_analyzer=ace_cfg.get("use_bulletpoint_analyzer", False),
        bulletpoint_analyzer_threshold=ace_cfg.get(
            "bulletpoint_analyzer_threshold", 0.90,
        ),
    )


def _build_ace_config(cfg, task_name, save_dir):
    """Build ACE's flat per-run config dict, pulling defaults from cfg['ace']."""
    ace_cfg = cfg.get("ace", {}) or {}
    return {
        "task_name": task_name,
        "num_epochs": ace_cfg.get("num_epochs", 1),
        "max_num_rounds": ace_cfg.get("max_num_rounds", 1),
        "curator_frequency": ace_cfg.get("curator_frequency", 10),
        "eval_steps": ace_cfg.get("eval_steps", 50),
        "save_steps": ace_cfg.get("save_steps", 50),
        "playbook_token_budget": ace_cfg.get("playbook_token_budget", 80000),
        "json_mode": ace_cfg.get("json_mode", True),
        "no_ground_truth": ace_cfg.get("no_ground_truth", False),
        "save_dir": str(save_dir),
        "test_workers": ace_cfg.get("test_workers", 8),
        "use_bulletpoint_analyzer": ace_cfg.get("use_bulletpoint_analyzer", False),
        "bulletpoint_analyzer_threshold": ace_cfg.get("bulletpoint_analyzer_threshold", 0.90),
        "early_stopping_patience": ace_cfg.get("early_stopping_patience"),
    }


# ---------------------------------------------------------------------------
#  Per-ordering loop
# ---------------------------------------------------------------------------

def _save_ace_checkpoint(ordering_dir, *, completed_phase_idx, ordering,
                         all_scores, all_instructions, playbook_history,
                         current_playbook):
    """Persist enough state to resume an ordering after the Nth phase completes.

    Written to <ordering_dir>/_ace_checkpoint.json after the cross-task eval
    block of each phase. Reading it back via _load_ace_checkpoint and feeding
    it into _run_ace_ordering(resume_from=N+1) skips phases 0..N and continues
    with the playbook ACE finished phase N with.
    """
    payload = {
        "completed_phase_idx": completed_phase_idx,  # 0-indexed; -1 = baseline only
        "ordering": list(ordering),
        "all_scores": all_scores,
        "all_instructions": all_instructions,
        "playbook_history": playbook_history,
        "current_playbook": current_playbook or "",
    }
    ckpt_path = ordering_dir / "_ace_checkpoint.json"
    tmp_path = ordering_dir / "_ace_checkpoint.json.tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    tmp_path.replace(ckpt_path)


def _load_ace_checkpoint(ordering_dir):
    """Return checkpoint dict or None. Validates ordering matches caller's."""
    ckpt_path = ordering_dir / "_ace_checkpoint.json"
    if not ckpt_path.exists():
        return None
    with open(ckpt_path) as f:
        return json.load(f)


def _run_ace_ordering(ordering, task_data, client, model, cfg, output_dir,
                      metrics_log_path, tracker, eval_num_threads=8,
                      order_label=None, resume_from=None):
    """Run one ACE ordering. Set `resume_from=k` (1-indexed task position) to
    skip the first k tasks and pick up from a previously-saved checkpoint."""
    from cl.utils.token_tracker import usage_diff

    task_names = list(task_data.keys())

    if order_label:
        ordering_dir = output_dir / f"{order_label}_{'_'.join(ordering)}"
    else:
        ordering_dir = output_dir
    ordering_dir.mkdir(parents=True, exist_ok=True)

    skip_baseline = cfg.get("skip_baseline", False)
    seed_mode = cfg.get("seed_mode", "default")
    stages = [] if skip_baseline else ["baseline"]
    stages += [f"after_{name}" for name in ordering]
    all_scores = {name: [] for name in task_names}
    all_instructions = {}
    playbook_history = []

    ace = _build_ace_instance(cfg)
    _last_usage = tracker.get_usage()

    # --- RESUME (if requested) ---
    # resume_from is 1-indexed task position to start at: 1 = restart after
    # baseline, 2 = restart at the 2nd task using the playbook from after the 1st,
    # etc. Skip-from-start (resume_from <= 0 or None) means a fresh run.
    start_phase = 0
    if resume_from is not None and resume_from > 0:
        ckpt = _load_ace_checkpoint(ordering_dir)
        if ckpt is None:
            print(f"  WARNING: --resume-from {resume_from} requested but "
                  f"{ordering_dir / '_ace_checkpoint.json'} not found. "
                  f"Starting from scratch.")
        else:
            if list(ckpt.get("ordering") or []) != list(ordering):
                sys.exit(
                    f"error: checkpoint ordering {ckpt.get('ordering')} does not "
                    f"match requested ordering {list(ordering)}; refusing to resume."
                )
            completed = int(ckpt.get("completed_phase_idx", -1))
            target = int(resume_from) - 1   # 1-indexed → 0-indexed phase to start AT
            if target > completed + 1:
                sys.exit(
                    f"error: --resume-from {resume_from} requires checkpoint with "
                    f"completed_phase_idx>={target - 1}, found {completed}. "
                    f"Run earlier phases first."
                )
            all_scores = ckpt["all_scores"]
            all_instructions = ckpt["all_instructions"]
            playbook_history = ckpt["playbook_history"]
            ace.playbook = ckpt.get("current_playbook") or ace.playbook
            ace.best_playbook = ace.playbook
            start_phase = target
            skip_baseline = True  # baseline scores already in all_scores from checkpoint
            print(f"  RESUMING from phase {start_phase + 1} ({ordering[start_phase]}); "
                  f"loaded playbook from checkpoint ({len(ace.playbook)} chars)")

    # --- BASELINE: default instructions, empty playbook ---
    if not skip_baseline:
        print(f"\n  {'─'*50}\n  Baseline (defaults, empty playbook)\n  {'─'*50}")
        baseline_scores = {}
        for tn in task_names:
            default = TASK_REGISTRY[tn].get("default_instruction", "")
            # Score baseline on eval_set (same as the post-training "after_X"
            # rows below) so each row of the cross-task matrix is on identical
            # examples. Falls back to val_set if no eval_set was configured.
            _bl_set = task_data[tn].get("eval_set", task_data[tn]["val_set"])
            score = score_on_task(default, tn, _bl_set,
                                  client, model, num_threads=eval_num_threads,
                                  stage="baseline", seed_mode=seed_mode)
            baseline_scores[tn] = score
            print(f"    {tn}: {score:.2f}")
        for name in task_names:
            all_scores[name].append(baseline_scores[name])
        all_instructions["baseline"] = "(defaults, empty playbook)"
        _last_usage = tracker.get_usage()

    # --- SEQUENTIAL ACE TRAINING ---
    for idx, task_name in enumerate(ordering):
        if idx < start_phase:
            print(f"\n  [resume] Skipping already-completed phase {idx + 1} ({task_name})")
            continue
        print(f"\n  {'─'*50}\n  ACE on {task_name}\n  {'─'*50}")

        processor = get_processor(task_name)
        train = processor.process_task_data(task_data[task_name]["train_set"])
        val = processor.process_task_data(task_data[task_name]["val_set"])

        save_dir = ordering_dir / f"ace_{task_name}"
        save_dir.mkdir(parents=True, exist_ok=True)
        ace_config = _build_ace_config(cfg, task_name, save_dir)

        # Offline mode requires val; we skip ACE's test phase by passing None —
        # the cross-task matrix is computed by our own eval_all_tasks below.
        ace.run(
            mode="offline",
            train_samples=train,
            val_samples=val,
            test_samples=None,
            data_processor=processor,
            config=ace_config,
        )

        # Carry the best playbook (chosen by ACE via val) into the next task.
        playbook = ace.best_playbook
        ace.playbook = playbook

        all_instructions[f"after_{task_name}"] = playbook
        playbook_history.append({"stage": f"after_{task_name}", "playbook": playbook})

        pb_path = ordering_dir / f"playbook_after_{task_name}.txt"
        with open(pb_path, "w") as f:
            f.write(playbook)

        # --- CROSS-TASK EVAL with our harness (comparable to other methods) ---
        # The ACE playbook alone lacks task-specific format contracts (e.g.,
        # "respond with SUPPORTED or NOT_SUPPORTED"), which live in each
        # task's default_instruction. Prepend the correct default per task so
        # the harness template's format slot (Label:, Answer:, etc.) receives
        # the right vocabulary. Without this, post-training hover collapses to
        # ~0% because the model outputs True/False instead of SUPPORTED/NOT_SUPPORTED.
        _pre_eval_usage = tracker.get_usage()
        print(f"\n  Evaluating on all tasks after {task_name}:")
        stage_scores = {}
        for _tn, _td in task_data.items():
            _default = TASK_REGISTRY[_tn].get("default_instruction", "")
            _combined = f"{_default}\n\n{playbook}" if _default else playbook
            try:
                _score = score_on_task(
                    _combined, _tn, _td.get("eval_set", _td["val_set"]),
                    client, model, num_threads=eval_num_threads,
                    stage=f"after_{task_name}", seed_mode=seed_mode,
                )
            except Exception as e:
                import traceback
                print(f"    {_tn}: cross-eval FAILED — {type(e).__name__}: {e}")
                traceback.print_exc()
                _score = 0.0
            stage_scores[_tn] = _score
            print(f"    {_tn}: {_score:.2f}")
        for name in task_names:
            all_scores[name].append(stage_scores[name])

        if metrics_log_path:
            _post_eval_usage = tracker.get_usage()
            log_stage(
                metrics_log_path, f"after_{task_name}", "ace",
                stage_scores, _post_eval_usage,
                optimization_usage=usage_diff(_pre_eval_usage, _last_usage),
                eval_usage=usage_diff(_post_eval_usage, _pre_eval_usage),
            )
            _last_usage = _post_eval_usage

        # Persist a per-phase checkpoint so a crash on phase N+1 doesn't lose
        # phases 0..N. Resume with --resume-from <idx+2>.
        try:
            _save_ace_checkpoint(
                ordering_dir,
                completed_phase_idx=idx,
                ordering=ordering,
                all_scores=all_scores,
                all_instructions=all_instructions,
                playbook_history=playbook_history,
                current_playbook=ace.best_playbook,
            )
        except Exception as ckpt_err:
            print(f"    WARNING: checkpoint save failed for phase {idx + 1}: {ckpt_err}")

    return {
        "ordering": list(ordering),
        "stages": stages,
        "scores": all_scores,
        "instructions": all_instructions,
        "playbook_history": playbook_history,
        "final_playbook": ace.best_playbook,
    }


# ---------------------------------------------------------------------------
#  Entry points
# ---------------------------------------------------------------------------

def run_sequential(cfg, strategy="replace", resume_from=None):
    """Run ACE over a single task ordering (cfg['tasks'] order).

    Tasks with `optimize: false` in the cfg are loaded into task_data (so the
    cross-task eval scores them every stage) but excluded from the ACE
    optimization ordering. Used for "stable knowledge" probes like
    temporalwiki_stable that should be measured but never trained on.

    `resume_from` (1-indexed task position) skips earlier phases and loads
    state from <output_dir>/_ace_checkpoint.json. Caller must pass the
    pre-existing output_dir via cfg["output_dir"].
    """
    start_time = time.time()
    task_names = [t["name"] for t in cfg["tasks"]]

    client, model, tracker, failure_tracker, wlog, output_dir, metrics_log_path, eval_num_threads = setup_run(cfg)
    task_data = load_all_datasets_raw(cfg, get_openevolve_tasks(task_names))

    optimization_order = [t["name"] for t in cfg["tasks"] if t.get("optimize", True)]
    if not optimization_order:
        sys.exit("error: no tasks to optimize on (all marked optimize: false)")
    eval_only = [n for n in task_names if n not in optimization_order]
    if eval_only:
        print(f"Eval-only tasks (skipped in ACE optimization ordering): {eval_only}")

    with tracker.track_to_file():
        result = _run_ace_ordering(
            ordering=optimization_order,
            task_data=task_data,
            client=client, model=model,
            cfg=cfg, output_dir=output_dir,
            metrics_log_path=metrics_log_path,
            tracker=tracker,
            eval_num_threads=eval_num_threads,
            resume_from=resume_from,
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
        "strategy": strategy,
        "runtime_seconds": round(elapsed, 1),
        "runtime_hours": round(elapsed / 3600, 2),
        "stages": result["stages"],
        "scores": result["scores"],
        "instructions": result["instructions"],
        "playbook_history": result["playbook_history"],
        "final_playbook": result["final_playbook"],
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
    """Run ACE across all task orderings. Fresh playbook per ordering."""
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
    task_data = load_all_datasets_raw(cfg, get_openevolve_tasks(task_names))

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

            result = _run_ace_ordering(
                ordering=ordering,
                task_data=task_data,
                client=client, model=model,
                cfg=cfg, output_dir=output_dir,
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

    print(f"\n{'='*60}\nCOMBINED RESULTS\n{'='*60}")
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

    if wlog:
        wlog.log_usage(token_usage)
        wlog.log_failures(failure_summary)
        wlog.finish()
