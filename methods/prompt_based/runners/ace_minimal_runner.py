"""ACE (Agentic Context Engineering) runner.

Implements the Generator -> Reflector -> Curator loop from Zhang et al. 2025.

A single Playbook (structured list of bullets) carries across task boundaries.
Per task, we iterate over batches of training examples:

  1. Generator solves the batch using the current playbook + task template.
  2. Reflector analyzes the traces and produces free-form insights.
  3. Curator emits delta ops (add / edit / remove) that mutate the playbook.

Curator output is applied deterministically — no LLM rewrite of the whole
playbook — which preserves unchanged bullets byte-identically across rounds
(avoids "context collapse").

Train traces drive reflection; the held-out eval set is only touched once for
the final cross-task matrix.
"""

import itertools
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

from methods.prompt_based.runners.openevolve_runner import (
    ALL_ORDERINGS,
    _call_llm,
    _extract_answer,
    _extract_label,
    _extract_mcq_answer,
    _safe_format,
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
#  Playbook data structure
# ---------------------------------------------------------------------------

@dataclass
class Bullet:
    id: int
    text: str
    helpful: int = 0
    harmful: int = 0


@dataclass
class Playbook:
    bullets: list = field(default_factory=list)
    _next_id: int = 1

    def apply(self, deltas):
        """Apply a list of delta ops. Malformed ops are skipped with a warning."""
        applied = {"add": 0, "edit": 0, "remove": 0, "skipped": 0}
        for op in deltas:
            try:
                kind = op.get("op")
                if kind == "add":
                    text = (op.get("bullet") or "").strip()
                    if not text:
                        applied["skipped"] += 1
                        continue
                    self.bullets.append(Bullet(id=self._next_id, text=text))
                    self._next_id += 1
                    applied["add"] += 1
                elif kind == "edit":
                    bid = int(op["id"])
                    text = (op.get("bullet") or "").strip()
                    if not text:
                        applied["skipped"] += 1
                        continue
                    for b in self.bullets:
                        if b.id == bid:
                            b.text = text
                            applied["edit"] += 1
                            break
                    else:
                        applied["skipped"] += 1
                elif kind == "remove":
                    bid = int(op["id"])
                    before = len(self.bullets)
                    self.bullets = [b for b in self.bullets if b.id != bid]
                    if len(self.bullets) < before:
                        applied["remove"] += 1
                    else:
                        applied["skipped"] += 1
                else:
                    applied["skipped"] += 1
            except (KeyError, ValueError, TypeError):
                applied["skipped"] += 1
        return applied

    def render(self):
        """Render the playbook as plain text for prompt injection."""
        if not self.bullets:
            return ""
        lines = ["Strategies learned from prior tasks (playbook):"]
        for b in self.bullets:
            lines.append(f"- [id={b.id}] {b.text}")
        return "\n".join(lines)

    def to_dict(self):
        return {
            "next_id": self._next_id,
            "bullets": [asdict(b) for b in self.bullets],
        }

    @classmethod
    def from_dict(cls, d):
        pb = cls()
        pb._next_id = d.get("next_id", 1)
        pb.bullets = [Bullet(**b) for b in d.get("bullets", [])]
        return pb


# ---------------------------------------------------------------------------
#  Prompt builder (plain, NOT delimited — faithful to original ACE)
# ---------------------------------------------------------------------------

def build_ace_prompt(task_name, playbook):
    """Build the full prompt template for a task given the current playbook.

    Structure: task-default instruction + playbook (if any) + task template.
    Returned string still contains task-level placeholders like {question}.
    """
    default = TASK_REGISTRY[task_name].get("default_instruction", "")
    template = TASK_REGISTRY[task_name].get("template", "")
    playbook_text = playbook.render()
    parts = [default]
    if playbook_text:
        parts.append(playbook_text)
    parts.append(template)
    return "\n\n".join(p for p in parts if p)


# ---------------------------------------------------------------------------
#  Generator — runs one example and returns a trace
# ---------------------------------------------------------------------------

def _format_kwargs(task_name, ex):
    """Per-task placeholder values for the prompt template."""
    if task_name == "hotpotqa":
        return {"context": ex["context"], "question": ex["question"]}
    if task_name == "ifeval":
        return {"prompt": ex["prompt"]}
    if task_name == "hover":
        return {"claim": ex["claim"]}
    if task_name in ("sciknoweval", "sciknoweval_bio", "gsm8k", "livebench_math", "toolalpaca", "tooluse", "finqa"):
        return {"question": ex["question"]}
    return {}


def _score_response(task_name, ex, response):
    """Score a generated response against an example. Returns (score, extracted)."""
    if task_name == "hotpotqa":
        from cl.evals.hotpot_evaluate_v1 import f1_score
        answer = _extract_answer(response)
        f1, _, _ = f1_score(answer, ex["answer"])
        return f1, answer
    if task_name == "ifeval":
        from cl.evals.ifeval_lib.evaluation_lib import (
            InputExample,
            test_instruction_following_strict,
        )
        inp = InputExample(
            key=ex["key"],
            instruction_id_list=ex["instruction_id_list"],
            prompt=ex["prompt"],
            kwargs=ex["kwargs"],
        )
        output = test_instruction_following_strict(inp, {ex["prompt"]: response})
        n_total = len(output.follow_instruction_list)
        n_followed = sum(output.follow_instruction_list)
        score = n_followed / n_total if n_total > 0 else 0.0
        return score, response.strip()[:200]
    if task_name == "hover":
        pred = _extract_label(response)
        return (1.0 if pred == ex["label"] else 0.0), pred
    if task_name == "sciknoweval":
        pred = _extract_answer(response)
        gold = ex["answer"].strip()
        if ex["task_type"].startswith("mcq"):
            pred_mcq = _extract_mcq_answer(pred)
            return (1.0 if pred_mcq == gold.upper() else 0.0), pred_mcq
        return (1.0 if gold in pred else 0.0), pred
    if task_name == "sciknoweval_bio":
        from cl.evals.sciknoweval_bio import _extract_mcq_answer as _extract_mcq_bio
        pred = _extract_mcq_bio(_extract_answer(response))
        gold = ex["answer"].strip().upper()
        return (1.0 if pred == gold else 0.0), pred
    if task_name == "tooluse":
        from cl.evals.toolalpaca import _parse_actions, _score_actions
        pred_actions = _parse_actions(response)
        score, _ = _score_actions(pred_actions, ex["golden_steps"])
        return score, str(pred_actions)[:200]
    if task_name == "finqa":
        from cl.evals.finqa import _extract_predicted_number, _numbers_match
        pred = _extract_predicted_number(response)
        return (1.0 if _numbers_match(pred, ex["answer"]) else 0.0), pred
    return 0.0, response.strip()[:200]


def _input_summary(task_name, ex, max_chars=400):
    """Short human-readable summary of the example input for the reflector."""
    if task_name == "hotpotqa":
        ctx = ex["context"]
        if len(ctx) > max_chars:
            ctx = ctx[:max_chars] + "..."
        return f"Context: {ctx}\nQuestion: {ex['question']}"
    if task_name == "ifeval":
        p = ex["prompt"]
        ids = ", ".join(ex.get("instruction_id_list", []))
        return f"Prompt: {p[:max_chars]}\nInstruction IDs: {ids}"
    if task_name == "hover":
        return f"Claim: {ex['claim']}"
    if task_name in ("sciknoweval", "sciknoweval_bio", "gsm8k", "livebench_math", "toolalpaca", "tooluse", "finqa"):
        q = ex["question"]
        return f"Question: {q[:max_chars]}"
    return str(ex)[:max_chars]


def _expected_display(task_name, ex):
    """Human-readable expected answer for the reflector."""
    if task_name == "hotpotqa":
        return ex["answer"]
    if task_name == "ifeval":
        return f"Must follow: {ex.get('instruction_id_list', [])}"
    if task_name == "hover":
        return ex["label"]
    if task_name in ("sciknoweval", "sciknoweval_bio", "gsm8k", "livebench_math", "finqa"):
        return ex["answer"]
    if task_name == "tooluse":
        return str(ex.get("golden_steps", ""))[:200]
    if task_name == "toolalpaca":
        return str(ex.get("golden_steps", ""))[:200]
    return ""


def generate_trace(ex, task_name, prompt_template, client, model, response_max_chars=800):
    """Run generator on one example; return a trace dict for the reflector."""
    kwargs = _format_kwargs(task_name, ex)
    formatted = _safe_format(prompt_template, task_name, **kwargs)
    if formatted is None:
        return {
            "task": task_name,
            "input": _input_summary(task_name, ex),
            "expected": _expected_display(task_name, ex),
            "response": "[format_error]",
            "extracted": "",
            "score": 0.0,
        }
    response = _call_llm(client, model, formatted)
    score, extracted = _score_response(task_name, ex, response)
    truncated = response if len(response) <= response_max_chars else response[:response_max_chars] + "..."
    return {
        "task": task_name,
        "input": _input_summary(task_name, ex),
        "expected": _expected_display(task_name, ex),
        "response": truncated,
        "extracted": str(extracted)[:200],
        "score": round(score, 3),
    }


def generate_batch(batch, task_name, prompt_template, client, model, num_threads=8):
    """Run generator over a batch of examples in parallel. Returns list of traces."""
    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = [
            pool.submit(generate_trace, ex, task_name, prompt_template, client, model)
            for ex in batch
        ]
        return [f.result() for f in as_completed(futures)]


# ---------------------------------------------------------------------------
#  Reflector + Curator prompts
# ---------------------------------------------------------------------------

REFLECT_TEMPLATE = """You are analyzing a language model's attempts at examples from the task "{task_name}".

Goal: identify patterns in what worked and what failed. These insights will feed a curator that maintains a cross-task "playbook" of strategies.

Current playbook (may be empty for the first task):
{current_playbook}

Traces from {n_examples} training examples (score is 0-1):
{traces}

Write 3-8 concise, actionable insights (one per line, prefixed with "-").
Prioritize:
1. Concrete failure patterns (what did the model get wrong and why?)
2. Strategies or formats that correlated with correct answers.
3. Whether each insight is likely task-specific or broadly useful across tasks.

Insights:"""


CURATE_TEMPLATE = """You maintain a playbook of strategies used by a language model across multiple tasks.

Current playbook:
{current_playbook}

New insights from task "{task_name}":
{insights}

Output a JSON array of delta operations to update the playbook. Use ONLY this format:
[
  {{"op": "add", "bullet": "<new strategy, one sentence or short paragraph>"}},
  {{"op": "edit", "id": <int>, "bullet": "<revised text>"}},
  {{"op": "remove", "id": <int>}}
]

Guidelines:
- Prefer small surgical edits. Do not rewrite bullets that are still correct.
- Add only for novel, broadly-useful lessons. Do not duplicate existing bullets.
- Edit when an insight refines or corrects an existing bullet.
- Remove bullets that have been proven wrong or superseded.
- Keep the total number of bullets at or below {max_bullets}. If the playbook
  is at the cap, remove a lower-value bullet before adding.
- If no change is warranted, return [].

Respond with the JSON array only, no prose or code fences."""


def _format_traces(traces, max_chars_per_trace=600):
    """Render traces as a numbered block for the reflector."""
    lines = []
    for i, t in enumerate(traces, start=1):
        lines.append(f"--- Example {i} (score={t['score']}) ---")
        lines.append(f"Input: {t['input']}")
        lines.append(f"Expected: {t['expected']}")
        resp = t["response"]
        if len(resp) > max_chars_per_trace:
            resp = resp[:max_chars_per_trace] + "..."
        lines.append(f"Model output: {resp}")
        lines.append(f"Extracted: {t['extracted']}")
        lines.append("")
    return "\n".join(lines)


def reflect(task_name, traces, playbook, client, model):
    """Call the Reflector LLM; return free-form insights string."""
    current = playbook.render() or "(empty — this is the first update)"
    prompt = REFLECT_TEMPLATE.format(
        task_name=task_name,
        current_playbook=current,
        n_examples=len(traces),
        traces=_format_traces(traces),
    )
    insights = _call_llm(client, model, prompt)
    return insights.strip()


def _parse_deltas(text):
    """Tolerant JSON extractor for Curator output. Returns list of ops."""
    if not text:
        return []
    # Strip code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    # Find first '[' and last ']' to be robust to preamble/postamble
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    blob = text[start : end + 1]
    try:
        data = json.loads(blob)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def curate(task_name, insights, playbook, client, model, max_bullets=100):
    """Call the Curator LLM; return parsed list of delta ops."""
    current = playbook.render() or "(empty)"
    prompt = CURATE_TEMPLATE.format(
        current_playbook=current,
        task_name=task_name,
        insights=insights,
        max_bullets=max_bullets,
    )
    raw = _call_llm(client, model, prompt)
    return _parse_deltas(raw)


# ---------------------------------------------------------------------------
#  Per-ordering sequential ACE loop
# ---------------------------------------------------------------------------

def _chunks(seq, size):
    """Split a list into contiguous chunks of at most `size` items."""
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _run_ace_ordering(ordering, task_data, client, model, cfg, output_dir,
                     metrics_log_path, tracker, eval_num_threads=8,
                     order_label=None):
    """Run ACE sequentially over one task ordering. Returns results dict."""
    task_names = list(task_data.keys())
    # Config lands under the method name — "ace-minimal" via --method ace-minimal,
    # or "ace" if someone points the wrapper config at this runner.
    ace_cfg = cfg.get("ace-minimal") or cfg.get("ace") or {}
    batch_size = ace_cfg.get("batch_size", 20)
    n_epochs = ace_cfg.get("n_epochs_per_task", 1)
    n_train_per_task = ace_cfg.get("n_train_per_task", None)  # None = use all
    max_bullets = ace_cfg.get("max_playbook_bullets", 100)
    reflector_model = ace_cfg.get("reflector_model") or model
    curator_model = ace_cfg.get("curator_model") or model

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
    playbook_history = []

    playbook = Playbook()
    _last_usage = tracker.get_usage()

    # --- BASELINE (default instructions only, no playbook) ---
    if not skip_baseline:
        print(f"\n  {'─'*50}")
        print(f"  Baseline (default instructions, empty playbook)")
        print(f"  {'─'*50}")
        baseline_scores = {}
        for tn in task_names:
            default = TASK_REGISTRY[tn].get("default_instruction", "")
            score = score_on_task(default, tn, task_data[tn]["val_set"],
                                  client, model, num_threads=eval_num_threads,
                                  stage="baseline")
            baseline_scores[tn] = score
            print(f"    {tn}: {score:.2f}")
        for name in task_names:
            all_scores[name].append(baseline_scores[name])
        all_instructions["baseline"] = "(task-specific defaults, empty playbook)"
        _last_usage = tracker.get_usage()

    # --- SEQUENTIAL TRAINING ---
    for idx, task_name in enumerate(ordering):
        print(f"\n  {'─'*50}")
        print(f"  ACE training on {task_name}")
        print(f"  {'─'*50}")

        train_set = task_data[task_name]["train_set"]
        if n_train_per_task is not None:
            train_set = train_set[:n_train_per_task]

        print(f"    train size: {len(train_set)}, batch_size: {batch_size}, "
              f"epochs: {n_epochs}, starting bullets: {len(playbook.bullets)}")

        for epoch in range(n_epochs):
            for batch_idx, batch in enumerate(_chunks(train_set, batch_size)):
                prompt_template = build_ace_prompt(task_name, playbook)

                traces = generate_batch(batch, task_name, prompt_template,
                                        client, model, num_threads=eval_num_threads)
                batch_mean = sum(t["score"] for t in traces) / len(traces) if traces else 0.0

                insights = reflect(task_name, traces, playbook,
                                   client, reflector_model)
                deltas = curate(task_name, insights, playbook,
                                client, curator_model, max_bullets=max_bullets)
                applied = playbook.apply(deltas)

                print(f"    epoch {epoch + 1} batch {batch_idx + 1}: "
                      f"mean={batch_mean:.2f}, "
                      f"deltas={applied}, bullets={len(playbook.bullets)}")

        # Log playbook state after this task
        playbook_history.append({
            "stage": f"after_{task_name}",
            "playbook": playbook.to_dict(),
        })
        all_instructions[f"after_{task_name}"] = playbook.render()

        # Save playbook snapshot to disk
        pb_path = ordering_dir / f"playbook_after_{task_name}.json"
        with open(pb_path, "w") as f:
            json.dump(playbook.to_dict(), f, indent=2)

        _pre_eval_usage = tracker.get_usage()

        # --- EVAL on all tasks with current playbook ---
        print(f"\n  Evaluating on all tasks after {task_name}:")
        rendered_playbook = playbook.render()
        # Use the full "general instructions" = default + playbook, but score_on_task
        # will concatenate with the task template via its own build_prompt. That helper
        # uses the openevolve v1 style (plain, not delimited), so it's fine.
        stage_scores = eval_all_tasks(
            rendered_playbook,
            task_data, client, model,
            num_threads=eval_num_threads,
            stage=f"after_{task_name}",
        )
        for name in task_names:
            all_scores[name].append(stage_scores[name])

        if metrics_log_path:
            from cl.utils.token_tracker import usage_diff
            _post_eval_usage = tracker.get_usage()
            log_stage(
                metrics_log_path, f"after_{task_name}", "ace",
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
        "playbook_history": playbook_history,
        "final_playbook": playbook.to_dict(),
    }


# ---------------------------------------------------------------------------
#  Entry points (mirror openevolve_runner signatures)
# ---------------------------------------------------------------------------

def run_sequential(cfg, strategy="replace"):
    """Run ACE over a single task ordering.

    The `strategy` arg exists for dispatch compatibility but is ignored —
    ACE is inherently sequential over one shared playbook.
    """
    start_time = time.time()
    task_names = [t["name"] for t in cfg["tasks"]]

    client, model, tracker, failure_tracker, wlog, output_dir, metrics_log_path, eval_num_threads = setup_run(cfg)
    task_data = load_all_datasets_raw(cfg, get_openevolve_tasks(task_names))

    with tracker.track_to_file():
        result = _run_ace_ordering(
            ordering=task_names,
            task_data=task_data,
            client=client,
            model=model,
            cfg=cfg,
            output_dir=output_dir,
            metrics_log_path=metrics_log_path,
            tracker=tracker,
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
    """Run ACE across all task orderings (fresh playbook per ordering)."""
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
            order_str = " → ".join(ordering)
            print(f"\n{'='*60}")
            print(f"ORDER {i}/{n_orderings}: {order_str}")
            print(f"{'='*60}")

            result = _run_ace_ordering(
                ordering=ordering,
                task_data=task_data,
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

    if wlog:
        wlog.log_usage(token_usage)
        wlog.log_failures(failure_summary)
        wlog.finish()
