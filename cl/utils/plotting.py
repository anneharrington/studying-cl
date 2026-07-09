"""Shared plotting functions for benchmark results.

Used by both runner scripts (inline plot after run) and standalone
plotting scripts (scripts/plot_results.py, scripts/plot_compare.py).
"""

import re

import matplotlib.pyplot as plt


# Canonical colors and labels for all tasks.
# Add new tasks here — they'll be picked up everywhere automatically.
# Year-suffixed task names (e.g. "sentiment10k_2022", "sealqa_2024") are
# handled by `task_color`/`task_label` below: the year is stripped and
# the base color is reused. Tasks missing from this dict still get a
# deterministic non-gray color via a tab10 fallback.
TASK_COLORS = {
    "hotpotqa": "#1f77b4",
    "ifeval": "#2ca02c",
    "hover": "#d62728",
    "sciknoweval": "#9467bd",
    "sciknoweval_bio": "#9467bd",
    "toolalpaca": "#ff7f0e",
    "tooluse": "#ff7f0e",
    "gsm8k": "#8c564b",
    "livebench_math": "#e377c2",
    "finqa": "#17becf",
    "sentiment10k": "#bcbd22",
    "sealqa": "#1f77b4",
    # finance_yr_<YYYY>: 6 sequential CL phases (2015..2020) plotted on the
    # same axis. Without explicit per-year entries, task_color's fallback
    # would hash the stripped base ("finance_yr") and collapse all 6 years
    # to one color. Pin distinct categorical colors so the per-year curves
    # are visually distinguishable in continual-learning overlays.
    "finance_yr_2015": "#1f77b4",
    "finance_yr_2016": "#ff7f0e",
    "finance_yr_2017": "#2ca02c",
    "finance_yr_2018": "#d62728",
    "finance_yr_2019": "#9467bd",
    "finance_yr_2020": "#8c564b",
    # temporalwiki_drift_s<i>: 3 sequential drift slices + an eval-only
    # stable probe. Same problem as finance_yr — without explicit entries
    # the slices collapse via the base hash. Stable in gray to read as
    # "eval-only / never trained".
    "temporalwiki_drift_s1": "#1f77b4",
    "temporalwiki_drift_s2": "#ff7f0e",
    "temporalwiki_drift_s3": "#2ca02c",
    "temporalwiki_stable":  "#7f7f7f",
}

TASK_LABELS = {
    "hotpotqa": "HotpotQA (F1)",
    "ifeval": "IFEval (Acc)",
    "hover": "HoVer (Acc)",
    "sciknoweval": "SciKnowEval (Acc)",
    "sciknoweval_bio": "SciKnowEval-Bio (Acc)",
    "toolalpaca": "ToolAlpaca (Acc)",
    "tooluse": "Tooluse (Acc)",
    "gsm8k": "GSM8K (Acc)",
    "livebench_math": "LiveBench Math (Acc)",
    "finqa": "FinQA (Acc)",
    "sentiment10k": "Sentiment10K (Acc)",
    "sealqa": "SealQA (Acc)",
    "finance_yr_2015": "Finance 2015 (Acc)",
    "finance_yr_2016": "Finance 2016 (Acc)",
    "finance_yr_2017": "Finance 2017 (Acc)",
    "finance_yr_2018": "Finance 2018 (Acc)",
    "finance_yr_2019": "Finance 2019 (Acc)",
    "finance_yr_2020": "Finance 2020 (Acc)",
    "temporalwiki_drift_s1": "TWiki s1 Nov→Dec (F1≥0.5)",
    "temporalwiki_drift_s2": "TWiki s2 Dec→Jan (F1≥0.5)",
    "temporalwiki_drift_s3": "TWiki s3 Jan→Feb (F1≥0.5)",
    "temporalwiki_stable":  "TWiki stable (F1≥0.5)",
}

_YEAR_SUFFIX_RE = re.compile(r"_(\d{4})$")
_TAB10 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def _strip_year(name):
    """Return (base_name, year_or_None) for `<base>_<4digits>`."""
    if not name:
        return name, None
    m = _YEAR_SUFFIX_RE.search(name)
    if m:
        return name[:m.start()], m.group(1)
    return name, None


def task_color(name):
    """Resolve a non-gray, deterministic color for any task name.

    Order: exact match in TASK_COLORS → base name (year suffix stripped) →
    deterministic tab10 fallback (hash-indexed). Never returns gray, so
    overlay plots with new tasks don't all collapse to "#333".
    """
    if name in TASK_COLORS:
        return TASK_COLORS[name]
    base, _year = _strip_year(name)
    if base in TASK_COLORS:
        return TASK_COLORS[base]
    return _TAB10[abs(hash(base or name)) % len(_TAB10)]


def task_label(name):
    """Human-readable label, year-aware. Returns the raw name for unknowns."""
    if name in TASK_LABELS:
        return TASK_LABELS[name]
    base, year = _strip_year(name)
    if base in TASK_LABELS:
        # e.g. "Sentiment10K (Acc) [2022]"
        lbl = TASK_LABELS[base]
        return f"{lbl} [{year}]" if year else lbl
    return name


def _format_tokens(n):
    """Format token count for display (e.g. 2.1M, 150K)."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


def _extract_tokens_per_stage(results):
    """Extract cumulative token counts per stage from metrics_log.

    Returns a dict with total, optimization, and eval token lists aligned
    with results["stages"], or None if unavailable.
    """
    metrics_log = results.get("metrics_log", [])
    stages = results.get("stages", [])
    if not metrics_log or not stages:
        return None

    stage_data = {}
    for entry in metrics_log:
        stage = entry.get("stage", "")
        stage_data[stage] = entry

    total_tokens = []
    opt_tokens = []
    eval_tokens = []

    for stage in stages:
        if stage == "baseline":
            total_tokens.append(0)
            opt_tokens.append(0)
            eval_tokens.append(0)
        elif stage in stage_data:
            entry = stage_data[stage]
            total_tokens.append(entry.get("total_tokens", 0))
            opt_entry = entry.get("optimization_tokens", {})
            eval_entry = entry.get("eval_tokens", {})
            opt_tokens.append(opt_entry.get("total_tokens", 0))
            eval_tokens.append(eval_entry.get("total_tokens", 0))
        else:
            return None

    return {
        "total": total_tokens,
        "optimization": opt_tokens,
        "eval": eval_tokens,
    }


def plot_sequential(results, output_path, title=None, show_tokens=False):
    """Plot results from sequential/append/mixed runs (single ordering).

    Works for any number of tasks — uses TASK_COLORS/TASK_LABELS for known
    tasks and falls back to matplotlib defaults for unknown ones.
    """
    stages = results["stages"]
    scores = results["scores"]
    task_names = list(scores.keys())

    x = list(range(len(stages)))

    if show_tokens:
        token_data = _extract_tokens_per_stage(results)
    else:
        token_data = None

    fig, ax = plt.subplots(figsize=(10, 6))

    for task_name in task_names:
        ax.plot(
            x,
            scores[task_name],
            marker="o",
            linewidth=2.5,
            markersize=8,
            color=task_color(task_name),
            label=task_label(task_name),
        )

    for i in range(1, len(stages)):
        ax.axvline(x=i - 0.5, color="gray", linestyle="--", alpha=0.5)

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title or "Sequential Optimization Performance", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    ax.set_xticks(x)
    ax.set_xlabel("Stage", fontsize=12)

    if show_tokens and token_data:
        tick_labels = []
        for i, stage in enumerate(stages):
            total = token_data["total"][i]
            if total > 0:
                tick_labels.append(f"{stage}\n({_format_tokens(total)} tokens)")
            else:
                tick_labels.append(stage)
        ax.set_xticklabels(tick_labels, fontsize=9)
    else:
        ax.set_xticklabels(stages, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close()


def plot_allorders(results, output_path, title=None):
    """Plot results from allorders runs (grid of subplots, one per ordering)."""
    orderings = results if isinstance(results, dict) and "orderings" not in results else results.get("orderings", results)

    n = len(orderings)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), sharey=True, squeeze=False)
    axes = axes.flatten()

    for idx, (order_key, result) in enumerate(sorted(orderings.items())):
        ax = axes[idx]
        stages = result["stages"]
        scores = result["scores"]
        ordering = result["ordering"]
        x = list(range(len(stages)))

        for task_name in scores:
            ax.plot(
                x, scores[task_name],
                marker="o", linewidth=2, markersize=6,
                color=task_color(task_name),
                label=task_label(task_name),
            )

        for i in range(1, len(stages)):
            ax.axvline(x=i - 0.5, color="gray", linestyle="--", alpha=0.4)

        order_str = " -> ".join(t.upper()[:3] for t in ordering)
        ax.set_title(f"Order {idx+1}: {order_str}", fontsize=10)
        ax.set_xticks(x)
        x_labels = stages if len(stages) <= 5 else [s[:8] for s in stages]
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        if idx == 0:
            ax.legend(fontsize=8, loc="lower left")
        if idx % cols == 0:
            ax.set_ylabel("Score", fontsize=10)

    # Hide unused subplots
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title or "All Task Orderings", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
#  v2-meta-single: position × task visualizations
# ---------------------------------------------------------------------------

def _aggregate_position_scores(results):
    """Collapse orderings into {task: {position: [post_opt_val_scores]}} + cold baselines.

    Accepts either a single-ordering result (has "position_scores") or an
    allorders result (has "orderings").
    """
    if "orderings" in results:
        iter_results = results["orderings"].values()
    else:
        iter_results = [results]

    by_cell = {}
    cold = {}
    for r in iter_results:
        for entry in r.get("position_scores", []):
            by_cell.setdefault(entry["task"], {}).setdefault(
                entry["position"], []
            ).append(entry["post_opt_val_score"])
        for tn, s in r.get("baseline_scores", {}).items():
            cold[tn] = s
    return by_cell, cold


def plot_meta_single(results, output_path, title=None):
    """Single-ordering plot for v2-meta-single runs.

    Bar chart: for each task in the ordering, two bars — cold baseline vs.
    post-opt val score, annotated with the position at which the task was
    optimized.
    """
    position_scores = results.get("position_scores", [])
    baseline_scores = results.get("baseline_scores", {})
    ordering = results.get("ordering", [p["task"] for p in position_scores])

    if not position_scores:
        print("plot_meta_single: no position_scores found; skipping.")
        return

    fig, ax = plt.subplots(figsize=(max(6, 2.2 * len(ordering)), 5))

    x = list(range(len(ordering)))
    cold_vals = [baseline_scores.get(tn, 0.0) for tn in ordering]
    # Map task -> its position entry (each task appears once in the sequence)
    entry_by_task = {e["task"]: e for e in position_scores}
    opt_vals = [entry_by_task[tn]["post_opt_val_score"] for tn in ordering]

    width = 0.38
    ax.bar([xi - width / 2 for xi in x], cold_vals, width,
           label="Cold baseline (task default)", color="#bbbbbb")
    ax.bar([xi + width / 2 for xi in x], opt_vals, width,
           label="Post-opt val score", color="#1f77b4")

    for xi, tn in zip(x, ordering):
        pos = entry_by_task[tn]["position"]
        delta = entry_by_task[tn]["post_opt_val_score"] - baseline_scores.get(tn, 0.0)
        ax.text(xi, max(cold_vals[xi], opt_vals[xi]) + 1.5,
                f"pos {pos}\nΔ={delta:+.1f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([task_label(tn) for tn in ordering], fontsize=9)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title or f"v2-meta-single: {' → '.join(ordering)}", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close()


def plot_meta_single_allorders(results, output_path, title=None):
    """Allorders plot for v2-meta-single runs.

    Two panels:
      (left)  position vs. mean post-opt val score, one line per task.
              Individual samples shown as faint dots. Cold baseline shown
              as a dashed horizontal line per task.
      (right) delta = post-opt val − cold baseline, same x/y structure.
              This isolates the "meta_prompt absorbed lessons" effect.
    """
    by_cell, cold = _aggregate_position_scores(results)
    if not by_cell:
        print("plot_meta_single_allorders: no position_scores found; skipping.")
        return

    task_names = list(results.get("task_names", sorted(by_cell.keys())))
    positions = sorted({p for cells in by_cell.values() for p in cells})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for tn in task_names:
        cells = by_cell.get(tn, {})
        xs, means = [], []
        for p in positions:
            scores = cells.get(p, [])
            if not scores:
                continue
            xs.append(p)
            means.append(sum(scores) / len(scores))
            # Individual samples as dots
            ax1.scatter([p] * len(scores), scores,
                        color=task_color(tn), alpha=0.35, s=28, zorder=2)
        if xs:
            ax1.plot(xs, means, marker="o", linewidth=2.5, markersize=9,
                     color=task_color(tn),
                     label=task_label(tn), zorder=3)
        if tn in cold:
            ax1.axhline(cold[tn], linestyle="--", alpha=0.4,
                        color=task_color(tn))

    ax1.set_xticks(positions)
    ax1.set_xlabel("Position in ordering", fontsize=11)
    ax1.set_ylabel("Post-opt val score", fontsize=11)
    ax1.set_title("Post-opt score by position\n(dashed = cold baseline)", fontsize=11)
    ax1.legend(fontsize=9, loc="best")
    ax1.grid(axis="y", alpha=0.3)

    for tn in task_names:
        cells = by_cell.get(tn, {})
        base = cold.get(tn)
        if base is None:
            continue
        xs, means = [], []
        for p in positions:
            scores = cells.get(p, [])
            if not scores:
                continue
            deltas = [s - base for s in scores]
            xs.append(p)
            means.append(sum(deltas) / len(deltas))
            ax2.scatter([p] * len(deltas), deltas,
                        color=task_color(tn), alpha=0.35, s=28, zorder=2)
        if xs:
            ax2.plot(xs, means, marker="o", linewidth=2.5, markersize=9,
                     color=task_color(tn),
                     label=task_label(tn), zorder=3)

    ax2.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax2.set_xticks(positions)
    ax2.set_xlabel("Position in ordering", fontsize=11)
    ax2.set_ylabel("Δ vs. cold baseline", fontsize=11)
    ax2.set_title("Lift from meta_prompt by position", fontsize=11)
    ax2.legend(fontsize=9, loc="best")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(title or "v2-meta-single: position × task", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close()


def plot_meta_single_compare(results_by_arm, output_path, title=None):
    """Compare multiple arms (e.g. evolving meta vs. static meta) on the same axes.

    `results_by_arm` is a dict: {arm_label: allorders_results_dict}.
    Produces a grid: one row per task, two columns (post-opt score; lift).
    Lines within a panel are one per arm — same task, different meta regime.
    """
    arm_labels = list(results_by_arm.keys())
    first = next(iter(results_by_arm.values()))
    task_names = list(first.get("task_names", []))
    if not task_names:
        by_cell_first, _ = _aggregate_position_scores(first)
        task_names = sorted(by_cell_first.keys())

    # Shared position set across arms
    positions_set = set()
    per_arm = {}
    for arm, res in results_by_arm.items():
        by_cell, cold = _aggregate_position_scores(res)
        per_arm[arm] = (by_cell, cold)
        for cells in by_cell.values():
            positions_set.update(cells.keys())
    positions = sorted(positions_set)

    arm_colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]

    fig, axes = plt.subplots(len(task_names), 2,
                             figsize=(11, 3.5 * len(task_names)),
                             squeeze=False)

    for row, tn in enumerate(task_names):
        ax_score = axes[row][0]
        ax_delta = axes[row][1]

        for i, arm in enumerate(arm_labels):
            by_cell, cold = per_arm[arm]
            cells = by_cell.get(tn, {})
            color = arm_colors[i % len(arm_colors)]

            xs, means, delta_means = [], [], []
            base = cold.get(tn)
            for p in positions:
                scores = cells.get(p, [])
                if not scores:
                    continue
                xs.append(p)
                means.append(sum(scores) / len(scores))
                if base is not None:
                    delta_means.append(means[-1] - base)
                ax_score.scatter([p] * len(scores), scores,
                                 color=color, alpha=0.3, s=22, zorder=2)
                if base is not None:
                    ax_delta.scatter([p] * len(scores),
                                     [s - base for s in scores],
                                     color=color, alpha=0.3, s=22, zorder=2)
            if xs:
                ax_score.plot(xs, means, marker="o", linewidth=2.2, markersize=8,
                              color=color, label=arm, zorder=3)
                if base is not None and delta_means:
                    ax_delta.plot(xs, delta_means, marker="o", linewidth=2.2,
                                  markersize=8, color=color, label=arm, zorder=3)
            if base is not None:
                ax_score.axhline(base, linestyle="--", alpha=0.35, color=color)

        ax_delta.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        ax_score.set_xticks(positions)
        ax_delta.set_xticks(positions)
        ax_score.set_ylabel(f"{task_label(tn)}\npost-opt val", fontsize=10)
        ax_delta.set_ylabel("Δ vs. cold", fontsize=10)
        ax_score.grid(axis="y", alpha=0.3)
        ax_delta.grid(axis="y", alpha=0.3)
        if row == 0:
            ax_score.set_title("Post-opt score (dashed = cold)", fontsize=11)
            ax_delta.set_title("Lift vs. cold baseline", fontsize=11)
            ax_score.legend(fontsize=9, loc="best")
            ax_delta.legend(fontsize=9, loc="best")
        if row == len(task_names) - 1:
            ax_score.set_xlabel("Position", fontsize=10)
            ax_delta.set_xlabel("Position", fontsize=10)

    fig.suptitle(title or "v2-meta-single: arm comparison", fontsize=13, y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close()
