#!/usr/bin/env python3
"""Analyze one or more sequential continual-learning runs.

Consumes the per-task `metrics/task<i>_<task>.jsonl` streams produced by
`run_sequential.py` (each line: {"step": int, "data": {"val-core/<source>/acc/mean@N": v, ...}})
and produces:

Per-run outputs (always, in each `<run_dir>`):
  - summary.json:   forgetting matrix + BWT + FWT + final-avg + raw per-task records
  - summary.md:     human-readable table
  - curves.png/pdf: per-task eval accuracy across the continual timeline

Cross-run outputs (when --results-dir is provided AND >=1 run_dir given):
  - <results-dir>/forgetting_matrix.csv    one row per run, headline + per-task final acc
  - <results-dir>/per_task_timeline.csv    one row per (run, task) across the timeline
  - <results-dir>/summary.md               markdown table: method x ordering -> metrics
  - <results-dir>/plots/<run_tag>.pdf      copy of each run's pdf (paper-ready layout)
  - <results-dir>/plots/method_comparison.pdf   grid: methods x orderings, all 6 runs on one page

Forgetting / transfer definitions (Chaudhry et al. 2018, Lopez-Paz & Ranzato 2017):

  rows of R: row 0 = zero-shot, row i (i>=1) = after training through task i (1-indexed task).
  cols of R: 0-indexed task position in the ordering.

  BWT       = mean_{k=0..T-2} (R[T, k] - R[k+1, k])      # final - (just-after-training-k)
  FWT       = mean_{k=1..T-1} (R[k, k] - R[0, k])        # (just-before-training-k) - zero-shot
  final_avg = mean_j R[T, j]

Zero-shot baseline R[0, j] comes from val_before_train of task 1.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# source -> task-key mapping. `data_source` values are set in data/format/*.py; they diverge
# from our short task keys, so we centralize the mapping here.
SOURCE_TO_TASK = {
    "sciknoweval": "bio",
    "finqa":       "finqa",
    "tooluse":     "tooluse",
}
TASKS = ["bio", "finqa", "tooluse"]

# e.g. "val-core/sciknoweval/acc/mean@16"
METRIC_RE = re.compile(r"^val-core/(?P<source>[^/]+)/acc/mean@(?P<n>\d+)$")


def parse_task_jsonl(path: Path) -> List[Dict]:
    """Return ordered list of {step, task_acc: {bio: x, finqa: y, tooluse: z}} from one JSONL."""
    out: List[Dict] = []
    if not path.exists():
        return out
    with open(path, "rb") as fh:
        for line in fh:
            if not line.strip():
                continue
            evt = json.loads(line)
            step = evt["step"]
            data = evt["data"]
            task_acc = {}
            for k, v in data.items():
                m = METRIC_RE.match(k)
                if not m:
                    continue
                task_key = SOURCE_TO_TASK.get(m.group("source"))
                if task_key is None:
                    continue
                task_acc[task_key] = float(v)
            if task_acc:
                out.append({"step": int(step), "task_acc": task_acc})
    # Sort and dedupe by step (keep the latest log per step)
    out.sort(key=lambda r: r["step"])
    return out


def load_run(run_dir: Path) -> Dict:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    metrics_dir = run_dir / "metrics"
    task_streams: List[Dict] = []
    for i, task in enumerate(manifest["ordering_tasks"], start=1):
        jsonl = metrics_dir / f"task{i}_{task}.jsonl"
        records = parse_task_jsonl(jsonl)
        task_streams.append({"task_idx": i, "task": task, "records": records})
    return {"manifest": manifest, "task_streams": task_streams}


def build_forgetting_matrix(run: Dict) -> Tuple[List[List[float]], List[float]]:
    """Return (R matrix [T+1 rows × T cols], zero_shot_row).

    R[0, :]   -> zero-shot accuracy from step 0 of task 1
    R[i, :]   -> accuracy after finishing task i (last eval in task i's stream)
    Cols follow manifest["ordering_tasks"].
    """
    order = run["manifest"]["ordering_tasks"]
    T = len(order)
    R: List[List[Optional[float]]] = [[None] * T for _ in range(T + 1)]

    streams = run["task_streams"]

    # R[0, :] -> step-0 eval of task-1 stream
    if streams and streams[0]["records"]:
        first_eval = streams[0]["records"][0]["task_acc"]
        for j, task in enumerate(order):
            R[0][j] = first_eval.get(task)

    # R[i, :] -> last eval of task i's stream (final accuracy on each task after training task i)
    for i, s in enumerate(streams, start=1):
        if not s["records"]:
            continue
        last_eval = s["records"][-1]["task_acc"]
        for j, task in enumerate(order):
            R[i][j] = last_eval.get(task)

    return R, (R[0] if R else [])


def compute_headline(R: List[List[Optional[float]]], order: List[str]) -> Dict:
    T = len(order)

    def _get(r, c):
        v = R[r][c]
        return float("nan") if v is None else v

    final_avg = sum(_get(T, j) for j in range(T)) / T if T else float("nan")

    # BWT: for each task trained before the last one, compare final vs just-after-its-own-training.
    bwt_terms = [_get(T, k) - _get(k + 1, k) for k in range(T - 1)]
    bwt = sum(bwt_terms) / len(bwt_terms) if bwt_terms else float("nan")

    # FWT: for each task after the first, compare pre-its-own-training accuracy vs zero-shot.
    fwt_terms = [_get(k, k) - _get(0, k) for k in range(1, T)]
    fwt = sum(fwt_terms) / len(fwt_terms) if fwt_terms else float("nan")

    return {
        "final_avg_acc": final_avg,
        "BWT": bwt,
        "FWT": fwt,
    }


def write_summary_md(run_dir: Path, run: Dict, R, headline) -> None:
    order = run["manifest"]["ordering_tasks"]
    T = len(order)
    lines = []
    mf = run["manifest"]
    lines.append(f"# {mf['run_tag']}")
    lines.append("")
    lines.append(
        f"- method=**{mf['method']}**  ordering=**{mf['ordering']}** "
        f"({' → '.join(order)})  model=`{mf['model']}`  "
        f"thinking={mf['enable_thinking']}  seed={mf['seed']}"
    )
    lines.append(f"- sdpo_commit=`{mf.get('sdpo_commit', '?')}`")
    lines.append("")
    lines.append("## Forgetting matrix R[i,j]   (rows = after task i, cols = eval task)")
    lines.append("")
    header = "| after\\\\eval | " + " | ".join(order) + " |"
    sep    = "|" + "---|" * (T + 1)
    lines.append(header)
    lines.append(sep)
    row_labels = ["zero-shot"] + [f"+{order[i]}" for i in range(T)]
    for i, label in enumerate(row_labels):
        row = [label] + [f"{R[i][j]:.3f}" if R[i][j] is not None else "—" for j in range(T)]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(f"- **Final average accuracy**: {headline['final_avg_acc']:.3f}")
    lines.append(f"- **BWT** (backward transfer): {headline['BWT']:+.3f}   (negative ⇒ forgetting)")
    lines.append(f"- **FWT** (forward transfer): {headline['FWT']:+.3f}   (positive ⇒ transfer)")
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n")


TASK_COLORS = {"bio": "#2b8cbe", "finqa": "#e41a1c", "tooluse": "#4daf4a"}


def _draw_run_onto_axis(ax, run: Dict, *, with_legend: bool = True) -> None:
    """Render a single run's per-task accuracy curves onto an existing matplotlib axis.

    Shared by per-run `curves.png`/`.pdf` and the cross-run method_comparison grid.
    """
    order = run["manifest"]["ordering_tasks"]
    global_step = 0
    boundaries = [0]
    series: Dict[str, List[Tuple[int, float]]] = {t: [] for t in TASKS}
    for s in run["task_streams"]:
        for rec in s["records"]:
            gs = global_step + rec["step"]
            for t in TASKS:
                v = rec["task_acc"].get(t)
                if v is not None:
                    series[t].append((gs, v))
        if s["records"]:
            global_step += max(r["step"] for r in s["records"])
        boundaries.append(global_step)
    for t, pts in series.items():
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", markersize=3, label=f"eval: {t}", color=TASK_COLORS.get(t))
    for i, b in enumerate(boundaries[1:-1], start=1):
        ax.axvline(b, color="grey", linestyle=":", alpha=0.5)
        ax.text(b, 1.01, f"→ {order[i] if i < len(order) else ''}",
                transform=ax.get_xaxis_transform(), ha="center", fontsize=7, color="grey")
    ax.set_xlabel("global step")
    ax.set_ylabel("acc@16 (mean)")
    ax.set_ylim(0, 1)
    if with_legend:
        ax.legend(loc="best", fontsize=8)


def plot_curves(run_dir: Path, run: Dict) -> None:
    """Write per-run `curves.png` (quick-look) and `curves.pdf` (paper-ready)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not installed; skipping curves.{png,pdf}", file=sys.stderr)
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    _draw_run_onto_axis(ax, run, with_legend=True)
    ax.set_title(run["manifest"]["run_tag"])
    fig.tight_layout()
    fig.savefig(run_dir / "curves.png", dpi=140)
    fig.savefig(run_dir / "curves.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Cross-run aggregation (only emitted when --results-dir is provided)
# ---------------------------------------------------------------------------

def _final_task_acc(run: Dict) -> Dict[str, Optional[float]]:
    """Final accuracy for each task after the FULL sequence (last eval of last task stream)."""
    out: Dict[str, Optional[float]] = {t: None for t in TASKS}
    streams = run["task_streams"]
    if streams and streams[-1]["records"]:
        last = streams[-1]["records"][-1]["task_acc"]
        for t in TASKS:
            out[t] = last.get(t)
    return out


def _forgetting_per_task(R, order: List[str]) -> Dict[str, float]:
    """forgetting[task] = max over s in {1..T} of R[s, col] - R[T, col],
    where col is the task's position in the ordering.
    """
    T = len(order)
    out: Dict[str, float] = {}
    for j, task in enumerate(order):
        seen = [R[s][j] for s in range(1, T + 1) if R[s][j] is not None]
        final = R[T][j]
        if final is None or not seen:
            out[task] = float("nan")
        else:
            out[task] = max(seen) - final
    return out


def write_cross_run_csvs(results_dir: Path, collected: List[Dict]) -> None:
    """Emit forgetting_matrix.csv (one row per run) + per_task_timeline.csv (one row per (run, task, eval_step))."""
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- forgetting_matrix.csv: one row per run ---
    fm_path = results_dir / "forgetting_matrix.csv"
    fm_cols = [
        "run_tag", "method", "ordering", "seed", "model", "enable_thinking",
        "final_avg_acc", "BWT", "FWT",
        "acc_final_bio", "acc_final_finqa", "acc_final_tooluse",
        "forget_bio", "forget_finqa", "forget_tooluse",
    ]
    with open(fm_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fm_cols)
        w.writeheader()
        for c in collected:
            mf, R, head = c["manifest"], c["R"], c["headline"]
            final_acc = _final_task_acc(c["run"])
            forget = _forgetting_per_task(R, mf["ordering_tasks"])
            w.writerow({
                "run_tag": mf["run_tag"],
                "method": mf["method"],
                "ordering": mf["ordering"],
                "seed": mf["seed"],
                "model": mf["model"],
                "enable_thinking": mf["enable_thinking"],
                "final_avg_acc": f"{head['final_avg_acc']:.6f}",
                "BWT": f"{head['BWT']:.6f}",
                "FWT": f"{head['FWT']:.6f}",
                **{f"acc_final_{t}": ("" if final_acc[t] is None else f"{final_acc[t]:.6f}") for t in TASKS},
                **{f"forget_{t}": f"{forget[t]:.6f}" for t in TASKS},
            })
    print(f"  wrote {fm_path}")

    # --- per_task_timeline.csv: row per (run, train_task_idx, eval_task, step, acc) ---
    tl_path = results_dir / "per_task_timeline.csv"
    with open(tl_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_tag", "method", "ordering", "seed",
                    "train_task_idx", "train_task", "eval_task", "step", "acc"])
        for c in collected:
            mf = c["manifest"]
            for s in c["run"]["task_streams"]:
                for rec in s["records"]:
                    for eval_task, acc in rec["task_acc"].items():
                        w.writerow([
                            mf["run_tag"], mf["method"], mf["ordering"], mf["seed"],
                            s["task_idx"], s["task"], eval_task, rec["step"],
                            f"{acc:.6f}",
                        ])
    print(f"  wrote {tl_path}")


def write_cross_run_summary_md(results_dir: Path, collected: List[Dict]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# SDPO sequential sweep — cross-run summary")
    lines.append("")
    lines.append(f"- runs analyzed: **{len(collected)}**")
    lines.append("")
    lines.append("## Headline (one row per run)")
    lines.append("")
    lines.append("| method | ordering | seed | final_avg_acc | BWT | FWT | acc_final (bio, finqa, tooluse) |")
    lines.append("|--------|----------|------|---------------|-----|-----|---------------------------------|")
    rows = []
    for c in collected:
        mf, head = c["manifest"], c["headline"]
        final_acc = _final_task_acc(c["run"])
        def fmt(v):
            return "—" if (v is None or (isinstance(v, float) and math.isnan(v))) else f"{v:.3f}"
        rows.append((mf["method"], mf["ordering"], mf["seed"],
                     f"| {mf['method']} | {mf['ordering']} | {mf['seed']} | "
                     f"{fmt(head['final_avg_acc'])} | {head['BWT']:+.3f} | {head['FWT']:+.3f} | "
                     f"{fmt(final_acc['bio'])}, {fmt(final_acc['finqa'])}, {fmt(final_acc['tooluse'])} |"))
    rows.sort()
    for _m, _o, _s, line in rows:
        lines.append(line)
    lines.append("")
    lines.append("## Per-task forgetting (max-so-far minus final)")
    lines.append("")
    lines.append("| method | ordering | forget_bio | forget_finqa | forget_tooluse |")
    lines.append("|--------|----------|------------|--------------|----------------|")
    frows = []
    for c in collected:
        mf = c["manifest"]
        f = _forgetting_per_task(c["R"], mf["ordering_tasks"])
        frows.append((mf["method"], mf["ordering"],
                      f"| {mf['method']} | {mf['ordering']} | {f['bio']:+.3f} | {f['finqa']:+.3f} | {f['tooluse']:+.3f} |"))
    frows.sort()
    for _m, _o, line in frows:
        lines.append(line)
    lines.append("")
    (results_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"  wrote {results_dir / 'summary.md'}")


def plot_method_comparison(results_dir: Path, collected: List[Dict]) -> None:
    """One figure, grid (method x ordering), each subplot shows one run's task-accuracy curves."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not installed; skipping method_comparison.pdf", file=sys.stderr)
        return
    # Build (method, ordering) -> run entry
    by_key = {(c["manifest"]["method"], c["manifest"]["ordering"]): c for c in collected}
    methods = sorted({c["manifest"]["method"] for c in collected})
    orderings = sorted({c["manifest"]["ordering"] for c in collected})
    if not methods or not orderings:
        return
    nrows, ncols = len(methods), len(orderings)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows),
                             sharex=False, sharey=True, squeeze=False)
    for i, m in enumerate(methods):
        for j, o in enumerate(orderings):
            ax = axes[i][j]
            if (m, o) in by_key:
                c = by_key[(m, o)]
                _draw_run_onto_axis(ax, c["run"], with_legend=(i == 0 and j == 0))
                mf, head = c["manifest"], c["headline"]
                ax.set_title(f"{m} / {o}  (final={head['final_avg_acc']:.2f}, "
                             f"BWT={head['BWT']:+.2f}, FWT={head['FWT']:+.2f})", fontsize=9)
            else:
                ax.text(0.5, 0.5, f"{m} / {o}\nmissing", ha="center", va="center",
                        transform=ax.transAxes, color="grey")
                ax.set_axis_off()
    fig.suptitle("Sequential continual-learning sweep: method × ordering", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = results_dir / "plots" / "method_comparison.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


def copy_per_run_pdfs(results_dir: Path, collected: List[Dict]) -> None:
    dst_dir = results_dir / "plots"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for c in collected:
        src = c["run_dir"] / "curves.pdf"
        if src.exists():
            shutil.copy2(src, dst_dir / f"{c['manifest']['run_tag']}.pdf")
    print(f"  copied {len(collected)} per-run PDFs to {dst_dir}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", nargs="+", type=Path,
                   help="one or more <out_root>/<run_tag> directories")
    p.add_argument("--results-dir", type=Path, default=None,
                   help="if set, also emit cross-run aggregates "
                        "(forgetting_matrix.csv, per_task_timeline.csv, summary.md, "
                        "plots/<run_tag>.pdf, plots/method_comparison.pdf) here")
    args = p.parse_args()

    collected: List[Dict] = []
    for run_dir in args.run_dirs:
        if not (run_dir / "manifest.json").exists():
            print(f"[skip] {run_dir}: no manifest.json", file=sys.stderr)
            continue
        run = load_run(run_dir)
        R, zero_shot = build_forgetting_matrix(run)
        headline = compute_headline(R, run["manifest"]["ordering_tasks"])
        summary = {
            "run_tag": run["manifest"]["run_tag"],
            "manifest": run["manifest"],
            "forgetting_matrix": R,
            "zero_shot": zero_shot,
            "headline": headline,
            "per_task_records": [
                {"task_idx": s["task_idx"], "task": s["task"], "records": s["records"]}
                for s in run["task_streams"]
            ],
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        write_summary_md(run_dir, run, R, headline)
        plot_curves(run_dir, run)
        print(f"{run['manifest']['run_tag']}: final_avg={headline['final_avg_acc']:.3f} "
              f"BWT={headline['BWT']:+.3f} FWT={headline['FWT']:+.3f} -> {run_dir}/summary.md")
        collected.append({
            "run_dir": run_dir, "run": run, "manifest": run["manifest"],
            "R": R, "headline": headline,
        })

    if args.results_dir and collected:
        print(f"\n=== cross-run aggregation -> {args.results_dir} ===")
        write_cross_run_csvs(args.results_dir, collected)
        write_cross_run_summary_md(args.results_dir, collected)
        copy_per_run_pdfs(args.results_dir, collected)
        plot_method_comparison(args.results_dir, collected)
    return 0


if __name__ == "__main__":
    sys.exit(main())
