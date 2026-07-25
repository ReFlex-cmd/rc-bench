"""Generate plots for chapter 4 of the thesis from ``reports/runs/*.json``.

Per ТЗ §5.2 produces:
- nrmse_by_task.png            — bar chart per task: NRMSE_range mean ± std per model
- hpo_convergence_<model>_<task>.png — best val_nrmse_range vs trial
- seeds_distribution_<model>_<task>.png — boxplot of per-seed test NRMSE_range
- prediction_horizon_lorenz63.png — closed-loop horizon in Lyapunov times for Lorenz-63

All figures use matplotlib's default style, no LaTeX dependency.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rc_bench.reporting.run_record import load_run_record

PLOTS_DIR = REPO / "reports" / "plots"
RUNS_DIR  = REPO / "reports" / "runs"

# Effective dt for Lorenz-63: data_provider uses dt=0.01, subsample=10 → dt_eff=0.1.
# Maximal Lyapunov exponent for the classical Lorenz parameters ≈ 0.906.
LORENZ_DT_EFFECTIVE = 0.1
LORENZ_LAMBDA_MAX   = 0.906


def _load_all():
    """Return list of (record, model, task) for every run file present."""
    records = []
    for path in sorted(RUNS_DIR.glob("*.json")):
        try:
            rec = load_run_record(path)
            model = rec.spec.model_type
            task = rec.spec.dataset.name
            records.append((rec, model, task))
        except Exception as exc:
            print(f"WARN: cannot load {path.name}: {exc}")
    return records


def plot_nrmse_by_task(records):
    by_task: dict = {}
    for rec, model, task in records:
        ms = rec.result.multi_seed_result
        if ms is None:
            continue
        by_task.setdefault(task, []).append((model, ms.mean.nrmse_range, ms.std.nrmse_range))

    for task, rows in by_task.items():
        rows.sort(key=lambda r: r[1])
        fig, ax = plt.subplots(figsize=(8, 4.5))
        names  = [r[0] for r in rows]
        means  = [r[1] for r in rows]
        stds   = [r[2] for r in rows]
        bars = ax.bar(names, means, yerr=stds, capsize=4,
                      color="steelblue", edgecolor="black")
        ax.set_ylabel("NRMSE_range (mean ± std, n_seeds)")
        ax.set_title(f"{task} — test NRMSE per model")
        for b, v in zip(bars, means):
            ax.text(b.get_x() + b.get_width()/2, v, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        out = PLOTS_DIR / f"nrmse_by_task_{task}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  saved {out.relative_to(REPO)}")


def plot_hpo_convergence(records):
    for rec, model, task in records:
        conv = rec.result.hpo_convergence
        if not conv:
            continue
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(range(1, len(conv)+1), conv, marker="o", markersize=3, linewidth=1)
        ax.set_xlabel("Trial number (completed)")
        ax.set_ylabel("Best val_nrmse_range so far")
        ax.set_title(f"HPO convergence — {model} / {task}")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = PLOTS_DIR / f"hpo_convergence_{model}_{task}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  saved {out.relative_to(REPO)}")


def plot_seeds_distribution(records):
    for rec, model, task in records:
        ms = rec.result.multi_seed_result
        if ms is None or len(ms.metrics_per_seed) < 2:
            continue
        vals = [m.nrmse_range for m in ms.metrics_per_seed]
        fig, ax = plt.subplots(figsize=(4, 3.5))
        ax.boxplot(vals, vert=True, widths=0.5,
                   boxprops=dict(color="steelblue", linewidth=1.5),
                   medianprops=dict(color="firebrick"))
        ax.scatter([1]*len(vals), vals, color="black", alpha=0.5, s=20, zorder=3)
        ax.set_ylabel("test NRMSE_range")
        ax.set_title(f"{model} / {task} (n={len(vals)} seeds)")
        ax.set_xticks([])
        plt.tight_layout()
        out = PLOTS_DIR / f"seeds_distribution_{model}_{task}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  saved {out.relative_to(REPO)}")


def plot_lorenz_horizon(records):
    rows = []
    for rec, model, task in records:
        if task != "lorenz63":
            continue
        ms = rec.result.multi_seed_result
        if ms is None:
            continue
        horizons = [m.prediction_horizon for m in ms.metrics_per_seed]
        lyap_times = [h * LORENZ_DT_EFFECTIVE * LORENZ_LAMBDA_MAX for h in horizons]
        rows.append((model, np.mean(lyap_times), np.std(lyap_times)))

    if not rows:
        return
    rows.sort(key=lambda r: r[1], reverse=True)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    names = [r[0] for r in rows]
    means = [r[1] for r in rows]
    stds  = [r[2] for r in rows]
    bars = ax.bar(names, means, yerr=stds, capsize=4,
                  color="darkorange", edgecolor="black")
    ax.axhline(8.0, color="gray", linestyle="--", linewidth=1,
               label="Pathak 2018 ≈ 8 λ_max·t")
    ax.set_ylabel("Prediction horizon (λ_max · t)")
    ax.set_title("Lorenz-63 closed-loop forecasting horizon")
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:.1f}",
                ha="center", va="bottom", fontsize=9)
    ax.legend()
    plt.tight_layout()
    out = PLOTS_DIR / "prediction_horizon_lorenz63.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out.relative_to(REPO)}")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    records = _load_all()
    print(f"Loaded {len(records)} run records.")
    if not records:
        print("Nothing to plot.")
        return

    print("\n[1/4] NRMSE bars per task...")
    plot_nrmse_by_task(records)
    print("\n[2/4] HPO convergence curves...")
    plot_hpo_convergence(records)
    print("\n[3/4] Per-seed boxplots...")
    plot_seeds_distribution(records)
    print("\n[4/4] Lorenz-63 prediction horizon...")
    plot_lorenz_horizon(records)
    print(f"\nAll plots → {PLOTS_DIR.relative_to(REPO)}/")


if __name__ == "__main__":
    main()
