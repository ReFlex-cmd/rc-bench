"""Matplotlib plotting helpers for RC-Bench reports."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np


def plot_metric_bar(
    records: list,  # List[RunRecord]
    metric: str = "nrmse_range",
    output_path: Optional[Path] = None,
) -> None:
    """Bar chart comparing reservoir×dataset combinations by *metric*."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels: List[str] = []
    values: List[float] = []
    for r in records:
        msr = r.result.multi_seed_result
        m = msr.mean if msr is not None else r.result.metrics
        if m is None:
            continue
        v = getattr(m, metric, None)
        if v is None:
            continue
        labels.append(f"{r.spec.model_type}\n{r.spec.dataset.name}")
        values.append(float(v))

    if not values:
        return

    x = np.arange(len(values))
    fig, ax = plt.subplots(figsize=(max(6, len(values) * 0.9), 4))
    ax.bar(x, values, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(metric)
    ax.set_title(f"RC-Bench — {metric}")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    out = output_path or Path(f"comparison_{metric}.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Optional[Path] = None,
    title: str = "Predictions vs True",
    max_steps: int = 500,
) -> None:
    """Line plot of y_true vs y_pred (up to *max_steps* steps)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(len(y_true), max_steps)
    t = np.arange(n)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, y_true[:n], label="true", linewidth=1.0, color="black")
    ax.plot(t, y_pred[:n], label="pred", linewidth=1.0, color="steelblue", linestyle="--")
    ax.set_xlabel("step")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()

    out = output_path or Path("predictions.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
