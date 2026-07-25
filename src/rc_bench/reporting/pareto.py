"""Quality-vs-cost Pareto plots for the edge story (PLOT-001).

The defence claims a trade-off, so the plot has to show the trade-off honestly:
each model as one point (accuracy against an inference cost measured in a
separate profiling pass), the non-dominated set marked, and nothing hidden by a
second y-axis or a rescaled cost.

Cost axes are logarithmic because the families differ by orders of magnitude —
a lag lookup against a reservoir state update — and a linear axis would collapse
every baseline onto the y-axis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

# Categorical slots 1 and 2 of the reference palette, light surface. The first
# three slots are the documented all-pairs-validated subset, which is the gate
# that applies to scatter plots.
FAMILY_COLORS: Dict[str, str] = {"baseline": "#2a78d6", "reservoir": "#eb6834"}
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8983"

# (key on the profile summary, axis label, unit conversion from the raw value)
COST_AXES: Dict[str, Tuple[str, str, float]] = {
    "latency": ("p50_ns", "Задержка одного шага, p50 (мс, log)", 1e-6),
    "memory": ("working_state_bytes", "Рабочее состояние (КиБ, log)", 1 / 1024),
}


@dataclass(frozen=True)
class ParetoPoint:
    family: str
    model: str
    horizon: int
    quality: float          # NRMSE_std — lower is better
    cost: float             # latency or memory — lower is better
    config_hash: str

    @property
    def label(self) -> str:
        return self.model


def load_points(bundle_dir: str | Path, cost: str) -> List[ParetoPoint]:
    """Join the published table with the resource profiles on the config hash.

    A profile measured on a different configuration than the published result is
    a traceability failure, not a plotting detail: it is refused here rather
    than silently plotted against the wrong point.
    """
    if cost not in COST_AXES:
        raise ValueError(f"unknown cost axis {cost!r}; expected one of {sorted(COST_AXES)}")
    cost_key, _, scale = COST_AXES[cost]

    bundle = Path(bundle_dir)
    rows = json.loads((bundle / "aggregates" / "matrix_table.json").read_text())
    profiles = json.loads((bundle / "profiles" / "summary.json").read_text())

    by_cell = {
        (entry.get("family"), entry.get("model"), entry.get("horizon")): entry
        for entry in profiles
    }

    points: List[ParetoPoint] = []
    for row in rows:
        key = (row["family"], row["model"], row["horizon"])
        profile = by_cell.get(key)
        if profile is None:
            raise ValueError(f"no resource profile for {key[0]}/{key[1]} h{key[2]}")
        if profile.get("config_hash") != row["config_hash"]:
            raise ValueError(
                f"{key[0]}/{key[1]} h{key[2]}: profile config {profile.get('config_hash')} "
                f"does not match the published result {row['config_hash']}"
            )
        value = profile.get(cost_key)
        if value is None:
            raise ValueError(f"{key[0]}/{key[1]} h{key[2]}: profile has no {cost_key}")
        points.append(
            ParetoPoint(
                family=row["family"],
                model=row["model"],
                horizon=row["horizon"],
                quality=row["nrmse_std"],
                cost=float(value) * scale,
                config_hash=row["config_hash"],
            )
        )
    return points


def pareto_front(points: Sequence[ParetoPoint]) -> List[ParetoPoint]:
    """Non-dominated points, sorted by cost.

    ``a`` dominates ``b`` when it is no worse on both axes and strictly better
    on at least one. Exact ties therefore keep both points: two models that
    measure identically are both on the frontier, and dropping one would be an
    arbitrary choice presented as a result.
    """
    front = [
        point
        for point in points
        if not any(
            other.quality <= point.quality
            and other.cost <= point.cost
            and (other.quality < point.quality or other.cost < point.cost)
            for other in points
        )
    ]
    return sorted(front, key=lambda p: (p.cost, p.quality))


def _style_axis(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=INK_MUTED, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)


def plot_quality_cost(
    points: Sequence[ParetoPoint],
    cost: str,
    output_path: str | Path,
    *,
    horizons: Sequence[int] = (1, 24),
) -> str:
    """Render one figure with a panel per horizon; return the written path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, axis_label, _ = COST_AXES[cost]
    fig, axes = plt.subplots(
        1, len(horizons), figsize=(11, 4.6), sharey=True, facecolor=SURFACE
    )
    axes = list(axes) if len(horizons) > 1 else [axes]

    for ax, horizon in zip(axes, horizons):
        panel = [p for p in points if p.horizon == horizon]
        _style_axis(ax)

        front = pareto_front(panel)
        if len(front) > 1:
            ax.plot(
                [p.cost for p in front],
                [p.quality for p in front],
                color=INK_MUTED,
                linewidth=1.0,
                linestyle="--",
                zorder=1,
                label="Парето-фронт" if horizon == horizons[0] else None,
            )

        for family in sorted({p.family for p in panel}):
            members = [p for p in panel if p.family == family]
            ax.scatter(
                [p.cost for p in members],
                [p.quality for p in members],
                s=90,
                color=FAMILY_COLORS.get(family, INK_MUTED),
                edgecolor=SURFACE,
                linewidth=2,
                zorder=3,
                label=family if horizon == horizons[0] else None,
            )

        front_ids = {id(p) for p in front}
        for point in panel:
            ax.annotate(
                point.label,
                (point.cost, point.quality),
                textcoords="offset points",
                xytext=(9, -3),
                fontsize=8.5,
                color=INK_PRIMARY if id(point) in front_ids else INK_SECONDARY,
            )

        ax.set_xscale("log")
        ax.set_title(f"Горизонт h = {horizon}", fontsize=11, color=INK_PRIMARY)
        ax.set_xlabel(axis_label, fontsize=9.5, color=INK_SECONDARY)

    axes[0].set_ylabel("NRMSE_std (ниже — лучше)", fontsize=9.5, color=INK_SECONDARY)
    axes[0].annotate(
        "лучше ↙",
        xy=(0.02, 0.04),
        xycoords="axes fraction",
        fontsize=8.5,
        color=INK_MUTED,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="upper right",
        frameon=False,
        fontsize=9,
        ncols=len(labels),
        bbox_to_anchor=(0.99, 0.99),
    )
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return output_path.as_posix()


def generate_pareto_plots(bundle_dir: str | Path, output_dir: str | Path) -> Dict[str, str]:
    """Write both Pareto figures for the bundle; return {cost: path}."""
    output_dir = Path(output_dir)
    written: Dict[str, str] = {}
    for cost in COST_AXES:
        points = load_points(bundle_dir, cost)
        written[cost] = plot_quality_cost(
            points, cost, output_dir / f"pareto_quality_{cost}.png"
        )
    return written
