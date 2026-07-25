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
from typing import Any, Dict, List, Sequence, Tuple

# Categorical slots 1 and 2 of the reference palette, light surface. The first
# three slots are the documented all-pairs-validated subset, which is the gate
# that applies to scatter plots.
FAMILY_COLORS: Dict[str, str] = {"baseline": "#2a78d6", "reservoir": "#eb6834"}
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8983"

# (key on the profile summary, axis label, unit conversion from the raw value)
#
# The latency axis uses the DEPLOYABLE measurement, not the as-implemented one:
# sklearn's per-sample predict API costs tens of microseconds of input
# validation, which exceeds the arithmetic of every model in this matrix and
# would rank the models by how many framework calls they happen to make. The
# as-implemented numbers stay in the profile artifacts and are discussed in the
# bundle README.
COST_AXES: Dict[str, Tuple[str, str, float]] = {
    "latency": ("deployable_p50_ns", "Задержка одного шага, p50 (мс, log)", 1e-6),
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


def _set_padded_xlim(ax, panel: Sequence[ParetoPoint]) -> None:
    """Leave room on the right for the last point's label.

    Matplotlib's autoscale bounds the markers, not the text beside them, so the
    rightmost label is clipped by the axes edge without explicit padding.
    """
    import math

    costs = [p.cost for p in panel]
    lo, hi = math.log10(min(costs)), math.log10(max(costs))
    span = (hi - lo) or 1.0
    ax.set_xlim(10 ** (lo - 0.15 * span), 10 ** (hi + 0.55 * span))


def _set_shared_ylim(ax, points: Sequence[ParetoPoint]) -> None:
    """Set one y range covering every panel (the axis is shared)."""
    qualities = [p.quality for p in points]
    y_lo, y_hi = min(qualities), max(qualities)
    y_span = (y_hi - y_lo) or 0.1
    ax.set_ylim(y_lo - 0.14 * y_span, y_hi + 0.08 * y_span)


def _place_labels(ax, panel: Sequence[ParetoPoint], *, front_ids: set) -> None:
    """Label every point, nudging labels apart when their points coincide.

    Two models can land on the same coordinates — at h=24 persistence and
    seasonal persistence are the same predictor — and stacked text is
    unreadable. Colliding labels step downward and get a leader line so each
    one still reads to its own marker.
    """
    import math

    x_lo, x_hi = (math.log10(v) for v in ax.get_xlim())
    y_lo, y_hi = ax.get_ylim()
    x_span = (x_hi - x_lo) or 1.0
    y_span = (y_hi - y_lo) or 1.0

    label_dx = 0.018          # gap between marker and text, axes fraction
    row_height = 0.062        # vertical step between stacked labels
    placed: List[Tuple[float, float, float]] = []  # (x_start, x_end, y)

    ordered = sorted(panel, key=lambda p: (-p.quality, p.cost))
    for point in ordered:
        xn = (math.log10(point.cost) - x_lo) / x_span
        yn = (point.quality - y_lo) / y_span
        width = 0.014 * len(point.label)

        x_text = xn + label_dx
        y_text = yn
        for step in range(6):
            candidate = yn - step * row_height
            collides = any(
                abs(candidate - other_y) < row_height * 0.85
                and x_text < other_end
                and x_text + width > other_start
                for other_start, other_end, other_y in placed
            )
            if not collides:
                y_text = candidate
                break
        placed.append((x_text, x_text + width, y_text))

        offset = abs(y_text - yn) > 1e-9
        ax.annotate(
            point.label,
            xy=(point.cost, point.quality),
            xycoords="data",
            xytext=(x_text, y_text - 0.012),
            textcoords="axes fraction",
            fontsize=8.5,
            va="bottom",
            color=INK_PRIMARY if id(point) in front_ids else INK_SECONDARY,
            arrowprops=(
                dict(arrowstyle="-", color=INK_MUTED, linewidth=0.6, shrinkA=0, shrinkB=3)
                if offset
                else None
            ),
        )


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
    panels: List[Tuple[Any, List[ParetoPoint], set]] = []

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

        ax.set_xscale("log")
        _set_padded_xlim(ax, panel)
        panels.append((ax, panel, {id(p) for p in front}))

        ax.set_title(f"Горизонт h = {horizon}", fontsize=11, color=INK_PRIMARY)
        ax.set_xlabel(axis_label, fontsize=9.5, color=INK_SECONDARY)

    # The y axis is shared, so its range must cover every panel and be final
    # before labels are positioned — a per-panel set_ylim would let the last
    # panel crop the others, and labels anchored to stale limits would drift
    # away from their markers.
    _set_shared_ylim(axes[0], points)
    for ax, panel, front_ids in panels:
        _place_labels(ax, panel, front_ids=front_ids)

    axes[0].set_ylabel("NRMSE_std (ниже — лучше)", fontsize=9.5, color=INK_SECONDARY)
    axes[0].annotate(
        "лучше ↙",
        xy=(0.02, 0.03),
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
