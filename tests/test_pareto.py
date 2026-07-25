"""PLOT-001: Pareto front derivation and quality-vs-cost figures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rc_bench.reporting.pareto import (
    ParetoPoint,
    generate_pareto_plots,
    load_points,
    pareto_front,
    plot_quality_cost,
)


def _point(model: str, quality: float, cost: float, *, horizon: int = 1) -> ParetoPoint:
    return ParetoPoint(
        family="reservoir",
        model=model,
        horizon=horizon,
        quality=quality,
        cost=cost,
        config_hash=f"hash_{model}",
    )


class TestParetoFront:
    def test_dominated_points_are_excluded(self):
        best = _point("best", quality=0.5, cost=1.0)
        worse_both = _point("worse", quality=0.7, cost=2.0)

        assert pareto_front([best, worse_both]) == [best]

    def test_a_cheaper_but_less_accurate_point_stays_on_the_front(self):
        accurate = _point("accurate", quality=0.5, cost=10.0)
        cheap = _point("cheap", quality=0.9, cost=0.1)

        front = pareto_front([accurate, cheap])
        assert [p.model for p in front] == ["cheap", "accurate"]

    def test_equal_points_both_survive(self):
        """A tie is not a ranking — dropping one would invent a result."""
        first = _point("first", quality=0.5, cost=1.0)
        second = _point("second", quality=0.5, cost=1.0)

        assert len(pareto_front([first, second])) == 2

    def test_equal_quality_keeps_only_the_cheaper_point(self):
        cheap = _point("cheap", quality=0.5, cost=1.0)
        costly = _point("costly", quality=0.5, cost=4.0)

        assert [p.model for p in pareto_front([cheap, costly])] == ["cheap"]

    def test_front_is_sorted_by_cost(self):
        points = [
            _point("c", quality=0.4, cost=9.0),
            _point("a", quality=0.9, cost=0.5),
            _point("b", quality=0.6, cost=3.0),
        ]
        assert [p.model for p in pareto_front(points)] == ["a", "b", "c"]


def _bundle(tmp_path: Path, *, profile_hash: str | None = None) -> Path:
    bundle = tmp_path / "jmlc_2026"
    rows = [
        {
            "family": "baseline",
            "model": "persistence",
            "horizon": h,
            "nrmse_std": 0.77 + 0.1 * (h == 24),
            "config_hash": f"base_h{h}",
        }
        for h in (1, 24)
    ] + [
        {
            "family": "reservoir",
            "model": "esn",
            "horizon": h,
            "nrmse_std": 0.66 + 0.2 * (h == 24),
            "config_hash": f"esn_h{h}",
        }
        for h in (1, 24)
    ]
    (bundle / "aggregates").mkdir(parents=True)
    (bundle / "aggregates" / "matrix_table.json").write_text(json.dumps(rows))

    profiles = [
        {
            "family": row["family"],
            "model": row["model"],
            "horizon": row["horizon"],
            "config_hash": profile_hash or row["config_hash"],
            "status": "completed",
            "p50_ns": 500.0 if row["family"] == "baseline" else 45_000.0,
            "working_state_bytes": 192 if row["family"] == "baseline" else 8_192,
        }
        for row in rows
    ]
    (bundle / "profiles").mkdir(parents=True)
    (bundle / "profiles" / "summary.json").write_text(json.dumps(profiles))
    return bundle


class TestLoadPoints:
    def test_costs_are_converted_to_display_units(self, tmp_path):
        bundle = _bundle(tmp_path)

        latency = {(p.model, p.horizon): p for p in load_points(bundle, "latency")}
        memory = {(p.model, p.horizon): p for p in load_points(bundle, "memory")}

        assert latency[("esn", 1)].cost == pytest.approx(0.045)   # 45_000 ns -> ms
        assert memory[("esn", 1)].cost == pytest.approx(8.0)      # 8_192 B -> KiB
        assert latency[("esn", 1)].quality == pytest.approx(0.66)

    def test_profile_from_another_config_is_refused(self, tmp_path):
        bundle = _bundle(tmp_path, profile_hash="deadbeef")

        with pytest.raises(ValueError, match="does not match the published result"):
            load_points(bundle, "latency")

    def test_missing_profile_is_refused(self, tmp_path):
        bundle = _bundle(tmp_path)
        profiles = json.loads((bundle / "profiles" / "summary.json").read_text())
        (bundle / "profiles" / "summary.json").write_text(json.dumps(profiles[:-1]))

        with pytest.raises(ValueError, match="no resource profile"):
            load_points(bundle, "latency")

    def test_unknown_cost_axis_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="unknown cost axis"):
            load_points(_bundle(tmp_path), "energy")


class TestFigures:
    def test_figure_is_written(self, tmp_path):
        points = load_points(_bundle(tmp_path), "latency")
        out = plot_quality_cost(points, "latency", tmp_path / "plots" / "pareto.png")

        assert Path(out).is_file()
        assert Path(out).stat().st_size > 5_000  # a real rendering, not an empty canvas

    def test_both_figures_are_generated(self, tmp_path):
        bundle = _bundle(tmp_path)
        written = generate_pareto_plots(bundle, tmp_path / "plots")

        assert set(written) == {"latency", "memory"}
        for path in written.values():
            assert Path(path).is_file()
