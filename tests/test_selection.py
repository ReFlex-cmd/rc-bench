"""SELECT-001: выбор архитектуры под ограничения устройства.

§9 описания проекта обещает отчёт, по которому можно найти Pareto-оптимальную
архитектуру под ограничения устройства. Тесты фиксируют четыре свойства:
отбрасываются только те кандидаты, что реально нарушают ограничение; причина
отказа называется; при пустом допустимом множестве отчёт говорит об этом, а не
подсовывает ближайшую модель; ограничение, которое нечем проверить, объявляется
неприменимым, а не превращается в отказ.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rc_bench.reporting.selection import (
    DeviceConstraints,
    render_markdown,
    select,
)

# (model, family, nrmse_std, deployable_p50_ns, working_state_bytes,
#  serialized_model_bytes, net_energy_per_inference_mj)
_CELLS = [
    ("persistence", "baseline", 0.7681, 300.0, 192, 512, 0.0004),
    ("ridge_ar", "baseline", 0.6760, 4_100.0, 384, 2_048, 0.0021),
    ("esn", "reservoir", 0.6512, 45_000.0, 8_192, 262_144, 0.0310),
    ("lsm", "reservoir", 0.7104, 120_000.0, 16_384, 524_288, 0.0890),
]


def _write_bundle(root: Path, *, energy: bool, horizons=(1, 24)) -> Path:
    rows = []
    profiles = []
    for model, family, nrmse, p50, state, model_bytes, energy_mj in _CELLS:
        for horizon in horizons:
            config_hash = f"{model}_h{horizon}"
            # Горизонт 24 труднее: качество хуже, стоимость шага та же.
            rows.append(
                {
                    "family": family,
                    "model": model,
                    "horizon": horizon,
                    "mode": "fair",
                    "nrmse_std": nrmse + (0.1 if horizon == 24 else 0.0),
                    "config_hash": config_hash,
                }
            )
            profiles.append(
                {
                    "family": family,
                    "model": model,
                    "horizon": horizon,
                    "config_hash": config_hash,
                    "status": "completed",
                    "deployable_p50_ns": p50,
                    "working_state_bytes": state,
                    "serialized_model_bytes": model_bytes,
                    "energy_status": "measured" if energy else "unavailable",
                    "net_energy_per_inference_mj": energy_mj if energy else None,
                }
            )

    (root / "aggregates").mkdir(parents=True, exist_ok=True)
    (root / "aggregates" / "matrix_table.json").write_text(json.dumps(rows))
    (root / "profiles").mkdir(parents=True, exist_ok=True)
    (root / "profiles" / "summary.json").write_text(json.dumps(profiles))
    return root


@pytest.fixture
def fake_bundle(tmp_path: Path) -> Path:
    return _write_bundle(tmp_path / "bundle", energy=True)


@pytest.fixture
def fake_bundle_no_energy(tmp_path: Path) -> Path:
    return _write_bundle(tmp_path / "bundle_no_energy", energy=False)


class TestSelection:
    def test_unconstrained_selection_returns_the_best_model_overall(self, fake_bundle):
        report = select(fake_bundle, horizon=1, constraints=DeviceConstraints())

        assert report.winner.model == "esn"
        assert all(candidate.feasible for candidate in report.candidates)

    def test_selects_the_best_quality_model_within_the_latency_budget(self, fake_bundle):
        """Бюджет 5 мкс отсекает оба резервуара; лучший из оставшихся —
        ridge_ar, а не глобально лучший esn."""
        report = select(
            fake_bundle, horizon=1, constraints=DeviceConstraints(max_p50_us=5.0)
        )

        assert report.winner.model == "ridge_ar"
        assert report.winner.nrmse_std == pytest.approx(0.6760, abs=1e-4)

        esn = next(c for c in report.candidates if c.model == "esn")
        assert esn.feasible is False
        assert any("p50" in violation for violation in esn.violations)

    def test_only_the_violated_constraint_is_named(self, fake_bundle):
        """Причина отказа — не «не подошла», а конкретное нарушенное число."""
        report = select(
            fake_bundle,
            horizon=1,
            constraints=DeviceConstraints(max_p50_us=5.0, max_state_bytes=10_000),
        )

        esn = next(c for c in report.candidates if c.model == "esn")
        lsm = next(c for c in report.candidates if c.model == "lsm")

        # esn: 8192 B укладывается в 10000, нарушена только задержка.
        assert len(esn.violations) == 1
        assert "p50" in esn.violations[0]
        # lsm: 120 мкс и 16384 B — нарушены оба.
        assert len(lsm.violations) == 2

    def test_a_candidate_exactly_at_the_limit_is_feasible(self, fake_bundle):
        """Ограничение — это «не более», а не «строго меньше»: модель ровно на
        границе бюджета устройство выдерживает."""
        report = select(
            fake_bundle, horizon=1, constraints=DeviceConstraints(max_p50_us=45.0)
        )

        esn = next(c for c in report.candidates if c.model == "esn")
        assert esn.feasible is True
        assert report.winner.model == "esn"

    def test_the_requested_horizon_is_the_only_one_considered(self, fake_bundle):
        report = select(fake_bundle, horizon=24, constraints=DeviceConstraints())

        assert {c.horizon for c in report.candidates} == {24}
        assert report.winner.nrmse_std == pytest.approx(0.7512, abs=1e-4)

    def test_impossible_constraints_report_no_winner(self, fake_bundle):
        report = select(
            fake_bundle, horizon=1, constraints=DeviceConstraints(max_state_bytes=1)
        )

        assert report.winner is None
        assert "нет допустимых" in render_markdown(report).lower()

    def test_energy_constraint_is_ignored_when_energy_is_unavailable(
        self, fake_bundle_no_energy
    ):
        """Молча отбросить все модели по ограничению, которое нечем проверить,
        — это неверный ответ, выданный с уверенным видом."""
        report = select(
            fake_bundle_no_energy,
            horizon=1,
            constraints=DeviceConstraints(max_energy_mj=0.01),
        )

        assert report.energy_status == "unavailable"
        assert report.winner is not None
        assert "энергия недоступна" in render_markdown(report).lower()

    def test_the_energy_constraint_does_bite_when_energy_was_measured(self, fake_bundle):
        report = select(
            fake_bundle, horizon=1, constraints=DeviceConstraints(max_energy_mj=0.01)
        )

        assert report.energy_status == "measured"
        assert report.winner.model == "ridge_ar"
        esn = next(c for c in report.candidates if c.model == "esn")
        assert any("energy" in violation for violation in esn.violations)

    def test_a_profile_measured_on_another_config_is_refused(self, fake_bundle):
        """Тот же инвариант traceability, что и в Pareto: профиль от другой
        конфигурации нельзя приписать опубликованному результату."""
        path = fake_bundle / "profiles" / "summary.json"
        profiles = json.loads(path.read_text())
        profiles[0]["config_hash"] = "deadbeef"
        path.write_text(json.dumps(profiles))

        with pytest.raises(ValueError, match="does not match the published result"):
            select(fake_bundle, horizon=1, constraints=DeviceConstraints())


class TestMarkdown:
    def test_markdown_names_the_frontier_and_the_evidence(self, fake_bundle):
        md = render_markdown(select(fake_bundle, horizon=1, constraints=DeviceConstraints()))

        assert "aggregates/matrix_table.json" in md
        assert "profiles/summary.json" in md
        assert "| esn |" in md

    def test_markdown_shows_the_frontier_so_the_winner_is_not_the_only_point(
        self, fake_bundle
    ):
        """Победитель — ответ на один заданный вопрос; фронт показывает, какой
        размен был бы у другого бюджета."""
        report = select(fake_bundle, horizon=1, constraints=DeviceConstraints())
        md = render_markdown(report)

        assert report.frontiers["latency"]
        assert "persistence" in md  # самый дешёвый — тоже на фронте

    def test_markdown_states_every_declared_constraint_including_the_slack_ones(
        self, fake_bundle
    ):
        md = render_markdown(
            select(
                fake_bundle,
                horizon=1,
                constraints=DeviceConstraints(max_p50_us=5.0, max_model_bytes=1_000_000),
            )
        )

        assert "5.0" in md
        assert "1000000" in md or "1 000 000" in md

    def test_rejected_candidates_keep_their_reason_in_the_table(self, fake_bundle):
        md = render_markdown(
            select(fake_bundle, horizon=1, constraints=DeviceConstraints(max_p50_us=5.0))
        )

        # Отвергнутые не выпадают из отчёта: читатель должен видеть, что́ было
        # рассмотрено и почему отброшено.
        assert "| lsm |" in md
        assert "p50" in md
