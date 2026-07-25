"""EVID-001: aggregation and traceability checks over an evidence bundle.

The fixtures build RunRecords directly instead of running the pipeline: these
tests are about what the bundle asserts, and a hand-built record is the only
way to produce the violations the validator has to catch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rc_bench.core.schema import (
    BaselineSpec,
    DatasetSpec,
    EvaluationContext,
    ExperimentSpec,
    MetricsResult,
    MetricsSummary,
    MultiSeedResult,
    ProtocolSpec,
    ReadoutSpec,
    ReservoirSpec,
    ResultSpec,
    SelectionResult,
)
from rc_bench.reporting.evidence import (
    build_rows,
    discover_modes,
    load_cells,
    validate_aggregates,
    validate_bundle,
    validate_bundle_modes,
    validate_records,
    write_aggregates,
)
from rc_bench.reporting.run_record import RunRecord, save_run_record

DIGEST = "4" * 64
ALPHA_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]


def _metrics(nrmse_std: float) -> MetricsResult:
    return MetricsResult(
        rmse=nrmse_std,
        nrmse_range=nrmse_std / 5,
        nrmse_std=nrmse_std,
        nrmse_var=nrmse_std**2,
        mae=nrmse_std * 0.8,
        mse=nrmse_std**2,
        prediction_horizon=0,
        val_nrmse_range=nrmse_std / 5,
        val_nrmse_std=nrmse_std,
        mase=nrmse_std * 0.9,
        mae_skill=0.3,
        train_time=1.0,
        inference_latency=0.1,
        peak_memory=1024,
    )


def _evaluation(horizon: int) -> EvaluationContext:
    return EvaluationContext(
        target_start_index=200 + horizon,
        n_test_targets=2199 if horizon == 1 else 2176,
        seasonal_period=24,
        mase_scale=0.7217,
        seasonal_test_mae=0.7462,
        n_mase_scale_terms=7084,
    )


def _protocol(horizon: int, **overrides) -> ProtocolSpec:
    base = dict(
        washout=200,
        train_frac=0.6,
        val_frac=0.2,
        forecasting_mode="fixed_horizon",
        horizon=horizon,
        selection_metric="nrmse_std",
        seasonal_period=24,
    )
    base.update(overrides)
    return ProtocolSpec(**base)


def _baseline_record(
    model: str, horizon: int, nrmse_std: float = 0.7, *, mode: str = "fair"
) -> RunRecord:
    spec = ExperimentSpec(
        dataset=DatasetSpec(name="uci_household_power", length=12_000, raw_sha256=DIGEST),
        baseline=BaselineSpec(type=model),
        protocol=_protocol(horizon, n_seeds=0, use_hpo=False, mode=mode),
        readout=ReadoutSpec(alpha_grid=ALPHA_GRID),
        seed=None,
    )
    result = ResultSpec(
        status="completed",
        config_hash=spec.config_hash(),
        frozen_config_hash=spec.config_hash(),
        resolved_spec=spec,
        metrics=_metrics(nrmse_std),
        model_family="baseline",
        deterministic=True,
        evaluated_seeds=[],
        selection=SelectionResult(method="none"),
        evaluation=_evaluation(horizon),
    )
    return RunRecord.make(spec, result)


def _reservoir_record(
    model: str,
    horizon: int,
    nrmse_std: float = 0.65,
    *,
    n_seeds: int = 5,
    hpo_budget: int = 20,
    mode: str = "fair",
) -> RunRecord:
    frozen = ExperimentSpec(
        dataset=DatasetSpec(name="uci_household_power", length=12_000, raw_sha256=DIGEST),
        reservoir=ReservoirSpec(type=model, params={}),
        protocol=_protocol(
            horizon, n_seeds=n_seeds, use_hpo=True, hpo_budget=hpo_budget, mode=mode
        ),
        readout=ReadoutSpec(alpha_grid=ALPHA_GRID),
        seed=42,
    )
    resolved = frozen.model_copy(deep=True)
    resolved.reservoir.params = {"sr": 0.9}
    seeds = [42 + i for i in range(n_seeds)]
    per_seed = [_metrics(nrmse_std + 0.01 * i) for i in range(n_seeds)]
    multi = MultiSeedResult(
        n_seeds=n_seeds,
        seeds=seeds,
        metrics_per_seed=per_seed,
        mean=MetricsSummary.from_metrics_list(per_seed, lambda xs: sum(xs) / len(xs)),
        std=MetricsSummary.from_metrics_list(per_seed, lambda xs: 0.01),
    )
    result = ResultSpec(
        status="completed",
        config_hash=resolved.config_hash(),
        frozen_config_hash=frozen.config_hash(),
        resolved_spec=resolved,
        multi_seed_result=multi,
        model_family="reservoir",
        deterministic=False,
        evaluated_seeds=seeds,
        selection=SelectionResult(method="optuna", metric="nrmse_std"),
        evaluation=_evaluation(horizon),
    )
    return RunRecord.make(frozen, result)


def _write(records, runs_dir: Path) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        family = record.result.model_family
        model = record.spec.model_type
        horizon = record.spec.protocol.horizon
        save_run_record(record, runs_dir / f"{family}_{model}_h{horizon}.json")
    return runs_dir


def _clean_records():
    return [
        _baseline_record("persistence", 1),
        _baseline_record("persistence", 24),
        _reservoir_record("esn", 1),
        _reservoir_record("esn", 24),
    ]


class TestAggregation:
    def test_rows_carry_the_seed_spread_only_for_multi_seed_cells(self, tmp_path):
        runs = _write(_clean_records(), tmp_path / "runs")
        rows = build_rows(load_cells(runs), runs)

        baseline = next(r for r in rows if r.family == "baseline" and r.horizon == 1)
        reservoir = next(r for r in rows if r.family == "reservoir" and r.horizon == 1)

        assert baseline.nrmse_std_sd is None
        assert baseline.evaluated_seeds == []
        assert baseline.hpo_budget is None
        assert reservoir.nrmse_std_sd == pytest.approx(0.01)
        assert reservoir.evaluated_seeds == [42, 43, 44, 45, 46]
        assert reservoir.hpo_budget == 20

    def test_rows_carry_the_protocol_mode(self, tmp_path):
        runs = _write(_clean_records(), tmp_path / "runs")
        rows = build_rows(load_cells(runs), runs)

        assert {row.mode for row in rows} == {"fair"}
        paths = write_aggregates(rows, tmp_path / "aggregates")
        header = Path(paths["csv"]).read_text().splitlines()[0]
        assert "mode" in header.split(",")

    def test_best_effort_aggregates_go_to_their_own_file(self, tmp_path):
        """Числа двух режимов не сводятся в одну таблицу: fair остаётся
        headline-таблицей бандла, best-effort живёт рядом под своим именем."""
        records = [
            _baseline_record("persistence", 1, mode="best_effort"),
            _reservoir_record("esn", 1, mode="best_effort", hpo_budget=60),
        ]
        runs = _write(records, tmp_path / "runs")
        rows = build_rows(load_cells(runs), runs)
        paths = write_aggregates(rows, tmp_path / "aggregates")

        assert Path(paths["json"]).name == "matrix_table_best_effort.json"
        assert Path(paths["csv"]).name == "matrix_table_best_effort.csv"
        assert not (tmp_path / "aggregates" / "matrix_table.json").exists()

    def test_writing_a_table_of_mixed_modes_is_refused(self, tmp_path):
        records = [
            _reservoir_record("esn", 1),
            _reservoir_record("lsm", 1, mode="best_effort", hpo_budget=40),
        ]
        runs = _write(records, tmp_path / "runs")
        rows = build_rows(load_cells(runs), runs)

        with pytest.raises(ValueError, match="mixes protocol modes"):
            write_aggregates(rows, tmp_path / "aggregates")

    def test_matrix_table_reports_training_time(self, tmp_path):
        """§5 описания проекта ставит время обучения первым пунктом ресурсного
        профиля. Оно измеряется в каждом прогоне и лежит в MetricsResult, но до
        опубликованной таблицы не доходило — то есть заявленный пункт профиля
        нельзя было проверить по бандлу."""
        runs = _write(_clean_records(), tmp_path / "runs")
        rows = build_rows(load_cells(runs), runs)

        reservoir = next(r for r in rows if r.family == "reservoir" and r.horizon == 1)
        baseline = next(r for r in rows if r.family == "baseline" and r.horizon == 1)

        assert reservoir.train_time_s == pytest.approx(1.0)
        assert reservoir.train_time_s_sd == pytest.approx(0.01)
        assert baseline.train_time_s == pytest.approx(1.0)
        # Детерминированный baseline запускается однократно: разброса нет, и
        # ноль на его месте читался бы как измеренное отсутствие разброса.
        assert baseline.train_time_s_sd is None

    def test_reservoir_point_estimate_is_the_across_seed_mean(self, tmp_path):
        runs = _write([_reservoir_record("esn", 1, 0.60)], tmp_path / "runs")
        (row,) = build_rows(load_cells(runs), runs)
        # seeds carry 0.60..0.64
        assert row.nrmse_std == pytest.approx(0.62)

    def test_rows_are_sorted_by_horizon_then_family_then_model(self, tmp_path):
        runs = _write(_clean_records(), tmp_path / "runs")
        rows = build_rows(load_cells(runs), runs)
        assert [(r.horizon, r.family, r.model) for r in rows] == [
            (1, "baseline", "persistence"),
            (1, "reservoir", "esn"),
            (24, "baseline", "persistence"),
            (24, "reservoir", "esn"),
        ]

    def test_written_table_round_trips_and_has_no_absolute_paths(self, tmp_path):
        runs = _write(_clean_records(), tmp_path / "runs")
        rows = build_rows(load_cells(runs), runs)
        paths = write_aggregates(rows, tmp_path / "aggregates")

        payload = json.loads(Path(paths["json"]).read_text())
        assert len(payload) == len(rows)
        assert Path(paths["csv"]).read_text().splitlines()[0].startswith("family,model,horizon")
        assert validate_aggregates(rows, paths["json"]) == []

    def test_written_csv_has_no_carriage_returns(self, tmp_path):
        """Гейт проекта включает `git diff --check`, который считает CR в конце
        строки лишним пробелом. С умолчанием модуля csv (CRLF) каждая
        регенерация бандла роняла бы гейт до индексации файла."""
        runs = _write(_clean_records(), tmp_path / "runs")
        rows = build_rows(load_cells(runs), runs)
        paths = write_aggregates(rows, tmp_path / "aggregates")

        assert b"\r" not in Path(paths["csv"]).read_bytes()

    def test_edited_table_is_detected(self, tmp_path):
        runs = _write(_clean_records(), tmp_path / "runs")
        rows = build_rows(load_cells(runs), runs)
        paths = write_aggregates(rows, tmp_path / "aggregates")

        payload = json.loads(Path(paths["json"]).read_text())
        payload[0]["nrmse_std"] = 0.0001
        Path(paths["json"]).write_text(json.dumps(payload))

        problems = validate_aggregates(rows, paths["json"])
        assert len(problems) == 1
        assert "disagrees with the raw RunRecords" in problems[0]


class TestModeAwareValidation:
    def test_equal_budget_rule_applies_only_to_fair_mode(self):
        """DEC-004 требует равного бюджета в fair. В best_effort равные бюджеты
        означали бы, что режим не сделал того, ради чего существует."""
        fair = [
            _reservoir_record("esn", 1, hpo_budget=20),
            _reservoir_record("lsm", 1, hpo_budget=40),
        ]
        assert any("unequal HPO budgets" in p for p in validate_records(fair))

        best = [
            _reservoir_record("esn", 1, mode="best_effort", hpo_budget=60),
            _reservoir_record("lsm", 1, mode="best_effort", hpo_budget=40),
        ]
        assert not [p for p in validate_records(best) if "unequal HPO budgets" in p]

    def test_mixed_modes_in_one_directory_are_rejected(self):
        records = [
            _reservoir_record("esn", 1),
            _reservoir_record("lsm", 1, mode="best_effort", hpo_budget=40),
        ]
        problems = validate_records(records)
        assert any("mixes protocol modes" in p for p in problems)

    def test_best_effort_budget_below_the_fair_budget_is_reported(self):
        """«Лучшее усилие», оплаченное меньшим бюджетом, чем честное сравнение,
        — это не best-effort, а опечатка в конфиге."""
        records = [_reservoir_record("esn", 1, mode="best_effort", hpo_budget=5)]
        problems = validate_records(records, fair_budget=20)
        assert any("smaller than the fair budget" in p for p in problems)

    def test_best_effort_budget_at_or_above_the_fair_budget_is_clean(self):
        records = [_reservoir_record("esn", 1, mode="best_effort", hpo_budget=60)]
        assert validate_records(records, fair_budget=20) == []

    def test_a_fair_reservoir_that_never_tuned_is_reported(self):
        """DEC-004 — весь смысл fair-режима, и обойти его проще всего не
        неравными бюджетами, а отсутствием настройки: ячейка без HPO не
        попадает в множество бюджетов, и правило равенства видит один бюджет
        вместо двух. ProtocolSpec.use_hpo по умолчанию False, поэтому дыра
        достижима забытой строкой в шаблоне, а не только опечаткой."""
        tuned = _reservoir_record("esn", 1, hpo_budget=20)
        untuned = _reservoir_record("lsm", 1, hpo_budget=20)
        untuned.spec.protocol.use_hpo = False
        untuned.resolved_spec.protocol.use_hpo = False
        untuned.result.config_hash = untuned.resolved_spec.config_hash()
        untuned.result.frozen_config_hash = untuned.spec.config_hash()

        problems = validate_records([tuned, untuned])
        assert any("fair" in p and "did not run HPO" in p for p in problems)

    def test_a_best_effort_reservoir_that_never_tuned_is_reported(self):
        """Порог бюджета живёт под use_hpo, поэтому ячейка с use_hpo=False
        обходила его целиком и публиковалась как «лучшее усилие», не сделав
        ни одного trial. Это ровно то неравное сравнение, ради запрета
        которого режим и заведён."""
        record = _reservoir_record("esn", 1, mode="best_effort", hpo_budget=60)
        record.spec.protocol.use_hpo = False
        record.resolved_spec.protocol.use_hpo = False
        record.result.config_hash = record.resolved_spec.config_hash()
        record.result.frozen_config_hash = record.spec.config_hash()

        problems = validate_records([record], fair_budget=20)
        assert any("best_effort" in p and "did not run HPO" in p for p in problems)


class TestRecordValidation:
    def test_clean_records_have_no_problems(self):
        assert validate_records(_clean_records(), expected_cells=4) == []

    def test_missing_cells_are_reported(self):
        problems = validate_records(_clean_records(), expected_cells=14)
        assert any("expected 14 cells, found 4" in p for p in problems)

    def test_unpinned_raw_digest_is_reported(self):
        record = _baseline_record("persistence", 1)
        record.spec.dataset.raw_sha256 = None
        record.resolved_spec.dataset.raw_sha256 = None
        record.result.config_hash = record.resolved_spec.config_hash()
        record.result.frozen_config_hash = record.spec.config_hash()

        problems = validate_records([record])
        assert any("raw_sha256 is not pinned" in p for p in problems)

    def test_divergent_evaluation_context_is_reported(self):
        records = _clean_records()
        records[2].result.evaluation.n_test_targets = 2100

        problems = validate_records(records)
        assert any("different evaluation context" in p for p in problems)

    def test_unequal_hpo_budget_across_reservoirs_is_reported(self):
        records = [
            _reservoir_record("esn", 1, hpo_budget=20),
            _reservoir_record("lsm", 1, hpo_budget=5),
        ]
        problems = validate_records(records)
        assert any("unequal HPO budgets" in p for p in problems)

    def test_seed_count_mismatch_is_reported(self):
        record = _reservoir_record("esn", 1, n_seeds=5)
        record.result.evaluated_seeds = [42, 43]

        problems = validate_records([record])
        assert any("evaluated 2" in p for p in problems)

    def test_missing_evaluation_context_is_reported(self):
        record = _reservoir_record("esn", 1)
        record.result.evaluation = None

        problems = validate_records([record])
        assert any("no EvaluationContext" in p for p in problems)

    def test_failed_cell_is_reported_not_skipped(self):
        record = _baseline_record("persistence", 1)
        record.result.status = "failed"

        problems = validate_records([record])
        assert any("status is 'failed'" in p for p in problems)

    def test_non_finite_metric_is_reported(self):
        record = _baseline_record("persistence", 1)
        record.result.metrics.mase = float("nan")

        problems = validate_records([record])
        assert any("metric mase is missing or non-finite" in p for p in problems)

    def test_dataset_disagreement_is_reported(self):
        records = _clean_records()
        records[1].spec.dataset.length = 6000
        records[1].resolved_spec.dataset.length = 6000
        records[1].result.config_hash = records[1].resolved_spec.config_hash()
        records[1].result.frozen_config_hash = records[1].spec.config_hash()

        problems = validate_records(records)
        assert any("disagree on the dataset identity" in p for p in problems)


def _fair_bundle(tmp_path: Path, *, profiles: bool = True) -> Path:
    bundle = tmp_path / "jmlc_2026"
    runs = _write(_clean_records(), bundle / "fair" / "runs")
    (bundle / "dataset_manifest.json").write_text(
        json.dumps({"raw_file": {"sha256": DIGEST}})
    )
    (bundle / "hardware_profile.json").write_text(json.dumps({"cpu_model": "test"}))
    rows = build_rows(load_cells(runs), runs)
    write_aggregates(rows, bundle / "aggregates")

    if profiles:
        summary = [
            {
                "family": row.family,
                "model": row.model,
                "horizon": row.horizon,
                "config_hash": row.config_hash,
                "status": "completed",
            }
            for row in rows
        ]
        (bundle / "profiles").mkdir(parents=True, exist_ok=True)
        (bundle / "profiles" / "summary.json").write_text(json.dumps(summary))
    return bundle


class TestBundleValidation:
    def _bundle(self, tmp_path: Path, *, profiles: bool = True) -> Path:
        return _fair_bundle(tmp_path, profiles=profiles)

    def test_complete_bundle_validates(self, tmp_path):
        bundle = self._bundle(tmp_path)
        assert validate_bundle(bundle, expected_cells=4) == []

    def test_missing_manifest_and_hardware_profile_are_reported(self, tmp_path):
        bundle = self._bundle(tmp_path)
        (bundle / "dataset_manifest.json").unlink()
        (bundle / "hardware_profile.json").unlink()

        problems = validate_bundle(bundle, expected_cells=4)
        assert any("missing dataset manifest" in p for p in problems)
        assert any("missing hardware profile" in p for p in problems)

    def test_digest_not_matching_the_bundle_manifest_is_reported(self, tmp_path):
        bundle = self._bundle(tmp_path)
        (bundle / "dataset_manifest.json").write_text(
            json.dumps({"raw_file": {"sha256": "9" * 64}})
        )

        problems = validate_bundle(bundle, expected_cells=4)
        assert any("does not match the bundle manifest" in p for p in problems)

    def test_missing_profile_for_a_published_cell_is_reported(self, tmp_path):
        bundle = self._bundle(tmp_path)
        summary = json.loads((bundle / "profiles" / "summary.json").read_text())
        (bundle / "profiles" / "summary.json").write_text(json.dumps(summary[:-1]))

        problems = validate_bundle(bundle, expected_cells=4)
        assert any("no resource profile" in p for p in problems)

    def test_profile_measured_on_another_config_is_reported(self, tmp_path):
        bundle = self._bundle(tmp_path)
        summary = json.loads((bundle / "profiles" / "summary.json").read_text())
        summary[0]["config_hash"] = "deadbeefdeadbeef"
        (bundle / "profiles" / "summary.json").write_text(json.dumps(summary))

        problems = validate_bundle(bundle, expected_cells=4)
        assert any("but the published result is" in p for p in problems)

    def test_profiles_can_be_waived_explicitly(self, tmp_path):
        bundle = self._bundle(tmp_path, profiles=False)
        assert validate_bundle(bundle, expected_cells=4, require_profiles=False) == []
        assert validate_bundle(bundle, expected_cells=4) != []


class TestModeAwareBundleValidation:
    """Бандл с двумя контурами: fair публикует профили, best_effort — нет."""

    def _bundle(self, tmp_path: Path, *, profiles: bool = True) -> Path:
        return _fair_bundle(tmp_path, profiles=profiles)

    def _add_best_effort(self, bundle: Path, *, hpo_budget: int = 60) -> Path:
        records = [
            _baseline_record("persistence", 1, mode="best_effort"),
            _baseline_record("persistence", 24, mode="best_effort"),
            _reservoir_record("esn", 1, mode="best_effort", hpo_budget=hpo_budget),
            _reservoir_record("esn", 24, mode="best_effort", hpo_budget=hpo_budget),
        ]
        runs = _write(records, bundle / "best_effort" / "runs")
        write_aggregates(build_rows(load_cells(runs), runs), bundle / "aggregates")
        return bundle

    def test_discover_modes_reports_only_the_modes_that_have_runs(self, tmp_path):
        bundle = self._bundle(tmp_path)
        assert discover_modes(bundle) == ["fair"]

        self._add_best_effort(bundle)
        assert discover_modes(bundle) == ["fair", "best_effort"]

    def test_both_modes_validate_although_only_fair_has_profiles(self, tmp_path):
        bundle = self._add_best_effort(self._bundle(tmp_path))
        assert validate_bundle_modes(bundle, expected_cells=4) == []

    def test_best_effort_paid_less_than_fair_is_reported_with_its_mode(self, tmp_path):
        """Бюджет fair-контура берётся из его же прогонов, а не из аргумента, —
        иначе гейт проверял бы best_effort против числа, которого в бандле нет."""
        bundle = self._add_best_effort(self._bundle(tmp_path), hpo_budget=5)

        problems = validate_bundle_modes(bundle, expected_cells=4)
        assert any(
            p.startswith("[best_effort]") and "smaller than the fair budget" in p
            for p in problems
        )

    def test_a_drifted_best_effort_table_is_reported(self, tmp_path):
        bundle = self._add_best_effort(self._bundle(tmp_path))
        path = bundle / "aggregates" / "matrix_table_best_effort.json"
        table = json.loads(path.read_text())
        table[0]["nrmse_std"] = 0.123
        path.write_text(json.dumps(table))

        problems = validate_bundle_modes(bundle, expected_cells=4)
        assert any(
            p.startswith("[best_effort]") and "disagrees with the raw RunRecords" in p
            for p in problems
        )

    def test_records_whose_mode_contradicts_their_directory_are_refused(self, tmp_path):
        """Каталог режима и поле protocol.mode — два независимых утверждения о
        том, из какой матрицы запись. Пока их никто не сверял, опечатка в
        --output прятала best_effort-ячейки внутрь fair-контура: правило
        равного бюджета не применялось (режим по записям — не fair), порог
        best_effort не применялся (режим по каталогу — fair), а таблица уезжала
        в matrix_table_best_effort.json, оставляя headline-агрегат вообще без
        проверки."""
        bundle = self._bundle(tmp_path)
        records = [
            _reservoir_record("esn", 1, mode="best_effort", hpo_budget=200),
            _reservoir_record("esn", 24, mode="best_effort", hpo_budget=5),
        ]
        _write(records, bundle / "fair" / "runs")

        problems = validate_bundle_modes(bundle, modes=["fair"], expected_cells=None)
        assert any(
            p.startswith("[fair]") and "records declare mode" in p for p in problems
        )

    def test_the_fair_budget_is_read_even_when_only_best_effort_is_checked(self, tmp_path):
        """`--modes best_effort` не повод отключить порог: fair-прогоны лежат
        в том же бандле, и бюджет честного сравнения из них читается."""
        bundle = self._add_best_effort(self._bundle(tmp_path), hpo_budget=5)

        problems = validate_bundle_modes(
            bundle, modes=["best_effort"], expected_cells=4
        )
        assert any("smaller than the fair budget" in p for p in problems)

    def test_missing_fair_profiles_are_still_required(self, tmp_path):
        bundle = self._add_best_effort(self._bundle(tmp_path, profiles=False))

        problems = validate_bundle_modes(bundle, expected_cells=4)
        assert any(
            p.startswith("[fair]") and "missing resource-profile summary" in p
            for p in problems
        )
        assert validate_bundle_modes(bundle, expected_cells=4, require_profiles=False) == []
