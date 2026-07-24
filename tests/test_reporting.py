"""Tests for Stage 9: Reporting + Reproducibility."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from rc_bench.core.data_provider import get_data_for_experiment
from rc_bench.core.schema import (
    BaselineSpec,
    DatasetSpec,
    EnergyResult,
    ExperimentSpec,
    ProtocolSpec,
    ReadoutSpec,
    ReservoirSpec,
    ResultSpec,
)
from rc_bench.reporting.report import _get_metrics_dict, generate_report
from rc_bench.reporting.run_record import RunRecord, load_run_record, save_run_record
from rc_bench.runners.pipeline import run_pipeline

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_T = 300
_DATA = get_data_for_experiment("narma10", length=_T, seed=42)

_SPEC = ExperimentSpec(
    dataset=DatasetSpec(name="narma10", length=_T),
    reservoir=ReservoirSpec(type="esn", params={"units": 20}),
    protocol=ProtocolSpec(washout=20, train_frac=0.6, val_frac=0.2, n_seeds=1),
    readout=ReadoutSpec(alpha_grid=[0.01, 0.1, 1.0]),
    seed=42,
)


def _run() -> ResultSpec:
    return run_pipeline(_DATA, _SPEC)


def _make_record(rtype: str = "esn", seed: int = 42) -> RunRecord:
    spec = _SPEC.model_copy(
        deep=True,
        update={"reservoir": ReservoirSpec(type=rtype, params={"units": 20}), "seed": seed},
    )
    result = run_pipeline(_DATA, spec)
    return RunRecord.make(spec, result)


def _make_baseline_record(btype: str = "persistence") -> RunRecord:
    # washout >= seasonal period so the lag-24 reference stays within the split.
    spec = ExperimentSpec(
        dataset=DatasetSpec(name="narma10", length=_T),
        baseline=BaselineSpec(type=btype),
        protocol=ProtocolSpec(
            washout=30,
            train_frac=0.6,
            val_frac=0.2,
            forecasting_mode="fixed_horizon",
            horizon=1,
            n_seeds=0,
            selection_metric="nrmse_std",
            seasonal_period=24,
        ),
        readout=ReadoutSpec(alpha_grid=[0.01, 0.1, 1.0]),
        seed=None,
    )
    result = run_pipeline(_DATA, spec)
    return RunRecord.make(spec, result)


# ---------------------------------------------------------------------------
# TestRunRecord
# ---------------------------------------------------------------------------


class TestRunRecord:
    def test_make_populates_metadata(self):
        result = _run()
        rec = RunRecord.make(_SPEC, result)
        assert rec.timestamp != ""
        assert rec.hostname != ""
        assert rec.python_version != ""
        assert rec.rc_bench_version != ""

    def test_make_preserves_spec_and_result(self):
        result = _run()
        rec = RunRecord.make(_SPEC, result)
        assert rec.spec == _SPEC
        assert rec.resolved_spec == result.resolved_spec
        assert rec.result.status == "completed"
        assert rec.result.config_hash == result.config_hash

    def test_roundtrip_json(self, tmp_path):
        result = _run()
        rec = RunRecord.make(_SPEC, result)
        path = tmp_path / "run.json"
        save_run_record(rec, path)
        loaded = load_run_record(path)
        assert loaded.spec == rec.spec
        assert loaded.resolved_spec == rec.resolved_spec
        assert loaded.result.config_hash == rec.result.config_hash
        assert loaded.timestamp == rec.timestamp
        assert loaded.hostname == rec.hostname

    def test_load_legacy_format(self, tmp_path):
        """Files saved by the old CLI (no metadata fields) must load without error."""
        result = _run()
        legacy_result = result.model_dump()
        legacy_result.pop("resolved_spec")
        legacy_result.pop("frozen_config_hash")
        legacy = {"spec": _SPEC.model_dump(), "result": legacy_result}
        path = tmp_path / "legacy.json"
        path.write_text(json.dumps(legacy, default=str))
        loaded = load_run_record(path)
        assert loaded.spec == _SPEC
        assert loaded.resolved_spec == _SPEC
        assert loaded.timestamp == ""  # default value

    def test_save_creates_parent_dirs(self, tmp_path):
        result = _run()
        rec = RunRecord.make(_SPEC, result)
        deep = tmp_path / "a" / "b" / "run.json"
        save_run_record(rec, deep)
        assert deep.exists()

    def test_git_hash_type(self):
        result = _run()
        rec = RunRecord.make(_SPEC, result)
        assert rec.git_hash is None or isinstance(rec.git_hash, str)

    def test_config_hash_stable_across_records(self):
        result = _run()
        rec1 = RunRecord.make(_SPEC, result)
        rec2 = RunRecord.make(_SPEC, result)
        assert rec1.result.config_hash == rec2.result.config_hash

    def test_json_file_is_valid_json(self, tmp_path):
        result = _run()
        rec = RunRecord.make(_SPEC, result)
        path = tmp_path / "run.json"
        save_run_record(rec, path)
        parsed = json.loads(path.read_text())
        assert "spec" in parsed
        assert "resolved_spec" in parsed
        assert "result" in parsed
        assert "timestamp" in parsed

    def test_hpo_record_keeps_atomic_frozen_resolved_pair(self):
        spec = _SPEC.model_copy(
            deep=True,
            update={
                "protocol": _SPEC.protocol.model_copy(
                    update={"use_hpo": True, "hpo_budget": 3}
                )
            },
        )
        result = run_pipeline(_DATA, spec)

        rec = RunRecord.make(spec, result)

        assert rec.spec == spec
        assert rec.resolved_spec == result.resolved_spec
        assert rec.spec.config_hash() == result.frozen_config_hash
        assert rec.resolved_spec.config_hash() == result.config_hash
        assert rec.spec != rec.resolved_spec


class TestEnergyContract:
    def test_pipeline_reports_energy_unavailable(self):
        result = _run()

        assert result.energy == EnergyResult()
        assert result.energy.model_dump() == {
            "status": "unavailable",
            "reason": "No supported hardware energy counter available",
            "backend": None,
        }

    def test_energy_contract_rejects_claimed_backend(self):
        with pytest.raises(ValueError):
            EnergyResult(backend="tdp-estimate")


# ---------------------------------------------------------------------------
# TestGetMetricsDict
# ---------------------------------------------------------------------------


class TestGetMetricsDict:
    def test_returns_all_metric_cols(self):
        result = _run()
        rec = RunRecord.make(_SPEC, result)
        d = _get_metrics_dict(rec)
        for col in ("nrmse_range", "rmse", "mae", "prediction_horizon", "train_time"):
            assert col in d

    def test_values_are_numeric(self):
        result = _run()
        rec = RunRecord.make(_SPEC, result)
        d = _get_metrics_dict(rec)
        for v in d.values():
            assert v is None or isinstance(v, (int, float))

    def test_returns_empty_for_failed_result(self):
        failed = ResultSpec(status="failed", config_hash="abc", error="boom")
        rec = RunRecord(spec=_SPEC, result=failed)
        assert _get_metrics_dict(rec) == {}


# ---------------------------------------------------------------------------
# TestGenerateReport
# ---------------------------------------------------------------------------


class TestGenerateReport:
    def _records(self, n: int = 2) -> list:
        types = ["esn", "fhn", "logistic"]
        return [_make_record(types[i % len(types)], seed=i) for i in range(n)]

    def test_creates_markdown_and_csv(self, tmp_path):
        generate_report(self._records(2), tmp_path)
        assert (tmp_path / "report.md").exists()
        assert (tmp_path / "report.csv").exists()

    def test_markdown_contains_header(self, tmp_path):
        generate_report(self._records(2), tmp_path)
        md = (tmp_path / "report.md").read_text()
        assert "# RC-Bench Report" in md
        assert "Records: 2" in md

    def test_markdown_contains_metric_column(self, tmp_path):
        generate_report(self._records(2), tmp_path)
        md = (tmp_path / "report.md").read_text()
        assert "nrmse_range" in md

    def test_csv_row_count(self, tmp_path):
        generate_report(self._records(3), tmp_path)
        lines = (tmp_path / "report.csv").read_text().splitlines()
        assert len(lines) == 4  # header + 3 data rows

    def test_csv_sorted_by_metric(self, tmp_path):
        generate_report(self._records(3), tmp_path, sort_metric="nrmse_range")
        with (tmp_path / "report.csv").open() as f:
            rows = list(csv.DictReader(f))
        values = [
            float(r["nrmse_range"])
            for r in rows
            if r.get("nrmse_range") not in (None, "", "None", "nan", "-")
        ]
        assert values == sorted(values)

    def test_csv_has_reservoir_and_dataset_columns(self, tmp_path):
        generate_report(self._records(2), tmp_path)
        with (tmp_path / "report.csv").open() as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
        assert "reservoir" in fieldnames
        assert "dataset" in fieldnames

    def test_empty_records_creates_files(self, tmp_path):
        generate_report([], tmp_path)
        assert (tmp_path / "report.md").exists()
        assert (tmp_path / "report.csv").read_text() == ""

    def test_output_dir_created_if_missing(self, tmp_path):
        new_dir = tmp_path / "new" / "subdir"
        generate_report(self._records(1), new_dir)
        assert (new_dir / "report.md").exists()

    def test_multi_seed_record_included(self, tmp_path):
        spec = _SPEC.model_copy(
            deep=True,
            update={"protocol": ProtocolSpec(washout=20, train_frac=0.6, val_frac=0.2, n_seeds=2)},
        )
        result = run_pipeline(_DATA, spec)
        rec = RunRecord.make(spec, result)
        generate_report([rec], tmp_path)
        with (tmp_path / "report.csv").open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["nrmse_range"] not in ("", "-", "None")


class TestBaselineReporting:
    def test_report_includes_baseline_without_crashing(self, tmp_path):
        rec = _make_baseline_record("persistence")
        generate_report([rec], tmp_path)
        assert rec.result.model_family == "baseline"
        assert "persistence" in (tmp_path / "report.csv").read_text()
        assert "persistence" in (tmp_path / "report.md").read_text()

    def test_plot_metric_bar_accepts_baseline(self, tmp_path):
        from rc_bench.reporting.plots import plot_metric_bar

        rec = _make_baseline_record("ridge_ar")
        out = tmp_path / "bar.png"
        plot_metric_bar([rec], metric="nrmse_range", output_path=out)
        assert out.exists()
        assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# TestPlots
# ---------------------------------------------------------------------------


class TestPlots:
    def test_plot_metric_bar_creates_file(self, tmp_path):
        from rc_bench.reporting.plots import plot_metric_bar

        records = [_make_record("esn"), _make_record("fhn")]
        out = tmp_path / "bar.png"
        plot_metric_bar(records, metric="nrmse_range", output_path=out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_plot_predictions_creates_file(self, tmp_path):
        from rc_bench.reporting.plots import plot_predictions

        rng = np.random.default_rng(0)
        y_true = rng.random(200)
        y_pred = y_true + rng.normal(0, 0.05, 200)
        out = tmp_path / "preds.png"
        plot_predictions(y_true, y_pred, output_path=out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_plot_metric_bar_empty_no_file(self, tmp_path):
        from rc_bench.reporting.plots import plot_metric_bar

        out = tmp_path / "bar.png"
        plot_metric_bar([], metric="nrmse_range", output_path=out)
        assert not out.exists()

    def test_plot_predictions_respects_max_steps(self, tmp_path):
        from rc_bench.reporting.plots import plot_predictions

        rng = np.random.default_rng(1)
        y = rng.random(1000)
        out = tmp_path / "preds_short.png"
        plot_predictions(y, y, output_path=out, max_steps=50)
        assert out.exists()


# ---------------------------------------------------------------------------
# TestPipelineArtifacts
# ---------------------------------------------------------------------------


class TestPipelineArtifacts:
    def test_no_artifacts_by_default(self):
        result = run_pipeline(_DATA, _SPEC)
        assert result.artifact_paths == {}

    def test_artifact_dir_alone_does_not_enable_predictions(self, tmp_path):
        art_dir = tmp_path / "artifacts"
        result = run_pipeline(_DATA, _SPEC, artifact_dir=art_dir)
        assert result.artifact_paths == {}
        assert not art_dir.exists()

    def test_predictions_npz_saved(self, tmp_path):
        result = run_pipeline(
            _DATA,
            _SPEC,
            artifact_dir=tmp_path,
            save_predictions=True,
        )
        assert "predictions" in result.artifact_paths
        npz_path = Path(result.artifact_paths["predictions"])
        assert npz_path.exists()
        arrays = np.load(npz_path)
        assert "y_test" in arrays
        assert "y_pred" in arrays
        assert arrays["seed"].item() == _SPEC.seed
        assert len(arrays["y_test"]) == len(arrays["y_pred"])

    def test_artifact_dir_created_if_missing(self, tmp_path):
        art_dir = tmp_path / "nested" / "artifacts"
        result = run_pipeline(
            _DATA,
            _SPEC,
            artifact_dir=art_dir,
            save_predictions=True,
        )
        assert art_dir.exists()
        assert "predictions" in result.artifact_paths

    def test_artifact_path_contains_config_hash(self, tmp_path):
        result = run_pipeline(
            _DATA,
            _SPEC,
            artifact_dir=tmp_path,
            save_predictions=True,
        )
        npz_name = Path(result.artifact_paths["predictions"]).name
        assert _SPEC.config_hash() in npz_name

    def test_predictions_opt_in_requires_artifact_dir(self):
        with pytest.raises(ValueError, match="artifact_dir"):
            run_pipeline(_DATA, _SPEC, save_predictions=True)

    def test_multi_seed_artifact_dir_alone_does_not_save_predictions(self, tmp_path):
        spec = _SPEC.model_copy(
            deep=True,
            update={"protocol": ProtocolSpec(washout=20, train_frac=0.6, val_frac=0.2, n_seeds=2)},
        )
        art_dir = tmp_path / "artifacts"
        result = run_pipeline(_DATA, spec, artifact_dir=art_dir)
        assert "predictions" not in result.artifact_paths
        assert not art_dir.exists()

    def test_multi_seed_opt_in_saves_one_representative_seed(self, tmp_path):
        spec = _SPEC.model_copy(
            deep=True,
            update={"protocol": ProtocolSpec(washout=20, train_frac=0.6, val_frac=0.2, n_seeds=3)},
        )
        result = run_pipeline(
            _DATA,
            spec,
            artifact_dir=tmp_path,
            save_predictions=True,
        )

        assert result.multi_seed_result is not None
        assert result.multi_seed_result.n_seeds == 3
        assert "predictions" in result.artifact_paths
        assert len(list(tmp_path.glob("*.npz"))) == 1

        npz_path = Path(result.artifact_paths["predictions"])
        assert f"seed{spec.seed}" in npz_path.name
        arrays = np.load(npz_path)
        assert arrays["seed"].item() == spec.seed
        assert len(arrays["y_test"]) == len(arrays["y_pred"])


# ---------------------------------------------------------------------------
# TestCLIReport
# ---------------------------------------------------------------------------


class TestCLIReport:
    def test_report_command_exit_zero(self, tmp_path):
        from typer.testing import CliRunner

        from rc_bench.cli.app import app

        result = run_pipeline(_DATA, _SPEC)
        rec = RunRecord.make(_SPEC, result)
        save_run_record(rec, tmp_path / "run1.json")

        out_dir = tmp_path / "out"
        runner = CliRunner()
        r = runner.invoke(app, ["report", str(tmp_path), "--output-dir", str(out_dir), "--no-plots"])
        assert r.exit_code == 0, r.output
        assert (out_dir / "report.md").exists()
        assert (out_dir / "report.csv").exists()

    def test_report_command_prints_paths(self, tmp_path):
        from typer.testing import CliRunner

        from rc_bench.cli.app import app

        result = run_pipeline(_DATA, _SPEC)
        rec = RunRecord.make(_SPEC, result)
        save_run_record(rec, tmp_path / "run1.json")

        runner = CliRunner()
        r = runner.invoke(app, ["report", str(tmp_path), "--no-plots"])
        assert "report.md" in r.output
        assert "report.csv" in r.output

    def test_run_cmd_saves_run_record(self, tmp_path):
        from typer.testing import CliRunner

        from rc_bench.cli.app import app

        spec_data = {
            "dataset": {"name": "narma10", "length": 300},
            "reservoir": {"type": "esn", "params": {"units": 20}},
            "protocol": {"washout": 20, "train_frac": 0.6, "val_frac": 0.2, "n_seeds": 1},
            "readout": {"alpha_grid": [0.01, 0.1, 1.0]},
            "seed": 42,
        }
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text(yaml.dump(spec_data))

        out_file = tmp_path / "run.json"
        runner = CliRunner()
        r = runner.invoke(app, ["run", str(spec_file), "--output", str(out_file)])
        assert r.exit_code == 0, r.output
        assert out_file.exists()

        loaded = load_run_record(out_file)
        assert loaded.result.status == "completed"
        assert loaded.timestamp != ""
        assert loaded.hostname != ""

    def test_run_cmd_artifacts_flag(self, tmp_path):
        from typer.testing import CliRunner

        from rc_bench.cli.app import app

        spec_data = {
            "dataset": {"name": "narma10", "length": 300},
            "reservoir": {"type": "esn", "params": {"units": 20}},
            "protocol": {"washout": 20, "train_frac": 0.6, "val_frac": 0.2, "n_seeds": 1},
            "readout": {"alpha_grid": [0.01, 0.1, 1.0]},
            "seed": 42,
        }
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text(yaml.dump(spec_data))

        art_dir = tmp_path / "artifacts"
        runner = CliRunner()
        r = runner.invoke(app, ["run", str(spec_file), "--artifacts", str(art_dir)])
        assert r.exit_code == 0, r.output
        npz_files = list(art_dir.glob("*.npz"))
        assert len(npz_files) == 1

    def test_report_empty_dir_exits_zero(self, tmp_path):
        from typer.testing import CliRunner

        from rc_bench.cli.app import app

        runner = CliRunner()
        r = runner.invoke(app, ["report", str(tmp_path)])
        assert r.exit_code == 0
