"""
Smoke tests for the rcbench CLI.
Uses Typer's CliRunner — no DB, no Celery, pure in-process.
"""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from rc_bench.cli.app import app

runner = CliRunner()

_ESN_SPEC = {
    "dataset": {"name": "narma10", "length": 400, "seed": 42},
    "reservoir": {"type": "esn", "params": {"n_units": 30}},
    # n_seeds=1 forces the single-run branch where result.metrics is populated;
    # multi-seed CLI tests should add it explicitly when needed.
    "protocol": {"washout": 20, "n_seeds": 1},
    "seed": 42,
}


# ---------------------------------------------------------------------------
# list-datasets
# ---------------------------------------------------------------------------

class TestListDatasets:
    def test_exits_zero(self):
        result = runner.invoke(app, ["list-datasets"])
        assert result.exit_code == 0

    def test_narma10_in_output(self):
        result = runner.invoke(app, ["list-datasets"])
        assert "narma10" in result.output


# ---------------------------------------------------------------------------
# list-reservoirs
# ---------------------------------------------------------------------------

class TestListReservoirs:
    def test_exits_zero(self):
        result = runner.invoke(app, ["list-reservoirs"])
        assert result.exit_code == 0

    def test_all_types_listed(self):
        result = runner.invoke(app, ["list-reservoirs"])
        for rtype in ("esn", "lsm", "fhn", "logistic"):
            assert rtype in result.output


# ---------------------------------------------------------------------------
# validate-spec
# ---------------------------------------------------------------------------

class TestValidateSpec:
    def _write_spec(self, tmp_path: Path, spec: dict, suffix: str = ".json") -> Path:
        p = tmp_path / f"spec{suffix}"
        if suffix in {".yaml", ".yml"}:
            import yaml
            p.write_text(yaml.dump(spec))
        else:
            p.write_text(json.dumps(spec))
        return p

    def test_valid_json_exits_zero(self, tmp_path):
        p = self._write_spec(tmp_path, _ESN_SPEC)
        result = runner.invoke(app, ["validate-spec", str(p)])
        assert result.exit_code == 0
        assert "Valid" in result.output

    def test_valid_yaml_exits_zero(self, tmp_path):
        p = self._write_spec(tmp_path, _ESN_SPEC, suffix=".yaml")
        result = runner.invoke(app, ["validate-spec", str(p)])
        assert result.exit_code == 0

    def test_invalid_reservoir_type_exits_nonzero(self, tmp_path):
        bad = {**_ESN_SPEC, "reservoir": {"type": "unicorn", "params": {}}}
        p = self._write_spec(tmp_path, bad)
        result = runner.invoke(app, ["validate-spec", str(p)])
        assert result.exit_code != 0

    def test_missing_file_exits_nonzero(self):
        result = runner.invoke(app, ["validate-spec", "/nonexistent/spec.json"])
        assert result.exit_code != 0

    def test_config_hash_in_output(self, tmp_path):
        p = self._write_spec(tmp_path, _ESN_SPEC)
        result = runner.invoke(app, ["validate-spec", str(p)])
        assert "config_hash=" in result.output


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

class TestRun:
    def _spec_file(self, tmp_path: Path) -> Path:
        p = tmp_path / "spec.json"
        p.write_text(json.dumps(_ESN_SPEC))
        return p

    def test_exits_zero(self, tmp_path):
        p = self._spec_file(tmp_path)
        result = runner.invoke(app, ["run", str(p)])
        assert result.exit_code == 0, result.output

    def test_metrics_in_output(self, tmp_path):
        p = self._spec_file(tmp_path)
        result = runner.invoke(app, ["run", str(p)])
        assert "nrmse_range" in result.output
        assert "prediction_horizon" in result.output

    def test_output_file_written(self, tmp_path):
        p = self._spec_file(tmp_path)
        out = tmp_path / "result.json"
        runner.invoke(app, ["run", str(p), "--output", str(out)])
        assert out.exists()
        record = json.loads(out.read_text())
        assert "spec" in record
        assert "result" in record
        assert record["result"]["status"] == "completed"

    def test_output_contains_all_metric_fields(self, tmp_path):
        p = self._spec_file(tmp_path)
        out = tmp_path / "result.json"
        runner.invoke(app, ["run", str(p), "--output", str(out)])
        metrics = json.loads(out.read_text())["result"]["metrics"]
        for field in ("nrmse_range", "nrmse_std", "nrmse_var", "rmse", "mae",
                      "prediction_horizon", "train_time", "inference_latency", "peak_memory"):
            assert field in metrics, f"Missing field: {field}"

    def test_seed_override(self, tmp_path):
        p = self._spec_file(tmp_path)
        out1 = tmp_path / "r1.json"
        out2 = tmp_path / "r2.json"
        runner.invoke(app, ["run", str(p), "--output", str(out1), "--seed", "42"])
        runner.invoke(app, ["run", str(p), "--output", str(out2), "--seed", "42"])
        m1 = json.loads(out1.read_text())["result"]["metrics"]["nrmse_range"]
        m2 = json.loads(out2.read_text())["result"]["metrics"]["nrmse_range"]
        assert m1 == m2

    def test_forwards_protocol_split_fractions_to_data_provider(
        self, tmp_path, monkeypatch
    ):
        spec = {
            **_ESN_SPEC,
            "protocol": {
                "washout": 20,
                "n_seeds": 1,
                "train_frac": 0.5,
                "val_frac": 0.3,
            },
        }
        p = tmp_path / "spec.json"
        p.write_text(json.dumps(spec))
        captured = {}
        data = object()

        def fake_get_data_for_experiment(**kwargs):
            captured.update(kwargs)
            return data

        def fake_run_pipeline(actual_data, *_args, **_kwargs):
            assert actual_data is data
            return object()

        monkeypatch.setattr(
            "rc_bench.core.data_provider.get_data_for_experiment",
            fake_get_data_for_experiment,
        )
        monkeypatch.setattr(
            "rc_bench.runners.pipeline.run_pipeline",
            fake_run_pipeline,
        )
        monkeypatch.setattr("rc_bench.cli.app._print_result", lambda _result: None)

        result = runner.invoke(app, ["run", str(p)])

        assert result.exit_code == 0, result.output
        assert captured["train_frac"] == 0.5
        assert captured["val_frac"] == 0.3

    def test_missing_spec_file_exits_nonzero(self):
        result = runner.invoke(app, ["run", "/nonexistent.json"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------

class TestAggregate:
    def _make_records(self, tmp_path: Path, n: int = 3) -> Path:
        """Run n experiments and write their records to tmp_path."""
        from rc_bench.cli.app import run_cmd
        rtypes = ["esn", "lsm", "fhn", "logistic"]
        for i in range(n):
            spec = {
                **_ESN_SPEC,
                "reservoir": {"type": rtypes[i % len(rtypes)], "params": {"n_units": 30}},
                "seed": i,
            }
            spec_file = tmp_path / f"spec_{i}.json"
            spec_file.write_text(json.dumps(spec))
            out_file = tmp_path / f"run_{i}.json"
            runner.invoke(app, ["run", str(spec_file), "--output", str(out_file)])
        return tmp_path

    def test_exits_zero(self, tmp_path):
        d = self._make_records(tmp_path, n=2)
        result = runner.invoke(app, ["aggregate", str(d)])
        assert result.exit_code == 0

    def test_reservoir_types_in_output(self, tmp_path):
        d = self._make_records(tmp_path, n=3)
        result = runner.invoke(app, ["aggregate", str(d)])
        assert "esn" in result.output or "lsm" in result.output

    def test_sort_by_column(self, tmp_path):
        d = self._make_records(tmp_path, n=2)
        result = runner.invoke(app, ["aggregate", str(d), "--sort-by", "rmse"])
        assert result.exit_code == 0

    def test_top_filter(self, tmp_path):
        d = self._make_records(tmp_path, n=3)
        result = runner.invoke(app, ["aggregate", str(d), "--top", "1"])
        assert result.exit_code == 0

    def test_empty_dir_exits_zero(self, tmp_path):
        result = runner.invoke(app, ["aggregate", str(tmp_path)])
        assert result.exit_code == 0

    def test_nonexistent_dir_exits_nonzero(self):
        result = runner.invoke(app, ["aggregate", "/nonexistent_dir"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# eda
# ---------------------------------------------------------------------------


class TestEDA:
    def test_missing_raw_file_exits_nonzero(self, tmp_path):
        result = runner.invoke(
            app,
            ["eda", str(tmp_path / "missing.txt")],
        )
        assert result.exit_code != 0
        assert "Raw data file not found" in result.output

    def test_manifest_filename_must_match_raw_file(self, tmp_path, monkeypatch):
        raw_path = tmp_path / "wrong.txt"
        raw_path.write_text("fixture")
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text("{}")
        monkeypatch.setattr(
            "rc_bench.data.download.DatasetManifest.load",
            lambda _path: SimpleNamespace(
                raw_filename="household_power_consumption.txt",
                raw_sha256="a" * 64,
            ),
        )

        result = runner.invoke(
            app,
            [
                "eda",
                str(raw_path),
                "--manifest",
                str(manifest_path),
                "--output-dir",
                str(tmp_path / "report"),
            ],
        )

        assert result.exit_code != 0
        assert "does not match manifest filename" in result.output

    def test_verified_raw_is_loaded_and_reported(self, tmp_path, monkeypatch):
        raw_path = tmp_path / "household_power_consumption.txt"
        raw_path.write_text("fixture")
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text("{}")
        output_dir = tmp_path / "report"
        series = object()
        captured = {}

        monkeypatch.setattr(
            "rc_bench.data.download.DatasetManifest.load",
            lambda _path: SimpleNamespace(
                raw_filename=raw_path.name,
                raw_sha256="b" * 64,
            ),
        )
        monkeypatch.setattr(
            "rc_bench.data.download.download_dataset",
            lambda _manifest, _directory: raw_path,
        )
        monkeypatch.setattr(
            "rc_bench.data.jmlc.load_uci_household_power_series",
            lambda _path: series,
        )

        def fake_generate(actual_series, actual_output, *, raw_sha256):
            captured.update(
                series=actual_series,
                output=actual_output,
                raw_sha256=raw_sha256,
            )
            return [actual_output / "eda_report.md"]

        monkeypatch.setattr(
            "rc_bench.reporting.eda.generate_eda_report",
            fake_generate,
        )

        result = runner.invoke(
            app,
            [
                "eda",
                str(raw_path),
                "--manifest",
                str(manifest_path),
                "--output-dir",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0, result.output
        assert captured == {
            "series": series,
            "output": output_dir,
            "raw_sha256": "b" * 64,
        }
        assert "eda_report.md" in result.output
