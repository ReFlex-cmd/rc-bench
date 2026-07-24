"""
Smoke tests for the rcbench CLI.
Uses Typer's CliRunner — no DB, no Celery, pure in-process.
"""

import json
import tempfile
from pathlib import Path

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
