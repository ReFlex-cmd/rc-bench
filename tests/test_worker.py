"""Tests for Stage 10: CLI/worker update.

Covers:
- _load_predictions_from_artifact helper (extracted from main.py plot endpoint)
- Pipeline artifact format compatibility with the plot endpoint
- ResultSpec availability in worker context (regression for missing-import bug)
- Config ARTIFACT_DIR default
- Worker artifact directory structure (settings.ARTIFACT_DIR / str(experiment_id))
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rc_bench.core.data_provider import get_data_for_experiment
from rc_bench.core.schema import (
    DatasetSpec,
    ExperimentSpec,
    ProtocolSpec,
    ReadoutSpec,
    ReservoirSpec,
    ResultSpec,
)
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

# ---------------------------------------------------------------------------
# TestLoadPredictionsHelper
# ---------------------------------------------------------------------------


class TestLoadPredictionsHelper:
    """Unit tests for main._load_predictions_from_artifact (no DB/config needed)."""

    # Import lazily so tests don't pull in Celery/DB at collection time.
    @staticmethod
    def _helper():
        from rc_bench.main import _load_predictions_from_artifact
        return _load_predictions_from_artifact

    def test_loads_y_test_and_y_pred(self, tmp_path):
        fn = self._helper()
        rng = np.random.default_rng(0)
        y_test = rng.random(100)
        y_pred = rng.random(100)
        npz = tmp_path / "preds.npz"
        np.savez(npz, y_test=y_test, y_pred=y_pred)

        loaded_y_test, loaded_y_pred = fn({"predictions": str(npz)})
        np.testing.assert_array_equal(loaded_y_test, y_test)
        np.testing.assert_array_equal(loaded_y_pred, y_pred)

    def test_raises_if_predictions_key_missing(self):
        fn = self._helper()
        with pytest.raises(FileNotFoundError, match="predictions"):
            fn({})

    def test_raises_if_predictions_key_empty_dict(self):
        fn = self._helper()
        with pytest.raises(FileNotFoundError):
            fn({"other_key": "/some/path.npz"})

    def test_raises_if_file_missing_on_disk(self, tmp_path):
        fn = self._helper()
        missing = tmp_path / "ghost.npz"
        with pytest.raises(FileNotFoundError, match="missing on disk"):
            fn({"predictions": str(missing)})

    def test_returns_numpy_arrays(self, tmp_path):
        fn = self._helper()
        npz = tmp_path / "p.npz"
        np.savez(npz, y_test=np.ones(50), y_pred=np.zeros(50))
        yt, yp = fn({"predictions": str(npz)})
        assert isinstance(yt, np.ndarray)
        assert isinstance(yp, np.ndarray)

    def test_shapes_match(self, tmp_path):
        fn = self._helper()
        n = 200
        npz = tmp_path / "p.npz"
        np.savez(npz, y_test=np.ones(n), y_pred=np.zeros(n))
        yt, yp = fn({"predictions": str(npz)})
        assert yt.shape == yp.shape == (n,)


# ---------------------------------------------------------------------------
# TestPipelineArtifactFormatCompatibility
# ---------------------------------------------------------------------------


class TestPipelineArtifactFormatCompatibility:
    """Verify that pipeline's npz output is readable by the plot endpoint helper."""

    def _run_with_artifacts(self, tmp_path: Path) -> ResultSpec:
        return run_pipeline(
            _DATA,
            _SPEC,
            artifact_dir=tmp_path,
            save_predictions=True,
        )

    def test_predictions_key_present(self, tmp_path):
        result = self._run_with_artifacts(tmp_path)
        assert "predictions" in result.artifact_paths

    def test_npz_readable_by_plot_helper(self, tmp_path):
        from rc_bench.main import _load_predictions_from_artifact

        result = self._run_with_artifacts(tmp_path)
        y_test, y_pred = _load_predictions_from_artifact(result.artifact_paths)
        assert y_test is not None
        assert y_pred is not None

    def test_npz_arrays_have_correct_keys(self, tmp_path):
        result = self._run_with_artifacts(tmp_path)
        npz_path = Path(result.artifact_paths["predictions"])
        arrays = np.load(npz_path)
        assert set(arrays.files) >= {"y_test", "y_pred"}

    def test_npz_arrays_are_1d_and_same_length(self, tmp_path):
        result = self._run_with_artifacts(tmp_path)
        npz_path = Path(result.artifact_paths["predictions"])
        arrays = np.load(npz_path)
        assert arrays["y_test"].ndim == 1
        assert arrays["y_pred"].ndim == 1
        assert len(arrays["y_test"]) == len(arrays["y_pred"])

    def test_no_legacy_separate_file_keys(self, tmp_path):
        """The old format used 'preds' + 'y_test' as separate paths. Verify it's gone."""
        result = self._run_with_artifacts(tmp_path)
        assert "preds" not in result.artifact_paths
        assert "y_test" not in result.artifact_paths


# ---------------------------------------------------------------------------
# TestResultSpecInWorkerContext
# ---------------------------------------------------------------------------


class TestResultSpecInWorkerContext:
    """Regression tests covering the missing ResultSpec import in tasks.py.

    These tests verify ResultSpec can be constructed the same way the worker does
    in both success and failure paths, without importing tasks.py (which needs Redis).
    """

    def test_failed_result_spec_constructs(self):
        spec = ResultSpec(status="failed", config_hash="", error="boom")
        assert spec.status == "failed"
        assert spec.error == "boom"
        assert spec.metrics is None

    def test_completed_result_spec_from_pipeline(self):
        result = run_pipeline(_DATA, _SPEC)
        assert result.status == "completed"
        assert result.config_hash != ""
        assert result.metrics is not None

    def test_result_spec_model_dump_is_json_serializable(self):
        import json
        result = run_pipeline(_DATA, _SPEC)
        dumped = result.model_dump()
        # Must be serializable (this is what the worker stores in JSONB)
        json.dumps(dumped, default=str)


# ---------------------------------------------------------------------------
# TestWorkerArtifactDirectoryStructure
# ---------------------------------------------------------------------------


class TestWorkerArtifactDirectoryStructure:
    """Verify the worker artifact path convention: ARTIFACT_DIR / str(experiment_id)."""

    def test_per_experiment_subdirectory(self, tmp_path):
        artifact_dir = tmp_path / "123"
        result = run_pipeline(
            _DATA,
            _SPEC,
            artifact_dir=artifact_dir,
            save_predictions=True,
        )
        assert artifact_dir.exists()
        npz = Path(result.artifact_paths["predictions"])
        assert npz.parent == artifact_dir

    def test_different_experiments_get_separate_dirs(self, tmp_path):
        dir_1 = tmp_path / "1"
        dir_2 = tmp_path / "2"
        r1 = run_pipeline(
            _DATA,
            _SPEC,
            artifact_dir=dir_1,
            save_predictions=True,
        )
        r2 = run_pipeline(
            _DATA,
            _SPEC,
            artifact_dir=dir_2,
            save_predictions=True,
        )
        assert Path(r1.artifact_paths["predictions"]).parent != \
               Path(r2.artifact_paths["predictions"]).parent

    def test_artifact_dir_created_if_not_exists(self, tmp_path):
        deep = tmp_path / "base" / "42"
        assert not deep.exists()
        run_pipeline(
            _DATA,
            _SPEC,
            artifact_dir=deep,
            save_predictions=True,
        )
        assert deep.exists()


# ---------------------------------------------------------------------------
# TestWorkerSplitFractions
# ---------------------------------------------------------------------------


class TestWorkerSplitFractions:
    """Regression coverage for protocol split fractions in the Celery entry point."""

    def test_forwards_protocol_split_fractions_to_data_provider(
        self, tmp_path, monkeypatch
    ):
        from rc_bench import tasks

        experiment_id = 17
        experiment = SimpleNamespace(
            config={
                "dataset": {"name": "narma10", "length": 300, "seed": 42},
                "reservoir": {"type": "esn", "params": {"n_units": 20}},
                "protocol": {
                    "washout": 20,
                    "n_seeds": 1,
                    "train_frac": 0.5,
                    "val_frac": 0.3,
                },
                "seed": 42,
            },
            status=tasks.ExperimentStatus.QUEUED,
        )
        captured = {}
        data = object()

        class FakeSession:
            def __init__(self):
                self.added = []

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def get(self, model, actual_id):
                assert model is tasks.Experiment
                assert actual_id == experiment_id
                return experiment

            def add(self, value):
                self.added.append(value)

            def commit(self):
                pass

        session = FakeSession()

        def fake_get_data_for_experiment(**kwargs):
            captured.update(kwargs)
            return data

        def fake_run_pipeline(actual_data, *_args, **_kwargs):
            assert actual_data is data
            assert _kwargs["save_predictions"] is False
            return ResultSpec(status="completed", config_hash="test-config")

        monkeypatch.setattr(tasks, "get_sync_db_session", lambda: session)
        monkeypatch.setattr(
            tasks, "get_data_for_experiment", fake_get_data_for_experiment
        )
        monkeypatch.setattr(tasks, "run_pipeline", fake_run_pipeline)
        monkeypatch.setattr(tasks.settings, "ARTIFACT_DIR", tmp_path)

        tasks.run_experiment_task.run(experiment_id)

        assert captured["train_frac"] == 0.5
        assert captured["val_frac"] == 0.3
        assert experiment.status == tasks.ExperimentStatus.COMPLETED


# ---------------------------------------------------------------------------
# TestConfigArtifactDir
# ---------------------------------------------------------------------------


class TestConfigArtifactDir:
    """Test Settings.ARTIFACT_DIR default without triggering module-level settings instance."""

    def test_artifact_dir_field_exists(self):
        from rc_bench.config import Settings
        assert "ARTIFACT_DIR" in Settings.model_fields

    def test_artifact_dir_default_is_path(self):
        from rc_bench.config import Settings
        field = Settings.model_fields["ARTIFACT_DIR"]
        # Default should resolve to a Path
        default_val = field.default
        assert isinstance(default_val, Path)

    def test_artifact_dir_default_value(self):
        from rc_bench.config import Settings
        field = Settings.model_fields["ARTIFACT_DIR"]
        assert Path("outputs/artifacts") in Path(str(field.default)).parents \
               or "artifacts" in str(field.default)
