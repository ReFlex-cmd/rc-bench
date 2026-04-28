"""
Tests for the protocol layer (Stage 4):
  - TimeSeriesSplitter
  - StateCollector
  - build_one_step_targets / build_fixed_horizon_targets / closed_loop_predict
  - reset_state() / step() / warmup() on all four reservoir types
  - run_experiment() with all three forecasting modes
"""

import numpy as np
import pytest

from rc_bench.protocol.splitter import Split, TimeSeriesSplitter
from rc_bench.protocol.state_collector import StateCollector
from rc_bench.protocol.forecasting import (
    build_one_step_targets,
    build_fixed_horizon_targets,
    closed_loop_predict,
)
from rc_bench.core.data_provider import get_data_for_experiment
from rc_bench.core.schema import (
    DatasetSpec, ExperimentSpec, ProtocolSpec, ReadoutSpec, ReservoirSpec,
)
from rc_bench.core.reservoirs.esn_service import ESNReservoir
from rc_bench.core.reservoirs.fhn_service import FHNReservoir
from rc_bench.core.reservoirs.lsm_service import LSMReservoir
from rc_bench.core.reservoirs.logistic_service import LogisticReservoir
from rc_bench.core.reservoirs.registry import get_reservoir
from rc_bench.runners.experiment_runner import run_experiment


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_T = 300
_DATA = get_data_for_experiment("narma10", length=_T, seed=42)

_WASHOUT = 20
_UNITS = 20


def _esn(seed=42):
    return ESNReservoir({"n_units": _UNITS, "seed": seed})


def _fhn(seed=42):
    return FHNReservoir({"units": _UNITS, "density": 0.1, "dt": 0.1, "internal_steps": 1, "seed": seed})


def _lsm(seed=42):
    return LSMReservoir({"units": _UNITS, "density": 0.1, "dt": 1.0, "tau_mem": 5.0, "seed": seed})


def _logistic(seed=42):
    return LogisticReservoir({"units": _UNITS, "seed": seed})


def _spec(mode, horizon=1):
    return ExperimentSpec(
        dataset=DatasetSpec(name="narma10", length=_T),
        reservoir=ReservoirSpec(type="fhn", params={"units": _UNITS, "density": 0.1, "dt": 0.1, "internal_steps": 1}),
        protocol=ProtocolSpec(washout=_WASHOUT, train_frac=0.6, val_frac=0.2,
                              forecasting_mode=mode, horizon=horizon),
        readout=ReadoutSpec(alpha_grid=[0.01, 0.1, 1.0]),
        seed=42,
    )


# ---------------------------------------------------------------------------
# TimeSeriesSplitter
# ---------------------------------------------------------------------------

class TestTimeSeriesSplitter:
    def test_split_sizes(self):
        T = 100
        X = np.zeros((T, 1))
        y = np.arange(T, dtype=float)
        splits = TimeSeriesSplitter(train_frac=0.6, val_frac=0.2).split(X, y)
        assert len(splits["train"].y) == 60
        assert len(splits["val"].y) == 20
        assert len(splits["test"].y) == 20

    def test_no_overlap(self):
        T = 100
        X = np.arange(T).reshape(-1, 1).astype(float)
        y = np.arange(T, dtype=float)
        splits = TimeSeriesSplitter().split(X, y)
        train_last = splits["train"].X[-1, 0]
        val_first = splits["val"].X[0, 0]
        assert val_first == train_last + 1

    def test_contiguous_coverage(self):
        T = 200
        y = np.arange(T, dtype=float)
        splits = TimeSeriesSplitter().split(y.reshape(-1, 1), y)
        total = sum(len(s.y) for s in splits.values())
        assert total == T

    def test_invalid_fracs_raise(self):
        with pytest.raises(ValueError):
            TimeSeriesSplitter(train_frac=0.7, val_frac=0.4)

    def test_returns_split_dataclass(self):
        X = np.zeros((50, 1))
        y = np.zeros(50)
        splits = TimeSeriesSplitter().split(X, y)
        for s in splits.values():
            assert isinstance(s, Split)


# ---------------------------------------------------------------------------
# StateCollector
# ---------------------------------------------------------------------------

class TestStateCollector:
    def test_output_shape(self):
        res = _fhn()
        X = _DATA["X_train"]
        col = StateCollector(res, washout=_WASHOUT)
        H = col.collect(X)
        assert H.shape == (len(X) - _WASHOUT, _UNITS)

    def test_zero_washout_no_trim(self):
        res = _fhn()
        X = _DATA["X_train"]
        col = StateCollector(res, washout=0)
        H = col.collect(X)
        assert H.shape[0] == len(X)

    def test_trim_targets(self):
        y = np.arange(100, dtype=float)
        col = StateCollector(_fhn(), washout=10)
        assert list(col.trim(y)) == list(y[10:])

    def test_negative_washout_raises(self):
        with pytest.raises(ValueError):
            StateCollector(_fhn(), washout=-1)


# ---------------------------------------------------------------------------
# build_one_step_targets
# ---------------------------------------------------------------------------

class TestBuildOneStepTargets:
    def test_identity(self):
        H = np.ones((10, 5))
        y = np.arange(10, dtype=float)
        H_out, y_out = build_one_step_targets(H, y)
        np.testing.assert_array_equal(H_out, H)
        np.testing.assert_array_equal(y_out, y)


# ---------------------------------------------------------------------------
# build_fixed_horizon_targets
# ---------------------------------------------------------------------------

class TestBuildFixedHorizonTargets:
    def test_horizon_1_alignment(self):
        H = np.arange(10).reshape(10, 1).astype(float)
        y = np.arange(10, dtype=float)
        H_out, y_out = build_fixed_horizon_targets(H, y, horizon=1)
        assert len(H_out) == 9
        # H[0] should predict y[1], H[1] → y[2], ...
        np.testing.assert_array_equal(H_out, H[:-1])
        np.testing.assert_array_equal(y_out, y[1:])

    def test_horizon_3_length(self):
        T = 20
        H = np.zeros((T, 4))
        y = np.zeros(T)
        H_out, y_out = build_fixed_horizon_targets(H, y, horizon=3)
        assert len(H_out) == T - 3
        assert len(y_out) == T - 3

    def test_horizon_zero_raises(self):
        with pytest.raises(ValueError):
            build_fixed_horizon_targets(np.zeros((5, 2)), np.zeros(5), horizon=0)

    def test_horizon_too_large_raises(self):
        with pytest.raises(ValueError):
            build_fixed_horizon_targets(np.zeros((5, 2)), np.zeros(5), horizon=5)


# ---------------------------------------------------------------------------
# reset_state / step / warmup — consistency with transform()
# ---------------------------------------------------------------------------

RESERVOIR_FACTORIES = [_fhn, _lsm, _logistic]  # ESN tested separately below


class TestStepAPIConsistency:
    """warmup() must produce the same H as transform() for non-ESN reservoirs."""

    @pytest.mark.parametrize("factory", RESERVOIR_FACTORIES)
    def test_warmup_matches_transform(self, factory):
        res = factory()
        X = _DATA["X_train"][:50]
        H_transform = res.transform(X)
        H_warmup = res.warmup(X)
        np.testing.assert_allclose(H_warmup, H_transform, rtol=1e-6, atol=1e-8)

    @pytest.mark.parametrize("factory", RESERVOIR_FACTORIES)
    def test_step_output_shape(self, factory):
        res = factory()
        res.reset_state()
        h = res.step(np.array([[0.5]]))
        assert h.shape == (_UNITS,)

    @pytest.mark.parametrize("factory", RESERVOIR_FACTORIES)
    def test_reset_state_is_reproducible(self, factory):
        res = factory()
        X = _DATA["X_train"][:30]
        H1 = res.warmup(X)
        H2 = res.warmup(X)   # warmup calls reset_state() internally
        np.testing.assert_array_equal(H1, H2)

    @pytest.mark.parametrize("factory", RESERVOIR_FACTORIES)
    def test_step_after_warmup_continues_state(self, factory):
        """step() after warmup(X[:T]) must equal warmup(X[:T+1])[-1]."""
        res = factory()
        X = _DATA["X_train"][:21]

        # Path A: warmup T steps, then one more step
        res.warmup(X[:20])
        h_step = res.step(X[20:21])

        # Path B: warmup T+1 steps from scratch, take last row
        h_warmup_last = res.warmup(X[:21])[-1]

        np.testing.assert_allclose(h_step, h_warmup_last, rtol=1e-6, atol=1e-8)


class TestESNStepAPI:
    def test_reset_state_no_crash(self):
        res = _esn()
        res.reset_state()   # should not raise

    def test_step_output_shape(self):
        res = _esn()
        res.reset_state()
        h = res.step(np.array([[0.5]]))
        assert h.shape == (_UNITS,)

    def test_warmup_output_shape(self):
        res = _esn()
        X = _DATA["X_train"][:30]
        H = res.warmup(X)
        assert H.shape == (30, _UNITS)


# ---------------------------------------------------------------------------
# closed_loop_predict
# ---------------------------------------------------------------------------

class TestClosedLoopPredict:
    @pytest.mark.parametrize("factory", RESERVOIR_FACTORIES)
    def test_output_shape(self, factory):
        from rc_bench.readout.ridge import RidgeReadout, select_alpha

        res = factory()
        X = _DATA["X_train"]
        y = _DATA["y_train"]

        H = res.transform(X)
        washout = _WASHOUT
        H_eff = H[washout:]
        y_eff = y[washout:]

        readout = RidgeReadout(alpha=0.1)
        readout.fit(H_eff, y_eff)

        X_seed = X[:washout]
        n_steps = 30
        y_pred = closed_loop_predict(readout, res, X_seed, n_steps)
        assert y_pred.shape == (n_steps,)

    @pytest.mark.parametrize("factory", RESERVOIR_FACTORIES)
    def test_output_is_finite(self, factory):
        from rc_bench.readout.ridge import RidgeReadout

        res = factory()
        X = _DATA["X_train"]
        y = _DATA["y_train"]
        H = res.transform(X)[_WASHOUT:]
        y_eff = y[_WASHOUT:]
        readout = RidgeReadout(0.1)
        readout.fit(H, y_eff)

        y_pred = closed_loop_predict(readout, res, X[:_WASHOUT], n_steps=20)
        assert np.all(np.isfinite(y_pred))

    def test_n_steps_zero_raises(self):
        from rc_bench.readout.ridge import RidgeReadout
        res = _fhn()
        readout = RidgeReadout(0.1)
        readout.fit(np.zeros((10, _UNITS)), np.zeros(10))
        with pytest.raises(ValueError):
            closed_loop_predict(readout, res, np.zeros((5, 1)), n_steps=0)


# ---------------------------------------------------------------------------
# run_experiment() with all three forecasting modes
# ---------------------------------------------------------------------------

class TestRunExperimentModes:
    def _run(self, mode, horizon=1):
        spec = _spec(mode, horizon)
        reservoir = get_reservoir(
            spec.reservoir.type,
            {"seed": spec.seed, **spec.reservoir.params},
        )
        return run_experiment(_DATA, spec, reservoir)

    def test_one_step_returns_metrics(self):
        result = self._run("one_step")
        assert isinstance(result["metrics"].nrmse_range, float)
        assert np.isfinite(result["metrics"].nrmse_range)

    def test_fixed_horizon_returns_metrics(self):
        result = self._run("fixed_horizon", horizon=3)
        m = result["metrics"]
        assert np.isfinite(m.nrmse_range)

    def test_fixed_horizon_preds_shorter(self):
        r_os = self._run("one_step")
        r_fh = self._run("fixed_horizon", horizon=3)
        # fixed_horizon predictions are horizon steps shorter than one_step
        assert len(r_fh["preds"]) == len(r_os["preds"]) - 3

    def test_closed_loop_returns_metrics(self):
        result = self._run("closed_loop")
        m = result["metrics"]
        assert np.isfinite(m.nrmse_range)

    def test_closed_loop_preds_shape_matches_y_test(self):
        result = self._run("closed_loop")
        assert len(result["preds"]) == len(result["y_test"])

    def test_one_step_deterministic(self):
        r1 = self._run("one_step")
        r2 = self._run("one_step")
        assert r1["metrics"].nrmse_range == r2["metrics"].nrmse_range

    def test_invalid_mode_raises(self):
        spec = _spec("one_step")
        spec = spec.model_copy(
            update={"protocol": spec.protocol.model_copy(update={"forecasting_mode": "one_step"})}
        )
        # Manually bypass pydantic to pass a bad value
        reservoir = _fhn()
        bad_spec = spec.model_copy(deep=True)
        object.__setattr__(bad_spec.protocol, "forecasting_mode", "bad_mode")
        with pytest.raises(ValueError, match="Unknown forecasting_mode"):
            run_experiment(_DATA, bad_spec, reservoir)
