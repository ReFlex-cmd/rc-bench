"""
Tests for Stage 8: HPO (search_spaces + tuner) and multi-seed evaluation.
HPO tests use n_trials=3 to stay fast; correctness over speed.
"""

import numpy as np
import pytest

from rc_bench.core.data_provider import get_data_for_experiment
from rc_bench.core.schema import (
    DatasetSpec, ExperimentSpec, MetricsResult, MetricsSummary,
    MultiSeedResult, ProtocolSpec, ReadoutSpec, ReservoirSpec,
)
from rc_bench.hpo.search_spaces import (
    SEARCH_SPACES, params_from_trial, suggest_params,
)
from rc_bench.hpo.tuner import apply_hpo_params, run_hpo
from rc_bench.runners.multi_seed import run_multi_seed
from rc_bench.runners.pipeline import run_pipeline

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_T = 300
_DATA = get_data_for_experiment("narma10", length=_T, seed=42)

_BASE_SPEC = ExperimentSpec(
    dataset=DatasetSpec(name="narma10", length=_T),
    # ESN: numerically robust enough that HPO with n_trials=3 reliably
    # produces at least one completed trial. FHN/LSM trial sweeps can NaN-out
    # at small budgets and leave the study with no best_trial.
    reservoir=ReservoirSpec(type="esn", params={"n_units": 30}),
    # Explicit n_seeds=1 so default tests exercise the single-run code path.
    # Multi-seed tests override n_seeds via _spec(n_seeds=...).
    protocol=ProtocolSpec(washout=20, train_frac=0.6, val_frac=0.2, n_seeds=1),
    readout=ReadoutSpec(alpha_grid=[0.01, 0.1, 1.0]),
    seed=42,
)


# ---------------------------------------------------------------------------
# TestSearchSpaces
# ---------------------------------------------------------------------------

class TestSearchSpaces:
    _ALL_RTYPES = {"esn", "fhn", "lsm", "logistic", "leaky_esn", "deep_esn", "qrc"}

    def test_all_reservoir_types_covered(self):
        assert set(SEARCH_SPACES) == self._ALL_RTYPES

    def test_readout_alpha_in_every_space(self):
        for rtype, space in SEARCH_SPACES.items():
            assert "readout_alpha" in space, f"readout_alpha missing for {rtype}"

    @pytest.mark.parametrize("rtype", ["esn", "fhn", "lsm", "logistic", "leaky_esn", "deep_esn", "qrc"])
    def test_suggest_params_keys(self, rtype):
        import optuna
        study = optuna.create_study()
        trial = study.ask()
        params = suggest_params(trial, rtype)
        assert "reservoir_params" in params
        assert "readout_alpha" in params

    @pytest.mark.parametrize("rtype", ["esn", "fhn", "lsm", "logistic", "leaky_esn", "deep_esn", "qrc"])
    def test_suggest_params_values_in_range(self, rtype):
        import optuna
        study = optuna.create_study()
        trial = study.ask()
        params = suggest_params(trial, rtype)
        assert params["readout_alpha"] > 0

    def test_logistic_r_max_ge_r_min(self):
        """r_max must never be less than r_min after suggestion."""
        import optuna
        for _ in range(10):
            study = optuna.create_study()
            trial = study.ask()
            params = suggest_params(trial, "logistic")
            rp = params["reservoir_params"]
            assert rp.get("r_max", 4.0) >= rp.get("r_min", 3.5)

    def test_params_from_trial_roundtrip(self):
        import optuna
        study = optuna.create_study()
        trial = study.ask()
        params = suggest_params(trial, "fhn")
        flat = {**params["reservoir_params"], "readout_alpha": params["readout_alpha"]}
        reconstructed = params_from_trial(flat, "fhn")
        assert reconstructed["readout_alpha"] == params["readout_alpha"]
        assert reconstructed["reservoir_params"] == params["reservoir_params"]


# ---------------------------------------------------------------------------
# TestRunHPO
# ---------------------------------------------------------------------------

class TestRunHPO:
    def test_returns_hpo_result(self):
        result = run_hpo(_BASE_SPEC, _DATA, n_trials=3, seed=0)
        assert isinstance(result.best_params, dict)
        assert isinstance(result.best_score, float)

    def test_best_score_finite_and_positive(self):
        result = run_hpo(_BASE_SPEC, _DATA, n_trials=3, seed=0)
        assert np.isfinite(result.best_score)
        assert result.best_score >= 0.0

    def test_best_params_structure(self):
        result = run_hpo(_BASE_SPEC, _DATA, n_trials=3, seed=0)
        assert "reservoir_params" in result.best_params
        assert "readout_alpha" in result.best_params
        assert result.best_params["readout_alpha"] > 0

    def test_deterministic_with_same_seed(self):
        r1 = run_hpo(_BASE_SPEC, _DATA, n_trials=3, seed=7)
        r2 = run_hpo(_BASE_SPEC, _DATA, n_trials=3, seed=7)
        assert r1.best_score == r2.best_score

    def test_different_seeds_may_differ(self):
        r1 = run_hpo(_BASE_SPEC, _DATA, n_trials=6, seed=1)
        r2 = run_hpo(_BASE_SPEC, _DATA, n_trials=6, seed=99)
        assert np.isfinite(r1.best_score) and np.isfinite(r2.best_score)

    def test_convergence_history_recorded(self):
        result = run_hpo(_BASE_SPEC, _DATA, n_trials=4, seed=0)
        # Convergence is best-so-far per completed trial; length ≤ n_trials
        assert isinstance(result.convergence, list)
        assert len(result.convergence) <= 4
        # Best-so-far must be monotonically non-increasing
        for prev, curr in zip(result.convergence, result.convergence[1:]):
            assert curr <= prev + 1e-12

    def test_diagnostics_counters(self):
        result = run_hpo(_BASE_SPEC, _DATA, n_trials=4, seed=0)
        d = result.diagnostics
        assert d["n_trials"] == 4
        assert d["n_completed"] + d["n_pruned"] + d["n_failed"] <= 4
        assert d["best_trial_number"] >= 0

    def test_apply_hpo_params_updates_spec(self):
        result = run_hpo(_BASE_SPEC, _DATA, n_trials=3, seed=0)
        new_spec = apply_hpo_params(_BASE_SPEC, result.best_params)
        assert len(new_spec.readout.alpha_grid) == 1
        assert new_spec.readout.alpha_grid[0] == result.best_params["readout_alpha"]

    def test_apply_hpo_params_does_not_mutate_original(self):
        result = run_hpo(_BASE_SPEC, _DATA, n_trials=3, seed=0)
        original_alpha_grid = list(_BASE_SPEC.readout.alpha_grid)
        apply_hpo_params(_BASE_SPEC, result.best_params)
        assert list(_BASE_SPEC.readout.alpha_grid) == original_alpha_grid


# ---------------------------------------------------------------------------
# TestMetricsSummary
# ---------------------------------------------------------------------------

def _make_metrics(nrmse_range: float, seed: int = 0) -> MetricsResult:
    return MetricsResult(
        rmse=nrmse_range * 2,
        nrmse_range=nrmse_range,
        nrmse_std=nrmse_range * 1.1,
        nrmse_var=nrmse_range ** 2,
        mae=nrmse_range * 0.9,
        mse=nrmse_range ** 2 * 4,
        prediction_horizon=int(10 / (nrmse_range + 0.1)),
        val_nrmse_range=nrmse_range * 1.05,
        train_time=0.1,
        inference_latency=0.01,
        peak_memory=1024,
    )


class TestMetricsSummary:
    def test_mean_of_identical_metrics(self):
        m = _make_metrics(0.5)
        summary = MetricsSummary.from_metrics_list([m, m, m], lambda xs: float(np.mean(xs)))
        assert summary.nrmse_range == pytest.approx(0.5)

    def test_std_of_identical_metrics_is_zero(self):
        m = _make_metrics(0.5)
        summary = MetricsSummary.from_metrics_list([m, m, m], lambda xs: float(np.std(xs)))
        assert summary.nrmse_range == pytest.approx(0.0)

    def test_mean_of_two_metrics(self):
        m1 = _make_metrics(0.2)
        m2 = _make_metrics(0.4)
        summary = MetricsSummary.from_metrics_list([m1, m2], lambda xs: float(np.mean(xs)))
        assert summary.nrmse_range == pytest.approx(0.3)

    def test_all_fields_float(self):
        m = _make_metrics(0.3)
        summary = MetricsSummary.from_metrics_list([m], lambda xs: float(np.mean(xs)))
        for field in MetricsSummary.model_fields:
            assert isinstance(getattr(summary, field), float)


# ---------------------------------------------------------------------------
# TestRunMultiSeed
# ---------------------------------------------------------------------------

class TestRunMultiSeed:
    def test_correct_n_seeds(self):
        result = run_multi_seed(_DATA, _BASE_SPEC, n_seeds=3)
        assert result.n_seeds == 3
        assert len(result.seeds) == 3
        assert len(result.metrics_per_seed) == 3

    def test_seeds_are_offset_from_base(self):
        result = run_multi_seed(_DATA, _BASE_SPEC, n_seeds=4)
        assert result.seeds == [_BASE_SPEC.seed + i for i in range(4)]

    def test_mean_nrmse_range_is_finite(self):
        result = run_multi_seed(_DATA, _BASE_SPEC, n_seeds=3)
        assert np.isfinite(result.mean.nrmse_range)

    def test_std_is_non_negative(self):
        result = run_multi_seed(_DATA, _BASE_SPEC, n_seeds=3)
        assert result.std.nrmse_range >= 0.0

    def test_std_zero_for_single_seed(self):
        result = run_multi_seed(_DATA, _BASE_SPEC, n_seeds=1)
        assert result.std.nrmse_range == pytest.approx(0.0)

    def test_deterministic(self):
        r1 = run_multi_seed(_DATA, _BASE_SPEC, n_seeds=2)
        r2 = run_multi_seed(_DATA, _BASE_SPEC, n_seeds=2)
        assert r1.mean.nrmse_range == r2.mean.nrmse_range

    def test_returns_multi_seed_result_type(self):
        result = run_multi_seed(_DATA, _BASE_SPEC, n_seeds=2)
        assert isinstance(result, MultiSeedResult)


# ---------------------------------------------------------------------------
# TestPipeline
# ---------------------------------------------------------------------------

class TestPipeline:
    def _spec(self, **protocol_kwargs) -> ExperimentSpec:
        return _BASE_SPEC.model_copy(
            update={"protocol": _BASE_SPEC.protocol.model_copy(update=protocol_kwargs)}
        )

    def test_single_run_populates_metrics(self):
        result = run_pipeline(_DATA, self._spec())
        assert result.status == "completed"
        assert result.metrics is not None
        assert result.multi_seed_result is None

    def test_multi_seed_run_populates_multi_seed_result(self):
        result = run_pipeline(_DATA, self._spec(n_seeds=3))
        assert result.multi_seed_result is not None
        assert result.multi_seed_result.n_seeds == 3
        assert result.metrics is None

    def test_hpo_run_stores_best_params(self):
        result = run_pipeline(_DATA, self._spec(use_hpo=True, hpo_budget=3))
        assert result.hpo_best_params is not None
        assert "readout_alpha" in result.hpo_best_params

    def test_hpo_plus_multi_seed(self):
        result = run_pipeline(_DATA, self._spec(use_hpo=True, hpo_budget=3, n_seeds=2))
        assert result.hpo_best_params is not None
        assert result.multi_seed_result is not None
        assert result.multi_seed_result.n_seeds == 2

    def test_result_config_hash_matches_spec(self):
        spec = self._spec()
        result = run_pipeline(_DATA, spec)
        assert result.frozen_config_hash == spec.config_hash()
        assert result.resolved_spec == spec
        assert result.config_hash == result.resolved_spec.config_hash()

    def test_hpo_result_keeps_frozen_and_resolved_specs_distinct(self):
        spec = self._spec(use_hpo=True, hpo_budget=3)
        frozen_dump = spec.model_dump()

        result = run_pipeline(_DATA, spec)

        assert spec.model_dump() == frozen_dump
        assert result.frozen_config_hash == spec.config_hash()
        assert result.resolved_spec is not None
        assert result.resolved_spec != spec
        assert result.config_hash == result.resolved_spec.config_hash()
        assert result.resolved_spec.readout.alpha_grid == [
            result.hpo_best_params["readout_alpha"]
        ]
