import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from rc_bench.core.data_provider import get_data_for_experiment
from rc_bench.core.schema import (
    DatasetSpec, ExperimentSpec, ProtocolSpec, ReadoutSpec, ReservoirSpec,
)
from rc_bench.core.reservoirs.esn_service import ESNReservoir
from rc_bench.core.reservoirs.lsm_service import LSMReservoir
from rc_bench.core.reservoirs.fhn_service import FHNReservoir
from rc_bench.core.reservoirs.logistic_service import LogisticReservoir
from rc_bench.core.reservoirs.leaky_esn_service import LeakyESNReservoir
from rc_bench.core.reservoirs.deep_esn_service import DeepESNReservoir
from rc_bench.core.reservoirs.qrc_service import QRCReservoir
from rc_bench.core.reservoirs.registry import get_reservoir, REGISTRY
from rc_bench.runners.experiment_runner import run_experiment

_DATA = get_data_for_experiment("narma10", length=600, seed=42)
_EXPECTED_KEYS = {"metrics", "best_alpha", "preds", "y_test", "reservoir_states_std"}

_SPECS = {
    "esn": ExperimentSpec(
        dataset=DatasetSpec(name="narma10"),
        reservoir=ReservoirSpec(type="esn", params={"n_units": 50}),
        protocol=ProtocolSpec(washout=50),
        seed=42,
    ),
    "lsm": ExperimentSpec(
        dataset=DatasetSpec(name="narma10"),
        reservoir=ReservoirSpec(type="lsm", params={"units": 50, "density": 0.1, "dt": 1.0, "input_scale": 2.0, "tau_mem": 5.0}),
        protocol=ProtocolSpec(washout=50),
        seed=42,
    ),
    "fhn": ExperimentSpec(
        dataset=DatasetSpec(name="narma10"),
        reservoir=ReservoirSpec(type="fhn", params={"units": 30, "density": 0.1, "dt": 0.1, "internal_steps": 1}),
        protocol=ProtocolSpec(washout=50),
        seed=42,
    ),
    "logistic": ExperimentSpec(
        dataset=DatasetSpec(name="narma10"),
        reservoir=ReservoirSpec(type="logistic", params={"units": 50}),
        protocol=ProtocolSpec(washout=50),
        seed=42,
    ),
    "leaky_esn": ExperimentSpec(
        dataset=DatasetSpec(name="narma10"),
        reservoir=ReservoirSpec(type="leaky_esn", params={"units": 50, "sr": 0.9, "leak_rate": 0.3}),
        protocol=ProtocolSpec(washout=50),
        seed=42,
    ),
    "deep_esn": ExperimentSpec(
        dataset=DatasetSpec(name="narma10"),
        reservoir=ReservoirSpec(type="deep_esn", params={"n_layers": 2, "units": 30}),
        protocol=ProtocolSpec(washout=50),
        seed=42,
    ),
    "qrc": ExperimentSpec(
        dataset=DatasetSpec(name="narma10"),
        reservoir=ReservoirSpec(type="qrc", params={"n_qubits": 16, "depth": 2}),
        protocol=ProtocolSpec(washout=50),
        seed=42,
    ),
}


def _reservoir_config(spec: ExperimentSpec) -> dict:
    return {"seed": spec.seed, **spec.reservoir.params}


class _ReservoirTestBase:
    rtype: str

    @property
    def spec(self) -> ExperimentSpec:
        return _SPECS[self.rtype]

    def _run(self):
        reservoir = get_reservoir(self.rtype, _reservoir_config(self.spec))
        return run_experiment(_DATA, self.spec, reservoir)

    def test_output_keys(self):
        result = self._run()
        assert set(result.keys()) == _EXPECTED_KEYS

    def test_metrics_typed(self):
        from rc_bench.core.schema import MetricsResult
        result = self._run()
        assert isinstance(result["metrics"], MetricsResult)

    def test_deterministic(self):
        r1 = self._run()
        r2 = self._run()
        assert r1["metrics"].nrmse_range == r2["metrics"].nrmse_range

    def test_nrmse_range_is_finite_float(self):
        val = self._run()["metrics"].nrmse_range
        assert isinstance(val, float)
        assert np.isfinite(val)

    def test_sanity_checks_pass(self):
        spec = self.spec
        reservoir = get_reservoir(self.rtype, _reservoir_config(spec))
        X = _DATA["X_train"]
        if reservoir.DEFAULT_SCALER == "zscore":
            X = StandardScaler().fit_transform(X)
        H = reservoir.transform(X)
        for name, passed in reservoir.sanity_check(H).items():
            assert passed, f"Sanity check failed: {name}"


class TestESN(_ReservoirTestBase):
    rtype = "esn"


class TestLSM(_ReservoirTestBase):
    rtype = "lsm"


class TestFHN(_ReservoirTestBase):
    rtype = "fhn"


class TestLogistic(_ReservoirTestBase):
    rtype = "logistic"


class TestLeakyESN(_ReservoirTestBase):
    rtype = "leaky_esn"


class TestDeepESN(_ReservoirTestBase):
    rtype = "deep_esn"


class TestQRC(_ReservoirTestBase):
    rtype = "qrc"


class TestRegistry:
    _ALL_TYPES = {"esn", "lsm", "fhn", "logistic", "leaky_esn", "deep_esn", "qrc"}

    def test_all_types_registered(self):
        assert set(REGISTRY) == self._ALL_TYPES

    def test_get_reservoir_returns_correct_class(self):
        for rtype, cls in [
            ("esn", ESNReservoir),
            ("lsm", LSMReservoir),
            ("fhn", FHNReservoir),
            ("logistic", LogisticReservoir),
            ("leaky_esn", LeakyESNReservoir),
            ("deep_esn", DeepESNReservoir),
            ("qrc", QRCReservoir),
        ]:
            assert isinstance(get_reservoir(rtype, _reservoir_config(_SPECS[rtype])), cls)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown reservoir type"):
            get_reservoir("unknown_type", {})

    def test_case_insensitive(self):
        assert isinstance(get_reservoir("ESN", _reservoir_config(_SPECS["esn"])), ESNReservoir)


class TestExperimentSpec:
    def test_config_hash_deterministic(self):
        spec = _SPECS["esn"]
        assert spec.config_hash() == spec.config_hash()

    def test_config_hash_differs_for_different_specs(self):
        assert _SPECS["esn"].config_hash() != _SPECS["lsm"].config_hash()

    def test_reservoir_type_validation(self):
        with pytest.raises(Exception):
            ReservoirSpec(type="unknown_type", params={})
