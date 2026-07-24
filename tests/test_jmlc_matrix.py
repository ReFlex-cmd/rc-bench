"""Tests for the JMLC matrix spec expansion (config -> per-cell ExperimentSpec)."""

from __future__ import annotations

from pathlib import Path

from rc_bench.runners.jmlc_matrix import (
    CANONICAL_CELLS,
    HORIZONS,
    build_cell_spec,
    load_template,
)

_TEMPLATE = {
    "dataset": {"name": "uci_household_power", "length": 12_000},
    "reservoir": {"type": "esn", "params": {}},
    "protocol": {
        "washout": 200,
        "train_frac": 0.6,
        "val_frac": 0.2,
        "forecasting_mode": "fixed_horizon",
        "horizon": 1,
        "use_hpo": True,
        "hpo_budget": 2,
        "n_seeds": 1,
        "selection_metric": "nrmse_std",
        "seasonal_period": 24,
    },
    "readout": {"alpha_grid": [0.001, 0.01, 0.1, 1.0, 10.0]},
    "seed": 42,
}


def test_canonical_matrix_is_14_cells():
    assert len(CANONICAL_CELLS) * len(HORIZONS) == 14


def test_baseline_cell_is_deterministic_and_seedless():
    spec = build_cell_spec(_TEMPLATE, "baseline", "persistence", 24)
    assert spec.model_family == "baseline"
    assert spec.model_type == "persistence"
    assert spec.seed is None
    assert spec.protocol.n_seeds == 0
    assert spec.protocol.use_hpo is False
    assert spec.protocol.horizon == 24
    assert spec.protocol.seasonal_period == 24
    assert spec.protocol.forecasting_mode == "fixed_horizon"


def test_reservoir_cell_keeps_seed_and_hpo():
    spec = build_cell_spec(_TEMPLATE, "reservoir", "lsm", 1)
    assert spec.model_family == "reservoir"
    assert spec.model_type == "lsm"
    assert spec.seed == 42
    assert spec.protocol.n_seeds == 1
    assert spec.protocol.use_hpo is True
    assert spec.protocol.selection_metric == "nrmse_std"


def test_every_canonical_cell_builds_a_valid_spec():
    for family, model in CANONICAL_CELLS:
        for horizon in HORIZONS:
            spec = build_cell_spec(_TEMPLATE, family, model, horizon)
            assert spec.protocol.forecasting_mode == "fixed_horizon"
            assert spec.protocol.horizon == horizon


def test_smoke_config_is_a_valid_template():
    config = Path(__file__).resolve().parents[1] / "configs" / "jmlc" / "smoke.yaml"
    template = load_template(config)  # raises if the template is not a valid spec
    assert template["dataset"]["name"] == "uci_household_power"
