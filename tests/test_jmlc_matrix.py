"""Tests for the JMLC matrix spec expansion (config -> per-cell ExperimentSpec)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rc_bench.core.data_provider import JMLC_RAW_DATA_ENV
from rc_bench.runners.jmlc_matrix import (
    CANONICAL_CELLS,
    HORIZONS,
    build_cell_spec,
    load_template,
    verify_pinned_raw_digest,
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


def test_matrix_config_pins_the_manifest_raw_sha256():
    """Every RunRecord must be traceable to the exact raw bytes it was run on.

    The spec's ``raw_sha256`` is the only link from a result back to the pinned
    dataset manifest, so a config whose digest drifts from the manifest would
    silently publish results attributed to the wrong bytes.
    """
    jmlc = Path(__file__).resolve().parents[1] / "configs" / "jmlc"
    manifest = json.loads((jmlc / "dataset_manifest.json").read_text())
    expected = manifest["raw_file"]["sha256"]

    for config in sorted(jmlc.glob("*.yaml")):
        template = load_template(config)
        assert template["dataset"].get("raw_sha256") == expected, config.name


class TestPinnedRawDigestVerification:
    """A pinned digest is only evidence if the run checks the bytes it read."""

    @staticmethod
    def _template(digest: str) -> dict:
        return {**_TEMPLATE, "dataset": {**_TEMPLATE["dataset"], "raw_sha256": digest}}

    @staticmethod
    def _raw_file(tmp_path: Path, monkeypatch, payload: bytes = b"raw bytes") -> str:
        path = tmp_path / "household_power_consumption.txt"
        path.write_bytes(payload)
        monkeypatch.setenv(JMLC_RAW_DATA_ENV, str(path))
        return hashlib.sha256(payload).hexdigest()

    def test_matching_digest_passes(self, tmp_path, monkeypatch):
        digest = self._raw_file(tmp_path, monkeypatch)
        verify_pinned_raw_digest(self._template(digest))  # must not raise

    def test_mismatched_digest_is_refused(self, tmp_path, monkeypatch):
        self._raw_file(tmp_path, monkeypatch)
        with pytest.raises(ValueError, match="raw_sha256"):
            verify_pinned_raw_digest(self._template("b" * 64))

    def test_unpinned_template_is_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setenv(JMLC_RAW_DATA_ENV, str(tmp_path / "missing.txt"))
        verify_pinned_raw_digest(_TEMPLATE)  # nothing pinned, nothing to check

    def test_missing_raw_file_is_reported(self, tmp_path, monkeypatch):
        monkeypatch.setenv(JMLC_RAW_DATA_ENV, str(tmp_path / "missing.txt"))
        with pytest.raises(ValueError, match="missing.txt"):
            verify_pinned_raw_digest(self._template("c" * 64))


def test_cell_specs_inherit_the_pinned_raw_sha256():
    digest = "a" * 64
    template = {**_TEMPLATE, "dataset": {**_TEMPLATE["dataset"], "raw_sha256": digest}}

    for family, model in CANONICAL_CELLS:
        for horizon in HORIZONS:
            spec = build_cell_spec(template, family, model, horizon)
            assert spec.dataset.raw_sha256 == digest
