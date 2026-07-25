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
    resolve_budget,
    resolve_mode,
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


def test_template_without_a_matrix_block_is_fair():
    """Существующие конфиги режима не объявляют — они обязаны остаться fair."""
    assert resolve_mode(_TEMPLATE) == "fair"
    assert build_cell_spec(_TEMPLATE, "reservoir", "esn", 1).protocol.mode == "fair"


def test_best_effort_template_gives_each_model_its_own_budget():
    template = {
        **_TEMPLATE,
        "matrix": {"mode": "best_effort", "budgets": {"esn": 60, "lsm": 40}},
    }

    assert resolve_mode(template) == "best_effort"
    assert resolve_budget(template, "esn") == 60
    assert resolve_budget(template, "lsm") == 40
    # Модель без индивидуальной строки наследует бюджет шаблона.
    assert resolve_budget(template, "logistic") == _TEMPLATE["protocol"]["hpo_budget"]

    spec = build_cell_spec(template, "reservoir", "lsm", 24)
    assert spec.protocol.mode == "best_effort"
    assert spec.protocol.hpo_budget == 40


def test_fair_template_rejects_per_model_budgets():
    """В fair-режиме индивидуальный бюджет — нарушение DEC-004, а не опция:
    такая конфигурация означает, что автор хотел best_effort и забыл
    переключить режим. Молча выполнить её — значит опубликовать неравное
    сравнение под вывеской равного."""
    template = {**_TEMPLATE, "matrix": {"mode": "fair", "budgets": {"esn": 60}}}

    with pytest.raises(ValueError, match="best_effort"):
        resolve_budget(template, "esn")


def test_unknown_mode_is_refused():
    template = {**_TEMPLATE, "matrix": {"mode": "generous"}}

    with pytest.raises(ValueError, match="generous"):
        resolve_mode(template)


def test_mode_declared_in_protocol_alone_is_honoured():
    """`rcbench validate-spec` видит только protocol, поэтому шаблон вправе
    объявить режим там; раннер обязан его прочитать."""
    template = {
        **_TEMPLATE,
        "protocol": {**_TEMPLATE["protocol"], "mode": "best_effort"},
    }

    assert resolve_mode(template) == "best_effort"


def test_disagreeing_mode_declarations_are_refused():
    """Если matrix и protocol расходятся, один из двух отчётов о шаблоне
    врёт — валидатор покажет одно, раннер выполнит другое."""
    template = {
        **_TEMPLATE,
        "matrix": {"mode": "best_effort"},
        "protocol": {**_TEMPLATE["protocol"], "mode": "fair"},
    }

    with pytest.raises(ValueError, match="make them agree"):
        resolve_mode(template)


def test_best_effort_config_declares_the_mode_consistently():
    """Опубликованный конфиг должен проходить обе проверки: и раннера, и
    валидатора спецификации."""
    config = Path(__file__).resolve().parents[1] / "configs" / "jmlc" / "best_effort.yaml"
    template = load_template(config)

    assert resolve_mode(template) == "best_effort"
    assert template["protocol"]["mode"] == "best_effort"
    assert build_cell_spec(template, "reservoir", "lsm", 24).protocol.hpo_budget == 40


def test_baseline_cells_record_the_mode_but_ignore_budgets():
    template = {
        **_TEMPLATE,
        "matrix": {"mode": "best_effort", "budgets": {"esn": 60}},
    }
    spec = build_cell_spec(template, "baseline", "persistence", 1)

    assert spec.protocol.mode == "best_effort"
    assert spec.protocol.use_hpo is False
    assert spec.protocol.n_seeds == 0


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
