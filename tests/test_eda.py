from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rc_bench.data.jmlc import HourlyPowerSeries
from rc_bench.reporting.eda import (
    ACF_MAX_LAG_HOURS,
    EDA_ARTIFACT_PATHS,
    EDAInvariantError,
    analyze_eda,
    generate_eda_report,
)


RAW_SHA256 = "a" * 64


def _series(
    *,
    length: int = 336,
    values: np.ndarray | None = None,
    imputed_indices: tuple[int, ...] = (10, 11, 80, 150, 250),
) -> HourlyPowerSeries:
    timestamps = np.arange(
        np.datetime64("2026-01-05T00", "h"),
        np.datetime64("2026-01-05T00", "h") + np.timedelta64(length, "h"),
        dtype="datetime64[h]",
    )
    if values is None:
        steps = np.arange(length)
        values = 2.0 + np.sin(2.0 * np.pi * steps / 24.0)
    values = np.asarray(values, dtype=np.float64).copy()
    observed = np.ones(length, dtype=np.bool_)
    minute_counts = np.full(length, 60, dtype=np.int64)
    low_counts = (0, 10, 29, 0, 5)
    for index, count in zip(imputed_indices, low_counts, strict=False):
        if index >= length:
            continue
        observed[index] = False
        minute_counts[index] = count
        values[index] = values[index - 1]

    return HourlyPowerSeries(
        timestamps=timestamps,
        values=values,
        observed_mask=observed,
        imputed_mask=~observed,
        valid_minute_counts=minute_counts,
        source_rows=length * 60 + 3,
        duplicate_rows_removed=3,
        missing_target_rows=7,
        source_hour_grid_size=length + 4,
        leading_hours_dropped=2,
        trailing_hours_unused=2,
    )


def test_range_coverage_and_missingness_have_explicit_scopes() -> None:
    series = _series()

    summary = analyze_eda(series, raw_sha256=RAW_SHA256)

    coverage = summary["coverage"]
    assert coverage["window_start"] == "2026-01-05T00"
    assert coverage["window_end"] == "2026-01-18T23"
    assert coverage["source_grid_start"] == "2026-01-04T22"
    assert coverage["source_grid_end"] == "2026-01-19T01"
    assert coverage["window_hours"] == 336

    minute = summary["missingness"]["minute_level_before_resampling"]
    assert minute["scope"] == "source-present rows after exact deduplication"
    assert minute["present_rows"] == 336 * 60
    assert minute["missing_target_rows"] == 7

    selected = summary["missingness"]["selected_window_minute_slots"]
    assert selected["scope"] == "selected window; absent and invalid slots combined"
    assert selected["total_slots"] == 336 * 60
    assert selected["valid_slots"] == int(series.valid_minute_counts.sum())
    assert selected["absent_or_invalid_slots"] == (
        336 * 60 - int(series.valid_minute_counts.sum())
    )

    hourly = summary["missingness"]["hour_level_after_resampling"]
    assert hourly["observed_hours"] == int(series.observed_mask.sum())
    assert hourly["imputed_hours"] == int(series.imputed_mask.sum())
    assert hourly["minimum_valid_minutes"] == 30


def test_distribution_and_outliers_use_observed_hours_only() -> None:
    values = np.linspace(1.0, 3.0, 336)
    values[-1] = 100.0
    series = _series(values=values)

    summary = analyze_eda(series, raw_sha256=RAW_SHA256)
    distribution = summary["target_distribution"]

    observed_values = series.values[series.observed_mask]
    assert distribution["scope"] == "observed hourly targets only"
    assert distribution["statistics"]["count"] == observed_values.size
    assert distribution["statistics"]["mean"] == pytest.approx(
        float(observed_values.mean())
    )
    assert distribution["outliers"]["method"] == "Tukey 1.5 IQR"
    assert distribution["outliers"]["count"] >= 1


def test_daily_and_weekly_profiles_are_calendar_aligned() -> None:
    length = 14 * 24
    hours = np.arange(length) % 24
    weekdays = (np.arange(length) // 24) % 7
    values = hours.astype(float) + 10.0 * weekdays
    series = _series(length=length, values=values, imputed_indices=())

    summary = analyze_eda(series, raw_sha256=RAW_SHA256)
    daily = summary["profiles"]["hour_of_day"]
    weekly = summary["profiles"]["day_of_week"]

    assert [row["hour"] for row in daily] == list(range(24))
    assert daily[5]["count"] == 14
    assert daily[5]["mean"] == pytest.approx(35.0)
    assert [row["day"] for row in weekly] == [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    assert weekly[0]["count"] == 48
    assert weekly[0]["mean"] == pytest.approx(11.5)


def test_pairwise_observed_acf_is_bounded_and_detects_daily_periodicity() -> None:
    series = _series(imputed_indices=())

    summary = analyze_eda(series, raw_sha256=RAW_SHA256)
    acf = summary["acf"]

    assert ACF_MAX_LAG_HOURS == 168
    assert acf["method"] == "pairwise-observed Pearson correlation"
    assert acf["lags_hours"] == list(range(169))
    assert len(acf["correlations"]) == 169
    assert len(acf["pair_counts"]) == 169
    assert acf["correlations"][0] == pytest.approx(1.0)
    assert acf["correlations"][24] > 0.99


def test_splits_and_leakage_checks_cover_both_contract_horizons() -> None:
    series = _series()

    summary = analyze_eda(series, raw_sha256=RAW_SHA256)

    assert [summary["splits"][name]["hours"] for name in ("train", "val", "test")] == [
        201,
        67,
        68,
    ]
    leakage = summary["leakage"]
    assert leakage["all_passed"] is True
    assert all(leakage["checks"].values())
    assert leakage["forecast_horizons"]["1"]["no_cross_boundary"] is True
    assert leakage["forecast_horizons"]["24"]["no_cross_boundary"] is True
    assert leakage["protocol"]["shuffle_used"] is False
    assert leakage["protocol"]["scaler_fit_performed"] is False
    assert leakage["protocol"]["statistics_scope"] == "observed targets only"


def test_noncausal_imputation_fails_before_any_output(tmp_path: Path) -> None:
    causal = _series()
    values = causal.values.copy()
    values[10] += 5.0
    output_dir = tmp_path / "report"
    noncausal = HourlyPowerSeries(
        timestamps=causal.timestamps,
        values=values,
        observed_mask=causal.observed_mask,
        imputed_mask=causal.imputed_mask,
        valid_minute_counts=causal.valid_minute_counts,
        source_rows=causal.source_rows,
        duplicate_rows_removed=causal.duplicate_rows_removed,
        missing_target_rows=causal.missing_target_rows,
        source_hour_grid_size=causal.source_hour_grid_size,
        leading_hours_dropped=causal.leading_hours_dropped,
        trailing_hours_unused=causal.trailing_hours_unused,
    )

    with pytest.raises(EDAInvariantError, match="causal forward-fill"):
        generate_eda_report(noncausal, output_dir, raw_sha256=RAW_SHA256)

    assert not output_dir.exists()


def test_generator_writes_exact_deterministic_sanitized_artifacts(
    tmp_path: Path,
) -> None:
    series = _series()
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first_paths = generate_eda_report(
        series,
        first_dir,
        raw_sha256=RAW_SHA256,
    )
    second_paths = generate_eda_report(
        series,
        second_dir,
        raw_sha256=RAW_SHA256,
    )

    assert {path.relative_to(first_dir).as_posix() for path in first_paths} == set(
        EDA_ARTIFACT_PATHS
    )
    assert {path.relative_to(second_dir).as_posix() for path in second_paths} == set(
        EDA_ARTIFACT_PATHS
    )
    assert all(path.is_file() and path.stat().st_size > 0 for path in first_paths)
    for relative_path in EDA_ARTIFACT_PATHS:
        assert (first_dir / relative_path).read_bytes() == (
            second_dir / relative_path
        ).read_bytes()

    first_json = (first_dir / "aggregates/eda_summary.json").read_text()
    second_json = (second_dir / "aggregates/eda_summary.json").read_text()
    first_md = (first_dir / "eda_report.md").read_text()
    second_md = (second_dir / "eda_report.md").read_text()
    assert first_json == second_json
    assert first_md == second_md
    assert json.loads(first_json)["dataset"]["raw_sha256"] == RAW_SHA256
    assert "NaN" not in first_json
    assert "Infinity" not in first_json
    assert str(tmp_path) not in first_json + first_md
    assert "household_power_consumption.txt" not in first_json + first_md
    assert "plots/eda_acf_168h.png" in first_md
