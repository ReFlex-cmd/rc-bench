from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from rc_bench.data.jmlc import (
    DEFAULT_WINDOW_HOURS,
    ConflictingDuplicateError,
    HourlyPowerSeries,
    InsufficientDataError,
    NonMonotonicTimeError,
    build_forecast_splits,
    load_uci_household_power_series,
    split_hourly_series,
)


UCI_HEADER = (
    "Date;Time;Global_active_power;Global_reactive_power;Voltage;"
    "Global_intensity;Sub_metering_1;Sub_metering_2;Sub_metering_3"
)


def _uci_row(timestamp: datetime, target: float | str) -> str:
    target_text = target if isinstance(target, str) else f"{target:.6f}"
    return (
        f"{timestamp:%d/%m/%Y};{timestamp:%H:%M:%S};{target_text};"
        "0.000;230.000;0.000;0.000;0.000;0.000"
    )


def _hour_rows(
    hour: datetime,
    values: list[float],
    *,
    missing_minutes: tuple[int, ...] = (),
) -> list[str]:
    rows = [
        _uci_row(hour + timedelta(minutes=minute), value)
        for minute, value in enumerate(values)
    ]
    rows.extend(
        _uci_row(hour + timedelta(minutes=minute), "?")
        for minute in missing_minutes
    )
    return rows


def _write_causal_fixture(path: Path) -> None:
    start = datetime(2006, 12, 16)
    rows: list[str] = []

    # Leading hour has no causal predecessor and insufficient coverage: drop it.
    rows.extend(_hour_rows(start, [999.0] * 29))
    # First observed hour: mean(1..30) = 15.5.
    rows.extend(_hour_rows(start + timedelta(hours=1), list(range(1, 31))))
    # Hour 2 is entirely absent from the source and must appear on the regular grid.
    # Hour 3 has 29 valid values plus "?", so its partial mean must not be used.
    rows.extend(
        _hour_rows(
            start + timedelta(hours=3),
            [300.0] * 29,
            missing_minutes=(29,),
        )
    )
    observed_hour = _hour_rows(start + timedelta(hours=4), [4.0] * 30)
    rows.extend(observed_hour)
    rows.append(observed_hour[7])  # exact duplicate, intentionally non-adjacent
    rows.extend(
        _hour_rows(
            start + timedelta(hours=5),
            [5.0] * 30,
            missing_minutes=(30,),
        )
    )
    # A valid trailing hour proves that the first deterministic window is selected.
    rows.extend(_hour_rows(start + timedelta(hours=6), [6.0] * 30))

    path.write_text(
        "\n".join([UCI_HEADER, *reversed(rows), ""]),
        encoding="utf-8",
    )


def _synthetic_hourly_series(length: int = 50) -> HourlyPowerSeries:
    start = np.datetime64("2026-01-01T00", "h")
    timestamps = np.arange(
        start,
        start + np.timedelta64(length, "h"),
        dtype="datetime64[h]",
    )
    values = np.arange(length, dtype=np.float64)
    observed = np.ones(length, dtype=np.bool_)
    imputed_indices = [index for index in (10, 35, 115, 145) if index < length]
    observed[imputed_indices] = False
    for index in imputed_indices:
        values[index] = values[index - 1]

    return HourlyPowerSeries(
        timestamps=timestamps,
        values=values,
        observed_mask=observed,
        imputed_mask=~observed,
        valid_minute_counts=np.where(observed, 60, 0),
    )


def test_hourly_loader_is_causal_regular_and_deterministic(tmp_path: Path) -> None:
    raw_path = tmp_path / "household_power_consumption.txt"
    _write_causal_fixture(raw_path)

    series = load_uci_household_power_series(raw_path, window_hours=5)

    expected_start = np.datetime64("2006-12-16T01", "h")
    expected_timestamps = np.arange(
        expected_start,
        expected_start + np.timedelta64(5, "h"),
        dtype="datetime64[h]",
    )
    np.testing.assert_array_equal(series.timestamps, expected_timestamps)
    np.testing.assert_allclose(series.values, [15.5, 15.5, 15.5, 4.0, 5.0])
    np.testing.assert_array_equal(
        series.observed_mask,
        [True, False, False, True, True],
    )
    np.testing.assert_array_equal(
        series.imputed_mask,
        [False, True, True, False, False],
    )
    np.testing.assert_array_equal(series.valid_minute_counts, [30, 0, 29, 30, 30])

    assert series.duplicate_rows_removed == 1
    assert series.source_hour_grid_size == 7
    assert series.leading_hours_dropped == 1
    assert series.trailing_hours_unused == 1
    assert series.missing_target_rows == 2
    assert not series.timestamps.flags.writeable
    assert not series.values.flags.writeable


def test_default_loader_requires_full_twelve_thousand_hour_window(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "small.txt"
    _write_causal_fixture(raw_path)

    assert DEFAULT_WINDOW_HOURS == 12_000
    with pytest.raises(
        InsufficientDataError,
        match=r"required 12000 consecutive hours.*available 6",
    ):
        load_uci_household_power_series(raw_path)


def test_conflicting_duplicate_timestamp_fails_explicitly(tmp_path: Path) -> None:
    timestamp = datetime(2006, 12, 16)
    raw_path = tmp_path / "conflict.txt"
    raw_path.write_text(
        "\n".join(
            [
                UCI_HEADER,
                _uci_row(timestamp, 1.0),
                _uci_row(timestamp, 2.0),
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ConflictingDuplicateError,
        match=r"conflicting values for duplicate timestamp 2006-12-16T00:00",
    ):
        load_uci_household_power_series(raw_path, window_hours=1)


def test_canonical_hourly_splits_remain_unshifted_for_runner_alignment() -> None:
    series = _synthetic_hourly_series()

    splits = split_hourly_series(series)

    assert (splits.train.size, splits.val.size, splits.test.size) == (30, 10, 10)
    assert splits.validation is splits.val
    assert splits.train.timestamps[-1] == series.timestamps[29]
    assert splits.val.timestamps[0] == series.timestamps[30]
    assert splits.test.timestamps[0] == series.timestamps[40]

    partitions = (splits.train, splits.val, splits.test)
    np.testing.assert_array_equal(
        np.concatenate([partition.timestamps for partition in partitions]),
        series.timestamps,
    )
    np.testing.assert_array_equal(
        np.concatenate([partition.y for partition in partitions]),
        series.values,
    )
    np.testing.assert_array_equal(
        np.concatenate([partition.observed_mask for partition in partitions]),
        series.observed_mask,
    )
    for partition in partitions:
        np.testing.assert_array_equal(partition.X[:, 0], partition.y)
        assert partition.X.shape == (partition.size, 1)
        assert not partition.X.flags.writeable
        assert not partition.y.flags.writeable


@pytest.mark.parametrize(
    ("horizon", "expected_sizes"),
    [
        (1, (89, 29, 29)),
        (24, (66, 6, 6)),
    ],
)
def test_forecast_splits_align_horizons_and_target_masks(
    horizon: int,
    expected_sizes: tuple[int, int, int],
) -> None:
    series = _synthetic_hourly_series(length=150)

    splits = build_forecast_splits(series, horizon=horizon)

    assert splits.horizon == horizon
    assert (splits.train.size, splits.val.size, splits.test.size) == expected_sizes
    assert splits.validation is splits.val
    assert splits.train.target_timestamps[-1] == series.timestamps[89]
    assert splits.val.input_timestamps[0] == series.timestamps[90]
    assert splits.val.target_timestamps[0] == series.timestamps[90 + horizon]
    assert splits.test.input_timestamps[0] == series.timestamps[120]
    assert splits.test.target_timestamps[0] == series.timestamps[120 + horizon]

    partition_cases = (
        (splits.train, slice(0, 90)),
        (splits.val, slice(90, 120)),
        (splits.test, slice(120, 150)),
    )
    for partition, source_slice in partition_cases:
        source_values = series.values[source_slice]
        source_times = series.timestamps[source_slice]
        source_imputed = series.imputed_mask[source_slice]
        np.testing.assert_array_equal(partition.X[:, 0], source_values[:-horizon])
        np.testing.assert_array_equal(partition.y, source_values[horizon:])
        np.testing.assert_array_equal(
            partition.input_timestamps,
            source_times[:-horizon],
        )
        np.testing.assert_array_equal(
            partition.target_timestamps,
            source_times[horizon:],
        )
        np.testing.assert_array_equal(
            partition.target_imputed_mask,
            source_imputed[horizon:],
        )
        np.testing.assert_array_equal(
            partition.target_timestamps - partition.input_timestamps,
            np.full(partition.size, np.timedelta64(horizon, "h")),
        )

    assert not splits.train.X.flags.writeable
    assert not splits.train.target_observed_mask.flags.writeable


def test_only_contract_horizons_are_supported() -> None:
    with pytest.raises(ValueError, match=r"horizon must be one of 1, 24"):
        build_forecast_splits(_synthetic_hourly_series(), horizon=2)


def test_hourly_series_rejects_non_monotonic_timestamps() -> None:
    timestamps = np.array(
        [
            np.datetime64("2026-01-01T00", "h"),
            np.datetime64("2026-01-01T02", "h"),
            np.datetime64("2026-01-01T01", "h"),
        ]
    )

    with pytest.raises(
        NonMonotonicTimeError,
        match="hourly timestamps must be strictly increasing and consecutive",
    ):
        HourlyPowerSeries(
            timestamps=timestamps,
            values=np.array([1.0, 2.0, 3.0]),
            observed_mask=np.ones(3, dtype=np.bool_),
            imputed_mask=np.zeros(3, dtype=np.bool_),
            valid_minute_counts=np.full(3, 60),
        )
