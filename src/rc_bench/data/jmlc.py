"""Deterministic preprocessing for the JMLC UCI power-consumption dataset."""

from __future__ import annotations

import csv
import logging
from array import array
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.typing import NDArray


DEFAULT_WINDOW_HOURS = 12_000
MIN_VALID_MINUTES_PER_HOUR = 30
SUPPORTED_HORIZONS = frozenset({1, 24})
TRAIN_FRACTION = 0.6
VALIDATION_FRACTION = 0.2

_EPOCH_ORDINAL = datetime(1970, 1, 1).toordinal()
_REQUIRED_COLUMNS = frozenset({"Date", "Time", "Global_active_power"})
_LOGGER = logging.getLogger(__name__)


class JMLCDataError(ValueError):
    """Base class for JMLC data validation and preprocessing failures."""


class DataFormatError(JMLCDataError):
    """Raised when the raw UCI text file does not follow its declared format."""


class ConflictingDuplicateError(JMLCDataError):
    """Raised when one minute timestamp has more than one target value."""


class InsufficientDataError(JMLCDataError):
    """Raised when the source cannot provide the required chronological window."""


class NonMonotonicTimeError(JMLCDataError):
    """Raised when an output time axis is not strictly consecutive."""


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntegerArray = NDArray[np.int64]
TimestampArray = NDArray[np.datetime64]


def _readonly_array(
    values: Iterable[object] | np.ndarray,
    *,
    dtype: object,
    name: str,
    dimensions: int = 1,
) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    if result.ndim != dimensions:
        raise JMLCDataError(f"{name} must be {dimensions}-dimensional")
    result.setflags(write=False)
    return result


def _validate_consecutive_timestamps(
    timestamps: TimestampArray,
    *,
    label: str,
) -> None:
    if timestamps.size == 0:
        raise InsufficientDataError(f"{label} must not be empty")
    if np.isnat(timestamps).any():
        raise NonMonotonicTimeError(f"{label} must not contain NaT")
    if timestamps.size > 1:
        hour_numbers = timestamps.astype("datetime64[h]").astype(np.int64)
        if np.any(np.diff(hour_numbers) != 1):
            raise NonMonotonicTimeError(
                f"{label} must be strictly increasing and consecutive"
            )


@dataclass(frozen=True)
class HourlyPowerSeries:
    """One causal, regular and immutable hourly power-consumption window."""

    timestamps: TimestampArray
    values: FloatArray
    observed_mask: BoolArray
    imputed_mask: BoolArray
    valid_minute_counts: IntegerArray
    source_rows: int = 0
    duplicate_rows_removed: int = 0
    missing_target_rows: int = 0
    source_hour_grid_size: int = 0
    leading_hours_dropped: int = 0
    trailing_hours_unused: int = 0

    def __post_init__(self) -> None:
        timestamps = _readonly_array(
            self.timestamps,
            dtype="datetime64[h]",
            name="timestamps",
        )
        values = _readonly_array(self.values, dtype=np.float64, name="values")
        observed = _readonly_array(
            self.observed_mask,
            dtype=np.bool_,
            name="observed_mask",
        )
        imputed = _readonly_array(
            self.imputed_mask,
            dtype=np.bool_,
            name="imputed_mask",
        )
        minute_counts = _readonly_array(
            self.valid_minute_counts,
            dtype=np.int64,
            name="valid_minute_counts",
        )

        _validate_consecutive_timestamps(
            timestamps,
            label="hourly timestamps",
        )
        length = timestamps.size
        arrays = {
            "values": values,
            "observed_mask": observed,
            "imputed_mask": imputed,
            "valid_minute_counts": minute_counts,
        }
        for name, candidate in arrays.items():
            if candidate.size != length:
                raise JMLCDataError(
                    f"{name} length {candidate.size} does not match timestamps "
                    f"length {length}"
                )

        if not np.isfinite(values).all():
            raise JMLCDataError("causally filled hourly values must all be finite")
        if not np.array_equal(imputed, ~observed):
            raise JMLCDataError(
                "observed_mask and imputed_mask must be exact complements"
            )
        if np.any(minute_counts < 0):
            raise JMLCDataError("valid_minute_counts must be non-negative")
        if np.any(minute_counts[observed] < MIN_VALID_MINUTES_PER_HOUR):
            raise JMLCDataError(
                "observed hours must have at least 30 valid minute values"
            )
        if np.any(minute_counts[imputed] >= MIN_VALID_MINUTES_PER_HOUR):
            raise JMLCDataError(
                "imputed hours must have fewer than 30 valid minute values"
            )

        statistic_names = (
            "source_rows",
            "duplicate_rows_removed",
            "missing_target_rows",
            "source_hour_grid_size",
            "leading_hours_dropped",
            "trailing_hours_unused",
        )
        for name in statistic_names:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise JMLCDataError(f"{name} must be a non-negative integer")

        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "observed_mask", observed)
        object.__setattr__(self, "imputed_mask", imputed)
        object.__setattr__(self, "valid_minute_counts", minute_counts)

    @property
    def window_hours(self) -> int:
        return int(self.timestamps.size)


@dataclass(frozen=True)
class HourlyPartition:
    """Unshifted hourly data for runner-owned horizon alignment."""

    timestamps: TimestampArray
    values: FloatArray
    observed_mask: BoolArray
    imputed_mask: BoolArray
    valid_minute_counts: IntegerArray

    def __post_init__(self) -> None:
        timestamps = _readonly_array(
            self.timestamps,
            dtype="datetime64[h]",
            name="timestamps",
        )
        values = _readonly_array(self.values, dtype=np.float64, name="values")
        observed = _readonly_array(
            self.observed_mask,
            dtype=np.bool_,
            name="observed_mask",
        )
        imputed = _readonly_array(
            self.imputed_mask,
            dtype=np.bool_,
            name="imputed_mask",
        )
        minute_counts = _readonly_array(
            self.valid_minute_counts,
            dtype=np.int64,
            name="valid_minute_counts",
        )

        _validate_consecutive_timestamps(
            timestamps,
            label="hourly partition timestamps",
        )
        length = timestamps.size
        arrays = {
            "values": values,
            "observed_mask": observed,
            "imputed_mask": imputed,
            "valid_minute_counts": minute_counts,
        }
        for name, candidate in arrays.items():
            if candidate.size != length:
                raise JMLCDataError(
                    f"{name} length {candidate.size} does not match timestamps "
                    f"length {length}"
                )
        if not np.isfinite(values).all():
            raise JMLCDataError("hourly partition values must all be finite")
        if not np.array_equal(imputed, ~observed):
            raise JMLCDataError(
                "observed_mask and imputed_mask must be exact complements"
            )

        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "observed_mask", observed)
        object.__setattr__(self, "imputed_mask", imputed)
        object.__setattr__(self, "valid_minute_counts", minute_counts)

    @property
    def X(self) -> FloatArray:
        """Unshifted one-feature input sequence; the runner applies horizon."""

        return self.values[:, np.newaxis]

    @property
    def y(self) -> FloatArray:
        """Unshifted target sequence; the runner applies horizon."""

        return self.values

    @property
    def size(self) -> int:
        return int(self.values.size)


@dataclass(frozen=True)
class HourlySplits:
    """Canonical unshifted 60/20/20 hourly partitions."""

    train: HourlyPartition
    val: HourlyPartition
    test: HourlyPartition

    def __post_init__(self) -> None:
        if not (
            self.train.timestamps[-1] < self.val.timestamps[0]
            and self.val.timestamps[-1] < self.test.timestamps[0]
        ):
            raise NonMonotonicTimeError(
                "hourly partitions must be chronological and disjoint"
            )
        if not (
            self.val.timestamps[0] - self.train.timestamps[-1]
            == np.timedelta64(1, "h")
            and self.test.timestamps[0] - self.val.timestamps[-1]
            == np.timedelta64(1, "h")
        ):
            raise NonMonotonicTimeError(
                "hourly partitions must preserve the consecutive time axis"
            )

    @property
    def validation(self) -> HourlyPartition:
        return self.val


@dataclass(frozen=True)
class ForecastPartition:
    """Chronological forecast samples for one train/validation/test partition."""

    X: FloatArray
    y: FloatArray
    input_timestamps: TimestampArray
    target_timestamps: TimestampArray
    input_observed_mask: BoolArray
    input_imputed_mask: BoolArray
    target_observed_mask: BoolArray
    target_imputed_mask: BoolArray

    def __post_init__(self) -> None:
        X = _readonly_array(self.X, dtype=np.float64, name="X", dimensions=2)
        if X.shape[1] != 1:
            raise JMLCDataError("X must contain exactly one input feature")
        y = _readonly_array(self.y, dtype=np.float64, name="y")
        input_timestamps = _readonly_array(
            self.input_timestamps,
            dtype="datetime64[h]",
            name="input_timestamps",
        )
        target_timestamps = _readonly_array(
            self.target_timestamps,
            dtype="datetime64[h]",
            name="target_timestamps",
        )
        input_observed = _readonly_array(
            self.input_observed_mask,
            dtype=np.bool_,
            name="input_observed_mask",
        )
        input_imputed = _readonly_array(
            self.input_imputed_mask,
            dtype=np.bool_,
            name="input_imputed_mask",
        )
        target_observed = _readonly_array(
            self.target_observed_mask,
            dtype=np.bool_,
            name="target_observed_mask",
        )
        target_imputed = _readonly_array(
            self.target_imputed_mask,
            dtype=np.bool_,
            name="target_imputed_mask",
        )

        size = y.size
        candidates = {
            "X": X.shape[0],
            "input_timestamps": input_timestamps.size,
            "target_timestamps": target_timestamps.size,
            "input_observed_mask": input_observed.size,
            "input_imputed_mask": input_imputed.size,
            "target_observed_mask": target_observed.size,
            "target_imputed_mask": target_imputed.size,
        }
        for name, candidate_size in candidates.items():
            if candidate_size != size:
                raise JMLCDataError(
                    f"{name} length {candidate_size} does not match y length {size}"
                )
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise JMLCDataError("forecast X and y must contain only finite values")
        if not np.array_equal(input_imputed, ~input_observed):
            raise JMLCDataError(
                "input observed/imputed masks must be exact complements"
            )
        if not np.array_equal(target_imputed, ~target_observed):
            raise JMLCDataError(
                "target observed/imputed masks must be exact complements"
            )

        _validate_consecutive_timestamps(
            input_timestamps,
            label="forecast input timestamps",
        )
        _validate_consecutive_timestamps(
            target_timestamps,
            label="forecast target timestamps",
        )

        object.__setattr__(self, "X", X)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "input_timestamps", input_timestamps)
        object.__setattr__(self, "target_timestamps", target_timestamps)
        object.__setattr__(self, "input_observed_mask", input_observed)
        object.__setattr__(self, "input_imputed_mask", input_imputed)
        object.__setattr__(self, "target_observed_mask", target_observed)
        object.__setattr__(self, "target_imputed_mask", target_imputed)

    @property
    def size(self) -> int:
        return int(self.y.size)


@dataclass(frozen=True)
class ForecastSplits:
    """Leakage-safe target-time partitions for one supported forecast horizon."""

    horizon: int
    train: ForecastPartition
    val: ForecastPartition
    test: ForecastPartition

    def __post_init__(self) -> None:
        if self.horizon not in SUPPORTED_HORIZONS:
            raise ValueError("horizon must be one of 1, 24")

        expected_delta = np.timedelta64(self.horizon, "h")
        for name, partition in (
            ("train", self.train),
            ("validation", self.val),
            ("test", self.test),
        ):
            if not np.all(
                partition.target_timestamps - partition.input_timestamps
                == expected_delta
            ):
                raise JMLCDataError(
                    f"{name} input/target timestamps are not aligned to "
                    f"horizon {self.horizon}"
                )

        if not (
            self.train.target_timestamps[-1] < self.val.target_timestamps[0]
            and self.val.target_timestamps[-1] < self.test.target_timestamps[0]
        ):
            raise NonMonotonicTimeError(
                "forecast target partitions must be chronological and disjoint"
            )

    @property
    def validation(self) -> ForecastPartition:
        return self.val


def _parse_timestamp(date_text: str, time_text: str, *, row_number: int) -> int:
    try:
        day, month, year = (int(part) for part in date_text.split("/"))
        hour, minute, second = (int(part) for part in time_text.split(":"))
        timestamp = datetime(year, month, day, hour, minute, second)
    except (TypeError, ValueError) as exc:
        raise DataFormatError(
            f"row {row_number}: invalid Date/Time {date_text!r} {time_text!r}"
        ) from exc
    if timestamp.second != 0:
        raise DataFormatError(
            f"row {row_number}: expected minute-aligned timestamp, got "
            f"{timestamp.isoformat()}"
        )
    return (
        ((timestamp.toordinal() - _EPOCH_ORDINAL) * 24 + timestamp.hour) * 60
        + timestamp.minute
    )


def _parse_target(target_text: str, *, row_number: int) -> float:
    normalized = target_text.strip()
    if normalized in {"", "?"}:
        return float("nan")
    try:
        value = float(normalized)
    except ValueError as exc:
        raise DataFormatError(
            f"row {row_number}: invalid Global_active_power {target_text!r}"
        ) from exc
    if not np.isfinite(value):
        raise DataFormatError(
            f"row {row_number}: Global_active_power must be finite or '?'"
        )
    return value


def _read_minute_records(raw_path: Path) -> tuple[IntegerArray, FloatArray, int]:
    minute_keys = array("q")
    targets = array("d")
    source_rows = 0

    try:
        source = raw_path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise DataFormatError(f"cannot open raw UCI file {raw_path}: {exc}") from exc

    with source:
        reader = csv.DictReader(source, delimiter=";")
        fieldnames = set(reader.fieldnames or ())
        missing_columns = sorted(_REQUIRED_COLUMNS - fieldnames)
        if missing_columns:
            raise DataFormatError(
                "raw UCI file is missing required columns: "
                + ", ".join(missing_columns)
            )

        for row in reader:
            source_rows += 1
            date_text = row.get("Date")
            time_text = row.get("Time")
            target_text = row.get("Global_active_power")
            if date_text is None or time_text is None or target_text is None:
                raise DataFormatError(f"row {reader.line_num}: malformed UCI row")
            minute_keys.append(
                _parse_timestamp(
                    date_text.strip(),
                    time_text.strip(),
                    row_number=reader.line_num,
                )
            )
            targets.append(
                _parse_target(target_text, row_number=reader.line_num)
            )

    if source_rows == 0:
        raise InsufficientDataError("raw UCI file contains no data rows")

    return (
        np.asarray(minute_keys, dtype=np.int64),
        np.asarray(targets, dtype=np.float64),
        source_rows,
    )


def _format_minute_key(minute_key: int) -> str:
    timestamp = np.datetime64("1970-01-01T00:00", "m") + np.timedelta64(
        int(minute_key),
        "m",
    )
    return str(timestamp)


def _sort_and_deduplicate(
    minute_keys: IntegerArray,
    targets: FloatArray,
) -> tuple[IntegerArray, FloatArray, int]:
    if np.any(minute_keys[1:] < minute_keys[:-1]):
        order = np.argsort(minute_keys, kind="stable")
        minute_keys = minute_keys[order]
        targets = targets[order]

    duplicate_positions = np.flatnonzero(
        minute_keys[1:] == minute_keys[:-1]
    ) + 1
    if duplicate_positions.size == 0:
        return minute_keys, targets, 0

    previous_values = targets[duplicate_positions - 1]
    duplicate_values = targets[duplicate_positions]
    equal_values = (previous_values == duplicate_values) | (
        np.isnan(previous_values) & np.isnan(duplicate_values)
    )
    if not np.all(equal_values):
        conflict_position = int(
            duplicate_positions[np.flatnonzero(~equal_values)[0]]
        )
        raise ConflictingDuplicateError(
            "conflicting values for duplicate timestamp "
            f"{_format_minute_key(int(minute_keys[conflict_position]))}"
        )

    keep = np.ones(minute_keys.size, dtype=np.bool_)
    keep[duplicate_positions] = False
    return (
        minute_keys[keep],
        targets[keep],
        int(duplicate_positions.size),
    )


def _aggregate_regular_hours(
    minute_keys: IntegerArray,
    targets: FloatArray,
) -> tuple[IntegerArray, FloatArray, BoolArray, IntegerArray]:
    hour_keys = minute_keys // 60
    first_hour = int(hour_keys[0])
    last_hour = int(hour_keys[-1])
    grid_size = last_hour - first_hour + 1
    regular_hour_keys = np.arange(
        first_hour,
        last_hour + 1,
        dtype=np.int64,
    )
    hour_offsets = hour_keys - first_hour
    valid_targets = ~np.isnan(targets)
    valid_counts = np.bincount(
        hour_offsets[valid_targets],
        minlength=grid_size,
    ).astype(np.int64, copy=False)
    target_sums = np.bincount(
        hour_offsets[valid_targets],
        weights=targets[valid_targets],
        minlength=grid_size,
    )

    observed = valid_counts >= MIN_VALID_MINUTES_PER_HOUR
    hourly_values = np.full(grid_size, np.nan, dtype=np.float64)
    hourly_values[observed] = (
        target_sums[observed] / valid_counts[observed]
    )
    return regular_hour_keys, hourly_values, observed, valid_counts


def load_uci_household_power_series(
    raw_path: str | Path,
    *,
    window_hours: int = DEFAULT_WINDOW_HOURS,
) -> HourlyPowerSeries:
    """Load the first causal, consecutive hourly window from the raw UCI TXT.

    Parsing is line-by-line and stores only compact timestamp/target arrays.
    Input order is irrelevant: minute timestamps are sorted before exact
    duplicate removal and hourly aggregation.
    """

    if (
        not isinstance(window_hours, int)
        or isinstance(window_hours, bool)
        or window_hours <= 0
    ):
        raise ValueError("window_hours must be a positive integer")

    minute_keys, targets, source_rows = _read_minute_records(Path(raw_path))
    minute_keys, targets, duplicate_rows_removed = _sort_and_deduplicate(
        minute_keys,
        targets,
    )
    missing_target_rows = int(np.isnan(targets).sum())
    hour_keys, hourly_values, observed, minute_counts = _aggregate_regular_hours(
        minute_keys,
        targets,
    )

    observed_indices = np.flatnonzero(observed)
    if observed_indices.size == 0:
        raise InsufficientDataError(
            "no hour has at least 30 valid Global_active_power minute values"
        )

    first_observed = int(observed_indices[0])
    available_hours = int(hour_keys.size - first_observed)
    if available_hours < window_hours:
        raise InsufficientDataError(
            f"required {window_hours} consecutive hours after first observed hour; "
            f"available {available_hours}"
        )

    selection = slice(first_observed, first_observed + window_hours)
    selected_observed = observed[selection].copy()
    selected_hourly_values = hourly_values[selection]
    selected_minute_counts = minute_counts[selection]

    last_observed_indices = np.maximum.accumulate(
        np.where(selected_observed, np.arange(window_hours), -1)
    )
    if np.any(last_observed_indices < 0):
        raise JMLCDataError(
            "causal fill invariant failed: selected window has no initial observation"
        )
    filled_values = selected_hourly_values[last_observed_indices]
    selected_hour_keys = hour_keys[selection]
    selected_timestamps = selected_hour_keys.astype("datetime64[h]")
    trailing_hours_unused = int(
        hour_keys.size - first_observed - window_hours
    )

    _LOGGER.info(
        "Prepared UCI hourly window: source_rows=%d, "
        "duplicate_rows_removed=%d, leading_hours_dropped=%d, "
        "trailing_hours_unused=%d",
        source_rows,
        duplicate_rows_removed,
        first_observed,
        trailing_hours_unused,
    )

    return HourlyPowerSeries(
        timestamps=selected_timestamps,
        values=filled_values,
        observed_mask=selected_observed,
        imputed_mask=~selected_observed,
        valid_minute_counts=selected_minute_counts,
        source_rows=source_rows,
        duplicate_rows_removed=duplicate_rows_removed,
        missing_target_rows=missing_target_rows,
        source_hour_grid_size=int(hour_keys.size),
        leading_hours_dropped=first_observed,
        trailing_hours_unused=trailing_hours_unused,
    )


def _forecast_partition(
    partition: HourlyPartition,
    *,
    horizon: int,
) -> ForecastPartition:
    return ForecastPartition(
        X=partition.values[:-horizon, np.newaxis],
        y=partition.values[horizon:],
        input_timestamps=partition.timestamps[:-horizon],
        target_timestamps=partition.timestamps[horizon:],
        input_observed_mask=partition.observed_mask[:-horizon],
        input_imputed_mask=partition.imputed_mask[:-horizon],
        target_observed_mask=partition.observed_mask[horizon:],
        target_imputed_mask=partition.imputed_mask[horizon:],
    )


def _hourly_partition(
    series: HourlyPowerSeries,
    selection: slice,
) -> HourlyPartition:
    return HourlyPartition(
        timestamps=series.timestamps[selection],
        values=series.values[selection],
        observed_mask=series.observed_mask[selection],
        imputed_mask=series.imputed_mask[selection],
        valid_minute_counts=series.valid_minute_counts[selection],
    )


def split_hourly_series(series: HourlyPowerSeries) -> HourlySplits:
    """Split the unshifted hourly sequence for runner-owned horizon alignment.

    Both ``X`` and ``y`` in each returned partition are the same unshifted
    sequence. Code that already applies ``H[:-h]``/``y[h:]`` must use this API,
    not :func:`build_forecast_splits`.
    """

    length = series.window_hours
    train_end = int(length * TRAIN_FRACTION)
    validation_end = train_end + int(length * VALIDATION_FRACTION)
    if train_end == 0 or validation_end == train_end or validation_end == length:
        raise InsufficientDataError(
            f"cannot create non-empty 60/20/20 splits from {length} hourly values"
        )

    return HourlySplits(
        train=_hourly_partition(series, slice(0, train_end)),
        val=_hourly_partition(series, slice(train_end, validation_end)),
        test=_hourly_partition(series, slice(validation_end, length)),
    )


def build_forecast_splits(
    series: HourlyPowerSeries,
    *,
    horizon: int,
) -> ForecastSplits:
    """Build horizon-aligned samples inside each canonical 60/20/20 split.

    This mirrors the current runner's ``X_split[:-h]``/``y_split[h:]``
    alignment. No input/target pair crosses a split boundary.
    """

    if (
        not isinstance(horizon, int)
        or isinstance(horizon, bool)
        or horizon not in SUPPORTED_HORIZONS
    ):
        raise ValueError("horizon must be one of 1, 24")

    hourly_splits = split_hourly_series(series)
    partitions = {
        "train": hourly_splits.train,
        "validation": hourly_splits.val,
        "test": hourly_splits.test,
    }
    for name, partition in partitions.items():
        if partition.size <= horizon:
            raise InsufficientDataError(
                f"{name} split has {partition.size} hours, insufficient for "
                f"horizon {horizon}"
            )

    return ForecastSplits(
        horizon=horizon,
        train=_forecast_partition(hourly_splits.train, horizon=horizon),
        val=_forecast_partition(hourly_splits.val, horizon=horizon),
        test=_forecast_partition(hourly_splits.test, horizon=horizon),
    )
