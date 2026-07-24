"""Reproducible data acquisition helpers for the JMLC benchmark."""

from rc_bench.data.download import (
    ChecksumMismatchError,
    DatasetDownloadError,
    DatasetManifestError,
    download_dataset,
)
from rc_bench.data.jmlc import (
    DEFAULT_WINDOW_HOURS,
    MIN_VALID_MINUTES_PER_HOUR,
    ConflictingDuplicateError,
    DataFormatError,
    ForecastPartition,
    ForecastSplits,
    HourlyPartition,
    HourlyPowerSeries,
    HourlySplits,
    InsufficientDataError,
    JMLCDataError,
    NonMonotonicTimeError,
    build_forecast_splits,
    load_uci_household_power_series,
    split_hourly_series,
)

__all__ = [
    "ChecksumMismatchError",
    "ConflictingDuplicateError",
    "DEFAULT_WINDOW_HOURS",
    "DataFormatError",
    "DatasetDownloadError",
    "DatasetManifestError",
    "ForecastPartition",
    "ForecastSplits",
    "HourlyPartition",
    "HourlyPowerSeries",
    "HourlySplits",
    "InsufficientDataError",
    "JMLCDataError",
    "MIN_VALID_MINUTES_PER_HOUR",
    "NonMonotonicTimeError",
    "build_forecast_splits",
    "download_dataset",
    "load_uci_household_power_series",
    "split_hourly_series",
]
