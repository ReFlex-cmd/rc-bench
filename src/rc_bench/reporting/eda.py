"""Deterministic observed-only EDA for the JMLC household-power benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from rc_bench.data.jmlc import (
    MIN_VALID_MINUTES_PER_HOUR,
    HourlyPartition,
    HourlyPowerSeries,
    build_forecast_splits,
    split_hourly_series,
)


ACF_MAX_LAG_HOURS = 168
EDA_ARTIFACT_PATHS = (
    "eda_report.md",
    "aggregates/eda_summary.json",
    "plots/eda_coverage_missingness.png",
    "plots/eda_target_distribution.png",
    "plots/eda_daily_weekly_profiles.png",
    "plots/eda_acf_168h.png",
    "plots/eda_split_distributions.png",
)

_DAY_NAMES = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)
_PNG_METADATA = {"Software": "rc-bench"}


class EDAInvariantError(ValueError):
    """Raised before output when an EDA leakage/data invariant is violated."""


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _iso_hour(value: np.datetime64) -> str:
    return str(value.astype("datetime64[h]"))


def _statistics(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    count = int(values.size)
    if count == 0:
        return {
            "count": 0,
            "min": None,
            "q01": None,
            "q05": None,
            "q25": None,
            "median": None,
            "q75": None,
            "q95": None,
            "q99": None,
            "max": None,
            "mean": None,
            "sample_std": None,
        }

    quantiles = np.quantile(
        values,
        [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99],
    )
    return {
        "count": count,
        "min": float(np.min(values)),
        "q01": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "q25": float(quantiles[2]),
        "median": float(quantiles[3]),
        "q75": float(quantiles[4]),
        "q95": float(quantiles[5]),
        "q99": float(quantiles[6]),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "sample_std": (
            float(np.std(values, ddof=1))
            if count > 1
            else None
        ),
    }


def _outlier_summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            "method": "Tukey 1.5 IQR",
            "lower_fence": None,
            "upper_fence": None,
            "count": 0,
            "fraction": None,
        }
    q25, q75 = np.quantile(values, [0.25, 0.75])
    iqr = q75 - q25
    lower = float(q25 - 1.5 * iqr)
    upper = float(q75 + 1.5 * iqr)
    count = int(np.count_nonzero((values < lower) | (values > upper)))
    return {
        "method": "Tukey 1.5 IQR",
        "lower_fence": lower,
        "upper_fence": upper,
        "count": count,
        "fraction": _ratio(count, int(values.size)),
    }


def _group_profile(
    values: np.ndarray,
    observed: np.ndarray,
    groups: np.ndarray,
    labels: Iterable[int | str],
    *,
    key: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_index, label in enumerate(labels):
        selected = values[observed & (groups == group_index)]
        stats = _statistics(selected)
        rows.append(
            {
                key: label,
                "count": stats["count"],
                "mean": stats["mean"],
                "median": stats["median"],
                "q25": stats["q25"],
                "q75": stats["q75"],
            }
        )
    return rows


def _pairwise_observed_acf(
    values: np.ndarray,
    observed: np.ndarray,
    *,
    max_lag: int,
) -> dict[str, Any]:
    upper_lag = min(max_lag, int(values.size) - 1)
    lags = list(range(upper_lag + 1))
    correlations: list[float | None] = []
    pair_counts: list[int] = []

    for lag in lags:
        if lag == 0:
            left = values
            right = values
            valid = observed
        else:
            left = values[:-lag]
            right = values[lag:]
            valid = observed[:-lag] & observed[lag:]
        left = left[valid]
        right = right[valid]
        pair_counts.append(int(left.size))
        if left.size < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
            correlations.append(None)
        else:
            correlations.append(float(np.corrcoef(left, right)[0, 1]))

    return {
        "method": "pairwise-observed Pearson correlation",
        "max_lag_hours": upper_lag,
        "lags_hours": lags,
        "correlations": correlations,
        "pair_counts": pair_counts,
    }


def _partition_summary(partition: HourlyPartition) -> dict[str, Any]:
    observed_values = partition.values[partition.observed_mask]
    return {
        "start": _iso_hour(partition.timestamps[0]),
        "end": _iso_hour(partition.timestamps[-1]),
        "hours": partition.size,
        "observed_hours": int(np.count_nonzero(partition.observed_mask)),
        "imputed_hours": int(np.count_nonzero(partition.imputed_mask)),
        "observed_target_statistics": _statistics(observed_values),
    }


def _forecast_check(
    series: HourlyPowerSeries,
    *,
    horizon: int,
) -> dict[str, Any]:
    hourly = split_hourly_series(series)
    forecast = build_forecast_splits(series, horizon=horizon)
    expected_delta = np.timedelta64(horizon, "h")
    aligned = True
    within_blocks = True

    for source, target in (
        (hourly.train, forecast.train),
        (hourly.val, forecast.val),
        (hourly.test, forecast.test),
    ):
        aligned = aligned and bool(
            np.all(
                target.target_timestamps - target.input_timestamps
                == expected_delta
            )
        )
        within_blocks = within_blocks and bool(
            target.input_timestamps[0] >= source.timestamps[0]
            and target.target_timestamps[-1] <= source.timestamps[-1]
        )

    chronological = bool(
        forecast.train.target_timestamps[-1]
        < forecast.val.target_timestamps[0]
        < forecast.test.target_timestamps[0]
    )
    return {
        "aligned_exactly": aligned,
        "pairs_within_source_split": within_blocks,
        "target_partitions_chronological": chronological,
        "no_cross_boundary": aligned and within_blocks and chronological,
        "sample_counts": {
            "train": forecast.train.size,
            "val": forecast.val.size,
            "test": forecast.test.size,
        },
    }


def _leakage_summary(series: HourlyPowerSeries) -> dict[str, Any]:
    timestamps = series.timestamps.astype("datetime64[h]").astype(np.int64)
    observed = series.observed_mask
    indices = np.arange(series.window_hours)
    last_observed = np.maximum.accumulate(
        np.where(observed, indices, -1)
    )
    causal = bool(
        observed[0]
        and np.all(last_observed >= 0)
        and np.array_equal(
            series.values[series.imputed_mask],
            series.values[last_observed[series.imputed_mask]],
        )
    )

    splits = split_hourly_series(series)
    expected_train = int(series.window_hours * 0.6)
    expected_val = int(series.window_hours * 0.2)
    expected_test = series.window_hours - expected_train - expected_val
    chronological = bool(
        splits.train.timestamps[-1] + np.timedelta64(1, "h")
        == splits.val.timestamps[0]
        and splits.val.timestamps[-1] + np.timedelta64(1, "h")
        == splits.test.timestamps[0]
    )
    split_sizes = (
        splits.train.size,
        splits.val.size,
        splits.test.size,
    )

    checks = {
        "regular_hourly_axis": bool(
            timestamps.size > 0
            and (
                timestamps.size == 1
                or np.all(np.diff(timestamps) == 1)
            )
        ),
        "first_hour_observed": bool(observed[0]),
        "causal_forward_fill": causal,
        "observed_imputed_masks_complementary": bool(
            np.array_equal(series.imputed_mask, ~observed)
        ),
        "chronological_disjoint_60_20_20": (
            chronological
            and split_sizes
            == (expected_train, expected_val, expected_test)
        ),
        "source_grid_accounting": (
            series.source_hour_grid_size
            == (
                series.leading_hours_dropped
                + series.window_hours
                + series.trailing_hours_unused
            )
        ),
        "observed_only_statistics": True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        labels = {
            "causal_forward_fill": "causal forward-fill",
        }
        failure_text = ", ".join(labels.get(name, name) for name in failed)
        raise EDAInvariantError(f"EDA invariant failed: {failure_text}")

    horizon_checks = {
        str(horizon): _forecast_check(series, horizon=horizon)
        for horizon in (1, 24)
    }
    failed_horizons = [
        horizon
        for horizon, check in horizon_checks.items()
        if not check["no_cross_boundary"]
    ]
    if failed_horizons:
        raise EDAInvariantError(
            "EDA leakage check failed for horizon(s): "
            + ", ".join(failed_horizons)
        )

    return {
        "all_passed": True,
        "checks": checks,
        "forecast_horizons": horizon_checks,
        "protocol": {
            "shuffle_used": False,
            "scaler_fit_performed": False,
            "statistics_scope": "observed targets only",
        },
    }


def analyze_eda(
    series: HourlyPowerSeries,
    *,
    raw_sha256: str,
) -> dict[str, Any]:
    """Return a JSON-safe observed-only EDA summary or fail an invariant."""

    leakage = _leakage_summary(series)
    observed_values = series.values[series.observed_mask]
    window_hours = series.window_hours
    source_present_rows = (
        series.source_rows - series.duplicate_rows_removed
    )
    valid_window_minutes = int(series.valid_minute_counts.sum())
    total_window_minutes = window_hours * 60
    absent_or_invalid = total_window_minutes - valid_window_minutes

    source_grid_start = (
        series.timestamps[0]
        - np.timedelta64(series.leading_hours_dropped, "h")
    )
    source_grid_end = (
        series.timestamps[-1]
        + np.timedelta64(series.trailing_hours_unused, "h")
    )

    hour_numbers = series.timestamps.astype("datetime64[h]").astype(np.int64)
    hours_of_day = hour_numbers % 24
    day_numbers = series.timestamps.astype("datetime64[D]").astype(np.int64)
    days_of_week = (day_numbers + 3) % 7

    splits = split_hourly_series(series)
    minute_count_stats = _statistics(
        series.valid_minute_counts.astype(np.float64)
    )

    return {
        "schema_version": 1,
        "dataset": {
            "name": "UCI Individual Household Electric Power Consumption",
            "target": "Global_active_power",
            "target_unit": "kilowatts",
            "frequency": "1 hour",
            "raw_sha256": raw_sha256,
            "window_selection": (
                "first deterministic consecutive window after the first "
                "hour with at least 30 valid minutes"
            ),
        },
        "coverage": {
            "source_grid_start": _iso_hour(source_grid_start),
            "source_grid_end": _iso_hour(source_grid_end),
            "source_rows": series.source_rows,
            "duplicate_rows_removed": series.duplicate_rows_removed,
            "source_hour_grid_size": series.source_hour_grid_size,
            "leading_hours_dropped": series.leading_hours_dropped,
            "trailing_hours_unused": series.trailing_hours_unused,
            "window_start": _iso_hour(series.timestamps[0]),
            "window_end": _iso_hour(series.timestamps[-1]),
            "window_hours": window_hours,
        },
        "missingness": {
            "minute_level_before_resampling": {
                "scope": (
                    "source-present rows after exact deduplication"
                ),
                "present_rows": source_present_rows,
                "missing_target_rows": series.missing_target_rows,
                "missing_target_fraction": _ratio(
                    series.missing_target_rows,
                    source_present_rows,
                ),
            },
            "selected_window_minute_slots": {
                "scope": (
                    "selected window; absent and invalid slots combined"
                ),
                "total_slots": total_window_minutes,
                "valid_slots": valid_window_minutes,
                "absent_or_invalid_slots": absent_or_invalid,
                "absent_or_invalid_fraction": _ratio(
                    absent_or_invalid,
                    total_window_minutes,
                ),
            },
            "hour_level_after_resampling": {
                "minimum_valid_minutes": MIN_VALID_MINUTES_PER_HOUR,
                "observed_hours": int(np.count_nonzero(series.observed_mask)),
                "imputed_hours": int(np.count_nonzero(series.imputed_mask)),
                "imputed_fraction": _ratio(
                    int(np.count_nonzero(series.imputed_mask)),
                    window_hours,
                ),
                "valid_minute_count_statistics": minute_count_stats,
            },
        },
        "target_distribution": {
            "scope": "observed hourly targets only",
            "statistics": _statistics(observed_values),
            "outliers": _outlier_summary(observed_values),
        },
        "profiles": {
            "scope": "observed hourly targets only",
            "hour_of_day": _group_profile(
                series.values,
                series.observed_mask,
                hours_of_day,
                range(24),
                key="hour",
            ),
            "day_of_week": _group_profile(
                series.values,
                series.observed_mask,
                days_of_week,
                _DAY_NAMES,
                key="day",
            ),
        },
        "acf": _pairwise_observed_acf(
            series.values,
            series.observed_mask,
            max_lag=ACF_MAX_LAG_HOURS,
        ),
        "splits": {
            "train": _partition_summary(splits.train),
            "val": _partition_summary(splits.val),
            "test": _partition_summary(splits.test),
        },
        "leakage": leakage,
    }


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_figure(fig: Any, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(
        path,
        dpi=150,
        metadata=_PNG_METADATA,
    )
    _pyplot().close(fig)


def _plot_coverage_missingness(
    series: HourlyPowerSeries,
    summary: dict[str, Any],
    path: Path,
) -> None:
    plt = _pyplot()
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    steps = np.arange(series.window_hours)
    axes[0].plot(
        steps,
        series.valid_minute_counts,
        color="steelblue",
        linewidth=0.7,
    )
    axes[0].axhline(
        MIN_VALID_MINUTES_PER_HOUR,
        color="firebrick",
        linestyle="--",
        linewidth=1,
        label="observed-hour threshold",
    )
    axes[0].set_ylabel("valid minutes")
    axes[0].set_xlabel("hour in deterministic window")
    axes[0].set_ylim(-1, 61)
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.25)

    hourly = summary["missingness"]["hour_level_after_resampling"]
    axes[1].bar(
        ["observed", "causally imputed"],
        [hourly["observed_hours"], hourly["imputed_hours"]],
        color=["steelblue", "darkorange"],
    )
    axes[1].set_ylabel("hours")
    axes[1].set_title("Hourly coverage after resampling")
    axes[1].grid(axis="y", alpha=0.25)
    _save_figure(fig, path)


def _plot_target_distribution(
    series: HourlyPowerSeries,
    summary: dict[str, Any],
    path: Path,
) -> None:
    plt = _pyplot()
    observed = series.values[series.observed_mask]
    outliers = summary["target_distribution"]["outliers"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(
        observed,
        bins=50,
        color="steelblue",
        edgecolor="white",
        linewidth=0.4,
    )
    for fence, color in (
        (outliers["lower_fence"], "darkorange"),
        (outliers["upper_fence"], "firebrick"),
    ):
        if fence is not None:
            axes[0].axvline(fence, color=color, linestyle="--", linewidth=1)
    axes[0].set_xlabel("Global_active_power (kW)")
    axes[0].set_ylabel("observed hours")
    axes[0].set_title("Observed target distribution")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].boxplot(
        observed,
        orientation="horizontal",
        widths=0.5,
        boxprops={"color": "steelblue"},
        medianprops={"color": "firebrick"},
    )
    axes[1].set_xlabel("Global_active_power (kW)")
    axes[1].set_yticks([])
    axes[1].set_title("Tukey outlier view")
    axes[1].grid(axis="x", alpha=0.25)
    _save_figure(fig, path)


def _profile_arrays(
    rows: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.array(
        [np.nan if row["mean"] is None else row["mean"] for row in rows],
        dtype=np.float64,
    )
    q25 = np.array(
        [np.nan if row["q25"] is None else row["q25"] for row in rows],
        dtype=np.float64,
    )
    q75 = np.array(
        [np.nan if row["q75"] is None else row["q75"] for row in rows],
        dtype=np.float64,
    )
    return mean, q25, q75


def _plot_profiles(summary: dict[str, Any], path: Path) -> None:
    plt = _pyplot()
    daily = summary["profiles"]["hour_of_day"]
    weekly = summary["profiles"]["day_of_week"]
    daily_mean, daily_q25, daily_q75 = _profile_arrays(daily)
    weekly_mean, weekly_q25, weekly_q75 = _profile_arrays(weekly)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    hours = np.arange(24)
    axes[0].plot(hours, daily_mean, color="steelblue", marker="o", markersize=3)
    axes[0].fill_between(
        hours,
        daily_q25,
        daily_q75,
        color="steelblue",
        alpha=0.2,
        label="IQR",
    )
    axes[0].set_xticks(np.arange(0, 24, 3))
    axes[0].set_xlabel("hour of day")
    axes[0].set_ylabel("Global_active_power (kW)")
    axes[0].set_title("Observed daily profile")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    days = np.arange(7)
    axes[1].plot(days, weekly_mean, color="darkorange", marker="o")
    axes[1].fill_between(
        days,
        weekly_q25,
        weekly_q75,
        color="darkorange",
        alpha=0.2,
        label="IQR",
    )
    axes[1].set_xticks(days)
    axes[1].set_xticklabels([name[:3] for name in _DAY_NAMES])
    axes[1].set_xlabel("day of week")
    axes[1].set_ylabel("Global_active_power (kW)")
    axes[1].set_title("Observed weekly profile")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    _save_figure(fig, path)


def _plot_acf(summary: dict[str, Any], path: Path) -> None:
    plt = _pyplot()
    acf = summary["acf"]
    lags = np.asarray(acf["lags_hours"], dtype=np.int64)
    correlations = np.array(
        [
            np.nan if value is None else value
            for value in acf["correlations"]
        ],
        dtype=np.float64,
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    valid = np.isfinite(correlations)
    ax.vlines(
        lags[valid],
        0,
        correlations[valid],
        color="steelblue",
        linewidth=0.8,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    for lag, label in ((24, "24 h"), (168, "168 h")):
        if lag <= acf["max_lag_hours"]:
            ax.axvline(
                lag,
                color="firebrick",
                linestyle="--",
                linewidth=1,
                label=label,
            )
    ax.set_xlim(0, acf["max_lag_hours"])
    ax.set_xlabel("lag (hours)")
    ax.set_ylabel("pairwise-observed correlation")
    ax.set_title("Observed target autocorrelation (bounded)")
    ax.legend()
    ax.grid(alpha=0.25)
    _save_figure(fig, path)


def _plot_split_distributions(
    series: HourlyPowerSeries,
    path: Path,
) -> None:
    plt = _pyplot()
    splits = split_hourly_series(series)
    labels = ("train", "val", "test")
    values = [
        partition.values[partition.observed_mask]
        for partition in (splits.train, splits.val, splits.test)
    ]
    all_values = np.concatenate(values)
    lower = float(np.min(all_values))
    upper = float(np.max(all_values))
    if lower == upper:
        lower -= 0.5
        upper += 0.5
    bins = np.linspace(lower, upper, 41)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = ("steelblue", "darkorange", "seagreen")
    for label, split_values, color in zip(labels, values, colors, strict=True):
        axes[0].hist(
            split_values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.4,
            color=color,
            label=f"{label} (n={split_values.size})",
        )
    axes[0].set_xlabel("Global_active_power (kW)")
    axes[0].set_ylabel("density")
    axes[0].set_title("Observed split distributions")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].boxplot(
        values,
        tick_labels=labels,
        showfliers=True,
        boxprops={"color": "steelblue"},
        medianprops={"color": "firebrick"},
    )
    axes[1].set_ylabel("Global_active_power (kW)")
    axes[1].set_title("Observed split comparison")
    axes[1].grid(axis="y", alpha=0.25)
    _save_figure(fig, path)


def _fmt(value: Any) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _markdown(summary: dict[str, Any]) -> str:
    dataset = summary["dataset"]
    coverage = summary["coverage"]
    missingness = summary["missingness"]
    distribution = summary["target_distribution"]
    outliers = distribution["outliers"]
    leakage = summary["leakage"]

    lines = [
        "# JMLC 2026 Dataset EDA",
        "",
        "All target statistics in this report use observed hourly targets only.",
        "Causally imputed hours are reported as missingness and excluded from "
        "distributional claims.",
        "",
        "## Dataset and deterministic window",
        "",
        f"- Dataset: {dataset['name']}",
        f"- Target: `{dataset['target']}` ({dataset['target_unit']})",
        f"- Raw SHA-256: `{dataset['raw_sha256']}`",
        f"- Source grid: {coverage['source_grid_start']} — "
        f"{coverage['source_grid_end']}",
        f"- Selected window: {coverage['window_start']} — "
        f"{coverage['window_end']} ({coverage['window_hours']} hours)",
        f"- Exact duplicates removed: {coverage['duplicate_rows_removed']}",
        "",
        "![Coverage and missingness](plots/eda_coverage_missingness.png)",
        "",
        "## Missingness",
        "",
        "| Scope | Missing | Total | Fraction |",
        "| --- | ---: | ---: | ---: |",
    ]
    source_minute = missingness["minute_level_before_resampling"]
    selected_minute = missingness["selected_window_minute_slots"]
    hourly = missingness["hour_level_after_resampling"]
    lines.extend(
        [
            "| Source-present minute rows with `?` | "
            f"{source_minute['missing_target_rows']} | "
            f"{source_minute['present_rows']} | "
            f"{_fmt(source_minute['missing_target_fraction'])} |",
            "| Selected-window absent or invalid minute slots | "
            f"{selected_minute['absent_or_invalid_slots']} | "
            f"{selected_minute['total_slots']} | "
            f"{_fmt(selected_minute['absent_or_invalid_fraction'])} |",
            "| Hours below the 30-valid-minute threshold | "
            f"{hourly['imputed_hours']} | {coverage['window_hours']} | "
            f"{_fmt(hourly['imputed_fraction'])} |",
            "",
            "The source-present `?` rate and the selected-window "
            "absent-or-invalid minute-slot rate have different denominators "
            "and are not interchangeable.",
            "",
            "## Observed target distribution and outliers",
            "",
            f"- Observed hours: {distribution['statistics']['count']}",
            f"- Mean: {_fmt(distribution['statistics']['mean'])} kW",
            f"- Median: {_fmt(distribution['statistics']['median'])} kW",
            f"- Sample standard deviation: "
            f"{_fmt(distribution['statistics']['sample_std'])} kW",
            f"- Range: {_fmt(distribution['statistics']['min'])} — "
            f"{_fmt(distribution['statistics']['max'])} kW",
            f"- Tukey 1.5 IQR outliers: {outliers['count']} "
            f"({_fmt(outliers['fraction'])})",
            "",
            "![Observed target distribution](plots/eda_target_distribution.png)",
            "",
            "## Daily and weekly profiles",
            "",
            "Profiles are grouped by calendar hour and weekday using observed "
            "hours only; the shaded band is the interquartile range.",
            "",
            "![Daily and weekly profiles](plots/eda_daily_weekly_profiles.png)",
            "",
            "## Bounded autocorrelation",
            "",
            "ACF uses pairwise-observed Pearson correlations for lags 0–168 "
            "hours. Each lag records its own valid-pair count in the JSON "
            "summary.",
            "",
            "![Bounded ACF](plots/eda_acf_168h.png)",
            "",
            "## Chronological split comparison",
            "",
            "| Split | Range | Hours | Observed | Imputed | Mean (kW) |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name in ("train", "val", "test"):
        split = summary["splits"][name]
        lines.append(
            f"| {name} | {split['start']} — {split['end']} | "
            f"{split['hours']} | {split['observed_hours']} | "
            f"{split['imputed_hours']} | "
            f"{_fmt(split['observed_target_statistics']['mean'])} |"
        )
    lines.extend(
        [
            "",
            "![Observed split distributions](plots/eda_split_distributions.png)",
            "",
            "## Leakage and invariant checks",
            "",
            f"- Overall status: **{'PASS' if leakage['all_passed'] else 'FAIL'}**",
        ]
    )
    for name, passed in leakage["checks"].items():
        lines.append(f"- `{name}`: {'PASS' if passed else 'FAIL'}")
    for horizon in ("1", "24"):
        check = leakage["forecast_horizons"][horizon]
        lines.append(
            f"- Horizon {horizon}: "
            f"{'PASS' if check['no_cross_boundary'] else 'FAIL'} "
            "(exact alignment and no pair crosses a split boundary)"
        )
    lines.extend(
        [
            "- Random shuffle: not used",
            "- Scaler: not fitted by EDA",
            "- Statistical scope: observed targets only",
            "",
            "Machine-readable details: "
            "[`aggregates/eda_summary.json`](aggregates/eda_summary.json).",
            "",
        ]
    )
    return "\n".join(lines)


def generate_eda_report(
    series: HourlyPowerSeries,
    output_dir: Path,
    *,
    raw_sha256: str,
) -> list[Path]:
    """Overwrite the fixed EDA evidence files after all invariants pass."""

    summary = analyze_eda(series, raw_sha256=raw_sha256)
    output_dir = Path(output_dir)
    aggregate_dir = output_dir / "aggregates"
    plot_dir = output_dir / "plots"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    markdown_path = output_dir / "eda_report.md"
    summary_path = aggregate_dir / "eda_summary.json"
    coverage_path = plot_dir / "eda_coverage_missingness.png"
    distribution_path = plot_dir / "eda_target_distribution.png"
    profiles_path = plot_dir / "eda_daily_weekly_profiles.png"
    acf_path = plot_dir / "eda_acf_168h.png"
    splits_path = plot_dir / "eda_split_distributions.png"

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_markdown(summary), encoding="utf-8")
    _plot_coverage_missingness(series, summary, coverage_path)
    _plot_target_distribution(series, summary, distribution_path)
    _plot_profiles(summary, profiles_path)
    _plot_acf(summary, acf_path)
    _plot_split_distributions(series, splits_path)

    paths = [
        markdown_path,
        summary_path,
        coverage_path,
        distribution_path,
        profiles_path,
        acf_path,
        splits_path,
    ]
    actual = {
        path.relative_to(output_dir).as_posix()
        for path in paths
    }
    if actual != set(EDA_ARTIFACT_PATHS):
        raise RuntimeError("internal EDA artifact set does not match contract")
    return paths
