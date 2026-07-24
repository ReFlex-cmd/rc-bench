"""Deterministic baseline predictors for the JMLC real-data matrix.

Persistence, seasonal persistence and the Ridge AR feature builder operate on
a single split's unshifted ``values`` array and the split-local target
positions the runner has already aligned (washout + horizon) and masked to
observed targets. All lags reach only into causal within-split history; a lag
that would read before index 0 is an explicit protocol error rather than a
silently wrapped negative NumPy index.
"""

from __future__ import annotations

import numpy as np

SEASONAL_PERIOD = 24
RIDGE_AR_LAGS = 24


def _causal_index(
    positions: np.ndarray,
    offset: np.ndarray | int,
    n_values: int,
) -> np.ndarray:
    """Return ``positions - offset`` after checking it stays in ``[0, n_values)``."""
    index = np.asarray(positions, dtype=np.int64) - offset
    if np.asarray(index).min(initial=0) < 0:
        raise ValueError(
            "baseline lag reaches before index 0: insufficient causal history"
        )
    if np.asarray(index).max(initial=0) >= n_values:
        raise ValueError("baseline target position is outside the value series")
    return index


def persistence_forecast(
    values: np.ndarray,
    target_positions: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Naive persistence: ``y_hat(tau) = values[tau - horizon]``."""
    values = np.asarray(values, dtype=float)
    index = _causal_index(target_positions, horizon, values.size)
    return values[index]


def seasonal_persistence_forecast(
    values: np.ndarray,
    target_positions: np.ndarray,
    season: int = SEASONAL_PERIOD,
) -> np.ndarray:
    """Seasonal persistence: ``y_hat(tau) = values[tau - season]``."""
    values = np.asarray(values, dtype=float)
    index = _causal_index(target_positions, season, values.size)
    return values[index]


def ar_lag_features(
    values: np.ndarray,
    target_positions: np.ndarray,
    horizon: int,
    n_lags: int = RIDGE_AR_LAGS,
) -> np.ndarray:
    """Autoregressive design matrix for horizon-``h`` forecasting.

    Row ``i`` for target position ``tau`` is the ``n_lags`` most recent inputs
    available at forecast origin ``tau - horizon``, most recent first:
    ``[values[tau-h], values[tau-h-1], ..., values[tau-h-(n_lags-1)]]``.
    """
    if n_lags < 1:
        raise ValueError("n_lags must be a positive integer")
    values = np.asarray(values, dtype=float)
    origins = np.asarray(target_positions, dtype=np.int64) - horizon
    # (m, n_lags): each origin minus 0..n_lags-1
    index = origins[:, None] - np.arange(n_lags, dtype=np.int64)[None, :]
    if index.min(initial=0) < 0:
        raise ValueError(
            "AR feature lag reaches before index 0: insufficient causal history"
        )
    if index.max(initial=0) >= values.size:
        raise ValueError("AR feature origin is outside the value series")
    return values[index]
