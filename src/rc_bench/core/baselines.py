"""Deterministic baseline predictors for the JMLC real-data matrix.

Persistence, seasonal persistence and the Ridge AR feature builder operate on
a single split's unshifted ``values`` array and the split-local target
positions the runner has already aligned (washout + horizon) and masked to
observed targets. All lags reach only into causal within-split history; a lag
that would read before index 0 is an explicit protocol error rather than a
silently wrapped negative NumPy index.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from rc_bench.core.metrics import nrmse_std

SEASONAL_PERIOD = 24
RIDGE_AR_LAGS = 24
RIDGE_AR_ALPHA_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)


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


@dataclass(frozen=True)
class RidgeARFit:
    """Result of a fixed-grid Ridge AR fit (DEC-013/DEC-015).

    ``scaler`` and ``final_model`` expose the fitted train-only scaler and the
    train+val model so callers can re-run a timed test prediction in isolation.
    """

    alpha: float
    val_nrmse_std: float
    candidates: tuple[tuple[float, float], ...]
    val_predictions: np.ndarray
    test_predictions: np.ndarray
    scaler: StandardScaler
    final_model: Ridge


def select_and_fit_ridge_ar(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    alpha_grid: "list[float] | tuple[float, ...]",
) -> RidgeARFit:
    """Fixed-grid Ridge AR with a train-only feature scaler (DEC-015).

    The z-score scaler is fit on train features only. For each alpha, Ridge is
    fit on train and scored on validation by NRMSE_std; the alpha with the
    smallest finite validation NRMSE_std is selected (ties resolved by grid
    order). The model is then refit on train+val with that alpha and evaluated
    once on test. Target values are never scaled and an intercept is fit.
    """
    alpha_grid = list(alpha_grid)
    if not alpha_grid:
        raise ValueError("alpha_grid must contain at least one value")

    y_train = np.asarray(y_train, dtype=float)
    y_val = np.asarray(y_val, dtype=float)

    scaler = StandardScaler().fit(np.asarray(X_train, dtype=float))
    X_train_s = scaler.transform(np.asarray(X_train, dtype=float))
    X_val_s = scaler.transform(np.asarray(X_val, dtype=float))
    X_test_s = scaler.transform(np.asarray(X_test, dtype=float))

    candidates: list[tuple[float, float]] = []
    best_alpha: float | None = None
    best_score = np.inf
    best_val_pred: np.ndarray | None = None
    for alpha in alpha_grid:
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(X_train_s, y_train)
        val_pred = model.predict(X_val_s)
        score = nrmse_std(y_val, val_pred)
        candidates.append((float(alpha), float(score)))
        if np.isfinite(score) and score < best_score:
            best_score = float(score)
            best_alpha = float(alpha)
            best_val_pred = val_pred

    if best_alpha is None or best_val_pred is None:
        raise ValueError(
            "Ridge AR selection failed: no finite validation NRMSE_std; "
            "degenerate validation targets"
        )

    X_tv_s = np.vstack([X_train_s, X_val_s])
    y_tv = np.concatenate([y_train, y_val])
    final_model = Ridge(alpha=best_alpha, fit_intercept=True)
    final_model.fit(X_tv_s, y_tv)
    test_pred = final_model.predict(X_test_s)

    return RidgeARFit(
        alpha=best_alpha,
        val_nrmse_std=best_score,
        candidates=tuple(candidates),
        val_predictions=best_val_pred,
        test_predictions=test_pred,
        scaler=scaler,
        final_model=final_model,
    )
