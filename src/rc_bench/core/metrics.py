from __future__ import annotations

import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def nrmse_range(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE / (max - min) of y_true."""
    y_true = np.asarray(y_true, dtype=float)
    denom = float(np.max(y_true) - np.min(y_true))
    if denom == 0:
        return float("nan")
    return float(rmse(y_true, y_pred) / denom)


def nrmse_std(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE / std(y_true)."""
    y_true = np.asarray(y_true, dtype=float)
    std = float(np.std(y_true))
    if std == 0:
        return float("nan")
    return float(rmse(y_true, y_pred) / std)


def nrmse_var(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MSE / var(y_true). Equals 1.0 for a constant (mean) predictor."""
    y_true = np.asarray(y_true, dtype=float)
    var = float(np.var(y_true))
    if var == 0:
        return float("nan")
    return float(mse(y_true, y_pred) / var)


def prediction_horizon(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.3,
) -> int:
    """Steps until |y_true[t] - y_pred[t]| / std(y_true) first exceeds threshold."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    std = float(np.std(y_true))
    if std == 0:
        return 0
    err = np.abs(y_true - y_pred) / std
    exceeded = np.where(err > threshold)[0]
    return int(exceeded[0]) if len(exceeded) > 0 else len(y_true)


# ---------------------------------------------------------------------------
# JMLC baseline-relative quality metrics (DEC-013)
# ---------------------------------------------------------------------------

def seasonal_naive_mae_scale(
    y: np.ndarray,
    season: int = 24,
    observed_mask: np.ndarray | None = None,
) -> tuple[float, int]:
    """Train-only seasonal-naive MAE scale for MASE.

    Returns ``(scale, n_terms)`` where ``scale`` is the mean of
    ``|y[t] - y[t - season]|`` over valid target indices and ``n_terms`` is the
    number of pairs averaged. When ``observed_mask`` is given, a pair counts
    only if both its endpoints are observed. An empty pair set or a zero scale
    is an explicit protocol error (DEC-013), never a NaN/Infinity.
    """
    y = np.asarray(y, dtype=float)
    if isinstance(season, bool) or not isinstance(season, (int, np.integer)) or season < 1:
        raise ValueError("season must be a positive integer")
    if y.ndim != 1:
        raise ValueError("y must be one-dimensional")
    if y.size <= season:
        raise ValueError(
            f"no seasonal pairs: series length {y.size} <= season {season}"
        )
    diffs = np.abs(y[season:] - y[:-season])
    if observed_mask is not None:
        observed_mask = np.asarray(observed_mask, dtype=bool)
        if observed_mask.shape != y.shape:
            raise ValueError("observed_mask must match y shape")
        valid = observed_mask[season:] & observed_mask[:-season]
        diffs = diffs[valid]
    n_terms = int(diffs.size)
    if n_terms == 0:
        raise ValueError("no seasonal pairs with both endpoints observed")
    scale = float(np.mean(diffs))
    if not np.isfinite(scale) or scale == 0.0:
        raise ValueError("seasonal-naive MAE scale is zero; degenerate series")
    return scale, n_terms


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    seasonal_scale: float,
) -> float:
    """Mean Absolute Scaled Error against a precomputed seasonal-naive scale."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        raise ValueError("cannot compute MASE on an empty target set")
    if not np.isfinite(seasonal_scale) or seasonal_scale <= 0.0:
        raise ValueError("seasonal_scale must be a positive finite value")
    return float(mae(y_true, y_pred) / seasonal_scale)


def mae_skill(mae_model: float, mae_reference: float) -> float:
    """Skill of a model's MAE relative to a reference: ``1 - MAE_model / MAE_ref``.

    The reference is seasonal persistence evaluated on the same target set. A
    zero (or non-finite) reference MAE is a protocol error, not an Infinity.
    """
    mae_model = float(mae_model)
    mae_reference = float(mae_reference)
    if not np.isfinite(mae_reference) or mae_reference <= 0.0:
        raise ValueError("reference MAE must be a positive finite value")
    return 1.0 - mae_model / mae_reference
