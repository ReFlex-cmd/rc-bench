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
