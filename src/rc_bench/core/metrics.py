import numpy as np

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean((y_true - y_pred) ** 2))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))

def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Normalized RMSE: RMSE / (max(y_true) - min(y_true))
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rmse = np.sqrt(mse(y_true, y_pred))
    denom = float(np.max(y_true) - np.min(y_true))
    if denom == 0:
        return float("nan")
    return float(rmse / denom)