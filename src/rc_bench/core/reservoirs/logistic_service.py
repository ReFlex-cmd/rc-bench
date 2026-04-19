import time
import numpy as np
from typing import Dict, Any
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from ..metrics import nrmse, mae, mse


def scale_inputs(X_train, X_val, X_test, scaler_name: str):
    if scaler_name.lower() == "zscore":
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)
        return X_train_s, X_val_s, X_test_s
    return X_train, X_val, X_test


def init_logistic_params(
    units: int,
    r_min: float,
    r_max: float,
    input_scale: float,
    seed: int,
):
    rng = np.random.default_rng(seed)
    r = rng.uniform(r_min, r_max, size=units)
    w_in = rng.uniform(-input_scale, input_scale, size=units)
    x0 = rng.uniform(0.1, 0.9, size=units)
    return r, w_in, x0


def logistic_reservoir_run(
    u: np.ndarray,
    r: np.ndarray,
    w_in: np.ndarray,
    x0: np.ndarray,
    coupling: float,
) -> np.ndarray:
    """Дискретный хаотический резервуар на логистическом отображении."""
    u = u.reshape(-1)
    T = u.shape[0]
    units = r.shape[0]

    x = x0.copy()
    H = np.zeros((T, units), dtype=float)

    for t in range(T):
        x_log = r * x * (1.0 - x)
        x = (1.0 - coupling) * x_log + coupling * (w_in * u[t])
        x = np.clip(x, 1e-6, 1.0 - 1e-6)
        H[t] = x

    return H


def select_alpha(
    H_train: np.ndarray, y_train: np.ndarray,
    H_val: np.ndarray, y_val: np.ndarray,
    alphas: list,
) -> Dict[str, Any]:
    best_alpha = None
    best_nrmse = np.inf

    for a in alphas:
        model = Ridge(alpha=a, fit_intercept=True)
        model.fit(H_train, y_train)
        y_val_pred = model.predict(H_val)
        score = nrmse(y_val, y_val_pred)
        if score < best_nrmse:
            best_nrmse = score
            best_alpha = a

    return {"alpha": float(best_alpha), "val_nrmse": float(best_nrmse)}


def run_logistic_experiment(
    data: Dict[str, np.ndarray],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Точка входа для Logistic-эксперимента."""

    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    washout = config.get("washout", 200)
    alpha_grid = config.get("ridge_alpha_grid", [0.001, 0.01, 0.1, 1.0, 10.0])
    seed = config.get("seed", 42)
    scaler_name = config.get("scaler", "zscore")
    np.random.seed(seed)

    # Гиперпараметры логистического резервуара
    units = int(config.get("units", 500))
    r_min = float(config.get("r_min", 3.8))
    r_max = float(config.get("r_max", 4.0))
    input_scale = float(config.get("input_scale", 0.05))
    coupling = float(config.get("alpha", 0.1))

    # Масштабирование входов
    X_train_s, X_val_s, X_test_s = scale_inputs(X_train, X_val, X_test, scaler_name)

    # Резервуар
    r, w_in, x0 = init_logistic_params(units, r_min, r_max, input_scale, seed)

    t0 = time.time()
    H_train_full = logistic_reservoir_run(X_train_s, r, w_in, x0, coupling)
    H_val_full = logistic_reservoir_run(X_val_s, r, w_in, x0, coupling)
    H_test_full = logistic_reservoir_run(X_test_s, r, w_in, x0, coupling)
    time_reservoir = time.time() - t0

    # Washout
    if washout > 0:
        H_train = H_train_full[washout:]
        y_train_eff = y_train[washout:]
        H_val = H_val_full[washout:]
        y_val_eff = y_val[washout:]
        H_test = H_test_full[washout:]
        y_test_eff = y_test[washout:]
    else:
        H_train, y_train_eff = H_train_full, y_train
        H_val, y_val_eff = H_val_full, y_val
        H_test, y_test_eff = H_test_full, y_test

    # Readout
    t1 = time.time()
    alpha_info = select_alpha(H_train, y_train_eff, H_val, y_val_eff, alpha_grid)
    best_alpha = alpha_info["alpha"]

    H_tv = np.vstack([H_train, H_val])
    y_tv = np.concatenate([y_train_eff, y_val_eff])

    ridge = Ridge(alpha=best_alpha, fit_intercept=True)
    ridge.fit(H_tv, y_tv)

    y_test_pred = ridge.predict(H_test)
    time_readout = time.time() - t1

    metrics = {
        "nrmse": nrmse(y_test_eff, y_test_pred),
        "mae": mae(y_test_eff, y_test_pred),
        "mse": mse(y_test_eff, y_test_pred),
        "execution_time": time_reservoir + time_readout,
    }

    return {
        "metrics": metrics,
        "meta": {
            "best_alpha": best_alpha,
            "val_nrmse": alpha_info["val_nrmse"],
            "reservoir_params": config,
        },
        "preds": y_test_pred,
        "y_test": y_test_eff,
    }
