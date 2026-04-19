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


def init_fhn_reservoir(
    units: int,
    density: float,
    sr: float,
    input_scale: float,
    seed: int,
):
    rng = np.random.default_rng(seed)

    v0 = rng.uniform(-1.0, 1.0, size=units)
    w0 = rng.uniform(-1.0, 1.0, size=units)

    W_rec = rng.normal(0.0, 1.0, size=(units, units))
    mask = rng.random((units, units)) < density
    W_rec *= mask

    eigvals = np.linalg.eigvals(W_rec)
    max_ev = np.max(np.abs(eigvals)) + 1e-8
    W_rec = W_rec / max_ev * sr

    W_in = rng.uniform(-input_scale, input_scale, size=(units, 1))
    return v0, w0, W_rec, W_in


def fhn_step(
    v: np.ndarray,
    w: np.ndarray,
    I_ext: np.ndarray,
    a: float,
    b: float,
    tau: float,
    dt: float,
) -> tuple:
    """Один шаг интегрирования ФитцХью–Нагумо (Эйлер)."""
    dv = v - (v ** 3) / 3.0 - w + I_ext
    dw = (v + a - b * w) / tau
    return v + dt * dv, w + dt * dw


def fhn_reservoir_run(
    u: np.ndarray,
    v0: np.ndarray,
    w0: np.ndarray,
    W_rec: np.ndarray,
    W_in: np.ndarray,
    a: float,
    b: float,
    tau: float,
    dt: float,
    internal_steps: int,
) -> np.ndarray:
    """Запуск FHN-резервуара. Выход: H [T, units]."""
    u = u.reshape(-1)
    T = u.shape[0]

    v = v0.copy()
    w = w0.copy()
    H = np.zeros((T, v0.shape[0]), dtype=float)

    for t in range(T):
        I_in = W_in[:, 0] * u[t]
        I_rec = W_rec @ v
        I_ext = I_in + I_rec

        for _ in range(internal_steps):
            v, w = fhn_step(v, w, I_ext, a, b, tau, dt)

        H[t] = v

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


def run_fhn_experiment(
    data: Dict[str, np.ndarray],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Точка входа для FHN-эксперимента."""

    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    washout = config.get("washout", 200)
    alpha_grid = config.get("ridge_alpha_grid", [0.001, 0.01, 0.1, 1.0, 10.0])
    seed = config.get("seed", 42)
    scaler_name = config.get("scaler", "zscore")
    np.random.seed(seed)

    # Гиперпараметры FHN
    units = int(config.get("units", 200))
    density = float(config.get("density", 0.1))
    sr = float(config.get("sr", 0.9))
    input_scale = float(config.get("input_scale", 0.5))
    a = float(config.get("a", 0.7))
    b = float(config.get("b", 0.8))
    tau = float(config.get("tau", 12.5))
    dt = float(config.get("dt", 0.1))
    internal_steps = int(config.get("internal_steps", 2))

    # Масштабирование входов
    X_train_s, X_val_s, X_test_s = scale_inputs(X_train, X_val, X_test, scaler_name)

    # Резервуар
    v0, w0, W_rec, W_in = init_fhn_reservoir(units, density, sr, input_scale, seed)

    t0 = time.time()
    H_train_full = fhn_reservoir_run(X_train_s, v0, w0, W_rec, W_in, a, b, tau, dt, internal_steps)
    H_val_full = fhn_reservoir_run(X_val_s, v0, w0, W_rec, W_in, a, b, tau, dt, internal_steps)
    H_test_full = fhn_reservoir_run(X_test_s, v0, w0, W_rec, W_in, a, b, tau, dt, internal_steps)
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
