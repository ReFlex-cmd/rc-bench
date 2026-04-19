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


def init_lsm_reservoir(
    units: int,
    density: float,
    w_rec_scale: float,
    input_scale: float,
    seed: int,
):
    rng = np.random.default_rng(seed)

    W_rec = rng.normal(0.0, 1.0, size=(units, units))
    mask = rng.random((units, units)) < density
    W_rec *= mask
    W_rec *= w_rec_scale / np.sqrt(units)

    W_in = rng.uniform(-input_scale, input_scale, size=(units, 1))
    return W_rec, W_in


def lif_lsm_run(
    u: np.ndarray,
    W_rec: np.ndarray,
    W_in: np.ndarray,
    v_th: float,
    v_reset: float,
    tau_mem: float,
    tau_syn: float,
    dt: float,
) -> np.ndarray:
    """Запуск LIF-резервуара в стиле LSM."""
    u = u.reshape(-1)
    T = u.shape[0]
    units = W_rec.shape[0]

    v = np.zeros(units, dtype=float)
    s = np.zeros(units, dtype=float)

    alpha_mem = np.exp(-dt / tau_mem)
    alpha_syn = np.exp(-dt / tau_syn)

    H = np.zeros((T, units), dtype=float)

    for t in range(T):
        I_in = W_in[:, 0] * u[t]
        I_rec = W_rec @ s
        I = I_in + I_rec

        v = alpha_mem * v + (1.0 - alpha_mem) * I
        spikes = (v >= v_th).astype(float)
        v = np.where(spikes > 0.0, v_reset, v)
        s = alpha_syn * s + spikes

        H[t] = s

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


def run_lsm_experiment(
    data: Dict[str, np.ndarray],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Точка входа для LSM-эксперимента."""

    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    washout = config.get("washout", 200)
    alpha_grid = config.get("ridge_alpha_grid", [0.001, 0.01, 0.1, 1.0, 10.0])
    seed = config.get("seed", 42)
    scaler_name = config.get("scaler", "zscore")
    np.random.seed(seed)

    # Гиперпараметры LSM
    units = int(config.get("units", 400))
    density = float(config.get("density", 0.05))
    w_rec_scale = float(config.get("w_rec_scale", 1.0))
    input_scale = float(config.get("input_scale", 0.5))
    v_th = float(config.get("v_th", 1.0))
    v_reset = float(config.get("v_reset", 0.0))
    tau_mem = float(config.get("tau_mem", 20.0))
    tau_syn = float(config.get("tau_syn", 10.0))
    dt = float(config.get("dt", 1.0))

    # Масштабирование входов
    X_train_s, X_val_s, X_test_s = scale_inputs(X_train, X_val, X_test, scaler_name)

    # Резервуар
    W_rec, W_in = init_lsm_reservoir(units, density, w_rec_scale, input_scale, seed)

    t0 = time.time()
    H_train_full = lif_lsm_run(X_train_s, W_rec, W_in, v_th, v_reset, tau_mem, tau_syn, dt)
    H_val_full = lif_lsm_run(X_val_s, W_rec, W_in, v_th, v_reset, tau_mem, tau_syn, dt)
    H_test_full = lif_lsm_run(X_test_s, W_rec, W_in, v_th, v_reset, tau_mem, tau_syn, dt)
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
