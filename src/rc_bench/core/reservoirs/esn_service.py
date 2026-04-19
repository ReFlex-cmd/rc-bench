import time
import numpy as np
from typing import Dict, Any
from reservoirpy.nodes import Reservoir
from sklearn.linear_model import Ridge

from ..metrics import nrmse, mae, mse

def build_reservoir(config: Dict[str, Any]) -> Reservoir:
    """Создает резервуар, используя параметры из конфига."""
    return Reservoir(
        units=config.get("n_units", 300),
        lr=config.get("lr", 1.0),
        sr=config.get("spectral_radius", 0.9),
        input_scaling=config.get("input_scaling", 0.5),
        rc_connectivity=config.get("rc_connectivity", 0.1),
        input_connectivity=config.get("input_connectivity", 0.1),
        seed=config.get("seed", 42),
    )

def select_alpha(
    H_train: np.ndarray, y_train: np.ndarray,
    H_val: np.ndarray, y_val: np.ndarray,
    alphas: list
) -> Dict[str, Any]:
    """Подбор alpha для Ridge (как в твоем скрипте)."""
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

def run_esn_experiment(
    data: Dict[str, np.ndarray], 
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Основная точка входа. 
    Принимает данные и конфиг.
    Возвращает словарь с метриками и результатами.
    """
    
    # 1. Распаковка данных
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    # 2. Настройки
    washout = config.get("washout", 200)
    alpha_grid = config.get("ridge_alpha_grid", [0.001, 0.01, 0.1, 1.0, 10.0])
    seed = config.get("seed", 42)
    np.random.seed(seed)

    # 3. Резервуар
    t0 = time.time()
    res = build_reservoir(config)
    
    # Прогоны (States)
    H_train_full = res.run(X_train)
    H_val_full = res.run(X_val)
    H_test_full = res.run(X_test)
    time_reservoir = time.time() - t0

    # 4. Washout (отбрасываем переходный процесс)
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

    # 5. Readout (Ridge) и подбор Alpha
    t1 = time.time()
    alpha_info = select_alpha(H_train, y_train_eff, H_val, y_val_eff, alpha_grid)
    best_alpha = alpha_info["alpha"]

    # Финальное обучение на Train + Val
    H_tv = np.vstack([H_train, H_val])
    y_tv = np.concatenate([y_train_eff, y_val_eff])

    ridge = Ridge(alpha=best_alpha, fit_intercept=True)
    ridge.fit(H_tv, y_tv)
    
    # 6. Прогноз и Метрики
    y_test_pred = ridge.predict(H_test)
    time_readout = time.time() - t1

    metrics = {
        "nrmse": nrmse(y_test_eff, y_test_pred),
        "mae": mae(y_test_eff, y_test_pred),
        "mse": mse(y_test_eff, y_test_pred),
        "execution_time": time_reservoir + time_readout
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