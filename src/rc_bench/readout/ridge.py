import numpy as np
from typing import Dict, Any, List
from sklearn.linear_model import Ridge

from rc_bench.core.metrics import nrmse_range


def select_alpha(
    H_train: np.ndarray,
    y_train: np.ndarray,
    H_val: np.ndarray,
    y_val: np.ndarray,
    alphas: List[float],
) -> Dict[str, Any]:
    best_alpha = alphas[0]
    best_score = np.inf
    for a in alphas:
        model = Ridge(alpha=a, fit_intercept=True)
        model.fit(H_train, y_train)
        score = nrmse_range(y_val, model.predict(H_val))
        if score < best_score:
            best_score = score
            best_alpha = a
    return {"alpha": float(best_alpha), "val_nrmse": float(best_score)}


class RidgeReadout:
    def __init__(self, alpha: float) -> None:
        self._ridge = Ridge(alpha=alpha, fit_intercept=True)

    def fit(self, H: np.ndarray, y: np.ndarray) -> None:
        self._ridge.fit(H, y)

    def predict(self, H: np.ndarray) -> np.ndarray:
        return self._ridge.predict(H)
