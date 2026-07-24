import numpy as np
from typing import Dict, Any, List
from sklearn.linear_model import Ridge

from rc_bench.core.metrics import nrmse_range, nrmse_std


def select_alpha(
    H_train: np.ndarray,
    y_train: np.ndarray,
    H_val: np.ndarray,
    y_val: np.ndarray,
    alphas: List[float],
    metric: str = "nrmse_range",
) -> Dict[str, Any]:
    """Select the readout alpha minimising ``metric`` on the validation split.

    ``metric`` is ``nrmse_range`` (legacy synthetic default) or ``nrmse_std``
    (JMLC headline, DEC-013). Both validation NRMSEs are reported for the
    selected model; ``val_score`` is the value of the selection metric.
    """
    metric_fn = nrmse_std if metric == "nrmse_std" else nrmse_range
    best_alpha = alphas[0]
    best_score = np.inf
    best_range = float("nan")
    best_std = float("nan")
    for a in alphas:
        model = Ridge(alpha=a, fit_intercept=True)
        model.fit(H_train, y_train)
        pred = model.predict(H_val)
        score = metric_fn(y_val, pred)
        if np.isfinite(score) and score < best_score:
            best_score = float(score)
            best_alpha = a
            best_range = float(nrmse_range(y_val, pred))
            best_std = float(nrmse_std(y_val, pred))
    return {
        "alpha": float(best_alpha),
        # ``val_nrmse`` keeps its historical meaning (NRMSE_range of the
        # selected model) so MetricsResult.val_nrmse_range is unchanged.
        "val_nrmse": best_range,
        "val_nrmse_std": best_std,
        "val_score": float(best_score),
        "selection_metric": metric,
    }


class RidgeReadout:
    def __init__(self, alpha: float) -> None:
        self._ridge = Ridge(alpha=alpha, fit_intercept=True)

    def fit(self, H: np.ndarray, y: np.ndarray) -> None:
        self._ridge.fit(H, y)

    def predict(self, H: np.ndarray) -> np.ndarray:
        return self._ridge.predict(H)
