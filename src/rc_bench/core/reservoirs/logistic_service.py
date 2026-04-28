import numpy as np
from typing import Dict, Any

from .base import BaseReservoir


def _logistic_run(
    u: np.ndarray,
    r: np.ndarray,
    w_in: np.ndarray,
    x0: np.ndarray,
    coupling: float,
) -> np.ndarray:
    u = u.reshape(-1)
    T = u.shape[0]
    x = x0.copy()
    H = np.zeros((T, r.shape[0]))

    for t in range(T):
        x_log = r * x * (1.0 - x)
        x = (1.0 - coupling) * x_log + coupling * (w_in * u[t])
        x = np.clip(x, 1e-6, 1.0 - 1e-6)
        H[t] = x

    return H


class LogisticReservoir(BaseReservoir):
    DEFAULT_SCALER = "zscore"

    def _build(self, config: Dict[str, Any]) -> None:
        seed = config.get("seed", 42)
        units = int(config.get("units", 500))
        r_min = float(config.get("r_min", 3.8))
        r_max = float(config.get("r_max", 4.0))
        input_scale = float(config.get("input_scale", 0.05))

        rng = np.random.default_rng(seed)
        self._r = rng.uniform(r_min, r_max, units)
        self._w_in = rng.uniform(-input_scale, input_scale, units)
        self._x0 = rng.uniform(0.1, 0.9, units)
        # "coupling" is the canonical key; "alpha" kept for backward compatibility
        self._coupling = float(config.get("coupling", config.get("alpha", 0.1)))

    def transform(self, X: np.ndarray) -> np.ndarray:
        return _logistic_run(X, self._r, self._w_in, self._x0, self._coupling)

    def sanity_check(self, H: np.ndarray) -> Dict[str, bool]:
        return {"chaotic_regime": bool(H.var(axis=0).mean() > 1e-4)}

    # ------------------------------------------------------------------
    # Step API
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        self._step_x = self._x0.copy()

    def step(self, x_t: np.ndarray) -> np.ndarray:
        u = float(np.asarray(x_t).reshape(-1)[0])
        x = self._step_x
        x_log = self._r * x * (1.0 - x)
        x = (1.0 - self._coupling) * x_log + self._coupling * (self._w_in * u)
        x = np.clip(x, 1e-6, 1.0 - 1e-6)
        self._step_x = x
        return x
