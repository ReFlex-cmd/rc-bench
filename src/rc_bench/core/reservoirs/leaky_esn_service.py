import numpy as np
from typing import Dict, Any

from .base import BaseReservoir


class LeakyESNReservoir(BaseReservoir):
    """Leaky Echo State Network (pure numpy, single layer).

    Update rule:
        x(t) = (1 - alpha) * x(t-1) + alpha * tanh(W_in * u(t) + W_rec @ x(t-1))

    Differs from the reservoirpy ESN wrapper: no external dependency,
    leaking rate ``alpha`` is the primary memory-control parameter.
    """

    DEFAULT_SCALER = "none"

    def _build(self, config: Dict[str, Any]) -> None:
        units = int(config.get("units", 300))
        sr = float(config.get("sr", 0.9))
        self._leak_rate = float(config.get("leak_rate", 0.3))
        input_scaling = float(config.get("input_scaling", 0.5))
        density = float(config.get("density", 0.1))
        seed = config.get("seed", 42)

        self._units = units
        rng = np.random.default_rng(seed)

        self._W_in = rng.uniform(-input_scaling, input_scaling, (units, 1))

        W_rec = rng.normal(0.0, 1.0, (units, units))
        W_rec *= rng.random((units, units)) < density
        max_ev = np.max(np.abs(np.linalg.eigvals(W_rec)))
        if max_ev > 1e-8:
            W_rec = W_rec / max_ev * sr
        self._W_rec = W_rec

    def transform(self, X: np.ndarray) -> np.ndarray:
        u = X.reshape(-1)
        T = len(u)
        x = np.zeros(self._units)
        H = np.zeros((T, self._units))
        alpha = self._leak_rate
        for t in range(T):
            pre = np.tanh(self._W_in[:, 0] * u[t] + self._W_rec @ x)
            x = (1.0 - alpha) * x + alpha * pre
            H[t] = x
        return H

    def sanity_check(self, H: np.ndarray) -> Dict[str, bool]:
        return {
            "bounded_states": bool(np.max(np.abs(H)) < 1e6),
            "non_trivial_variance": bool(H.std() > 1e-6),
        }

    # ------------------------------------------------------------------
    # Step API
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        self._step_x = np.zeros(self._units)

    def step(self, x_t: np.ndarray) -> np.ndarray:
        u = float(np.asarray(x_t).reshape(-1)[0])
        alpha = self._leak_rate
        pre = np.tanh(self._W_in[:, 0] * u + self._W_rec @ self._step_x)
        self._step_x = (1.0 - alpha) * self._step_x + alpha * pre
        return self._step_x

    def step_operation_counts(self) -> Dict[str, int]:
        nnz = int(np.count_nonzero(self._W_rec))
        return {
            # step(): pre = tanh(W_in*u + W_rec@x); x = (1-a)*x + a*pre.
            #   W_rec @ x       -> nnz MAC (one per nonzero weight)
            #   W_in * u        -> units MAC (scalar-vector multiply)
            #   (1-a)*x + a*pre -> 2*units MAC (two scalar-vector multiplies;
            #                      the final "+" is not counted separately,
            #                      see activity.py's MAC-counting convention)
            "reservoir_macs": nnz + 3 * self._units,
            "reservoir_nonlinearities": self._units,
            "reservoir_nonzero_recurrent_weights": nnz,
        }
