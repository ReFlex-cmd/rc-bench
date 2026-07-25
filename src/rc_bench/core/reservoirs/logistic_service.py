"""Logistic-map reservoir.

Each node is a logistic map x_i(t+1) = r_i * x_i(t) * (1 - x_i(t)) with its
own r_i ∈ [r_min, r_max], driven by an additive input mix:

    x_i(t+1) = r_i * x_i(t) * (1 - x_i(t)) + coupling * w_in_i * u(t)

Per ТЗ §1.6, this is the "additive input mixing" form (rather than convex
combination) — the logistic dynamics are not displaced by the input, only
biased. After update, x is clipped to [0, 1] to keep the next-step logistic
map well-defined; we use a hard clip rather than an ε-margin to avoid
distorting the chaotic regime near the boundary.

Different ``r_i`` per node guarantees computational diversity even without
recurrent inter-node coupling.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict

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
        x = r * x * (1.0 - x) + coupling * (w_in * u[t])
        x = np.clip(x, 0.0, 1.0)
        H[t] = x

    return H


class LogisticReservoir(BaseReservoir):
    DEFAULT_SCALER = "zscore"

    def _build(self, config: Dict[str, Any]) -> None:
        seed = config.get("seed", 42)
        units = int(config.get("units", 500))
        r_min = float(config.get("r_min", 3.8))
        r_max = float(config.get("r_max", 4.0))
        if r_max < r_min:
            r_max = r_min
        input_scale = float(config.get("input_scale", 0.05))

        rng = np.random.default_rng(seed)
        self._r = rng.uniform(r_min, r_max, units)
        self._w_in = rng.uniform(-input_scale, input_scale, units)
        self._x0 = rng.uniform(0.1, 0.9, units)
        # `coupling` is the input-injection strength ε in the additive form.
        # Backward-compat alias `alpha` kept for legacy configs.
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
        x = self._r * x * (1.0 - x) + self._coupling * (self._w_in * u)
        x = np.clip(x, 0.0, 1.0)
        self._step_x = x
        return x

    def step_operation_counts(self) -> Dict[str, int]:
        units = int(self._r.shape[0])
        return {
            # r*x*(1-x): 2 MAC на узел; + coupling*(w_in*u): ещё 1.
            "reservoir_macs": 3 * units,
            "reservoir_nonlinearities": 0,     # логистическое отображение — сама арифметика
            "reservoir_nonzero_recurrent_weights": 0,   # межузловых связей нет
        }
