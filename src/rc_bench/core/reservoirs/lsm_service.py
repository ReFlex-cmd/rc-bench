import numpy as np
from typing import Dict, Any

from .base import BaseReservoir


def _lif_run(
    u: np.ndarray,
    W_rec: np.ndarray,
    W_in: np.ndarray,
    v_th: float,
    v_reset: float,
    alpha_mem: float,
    alpha_syn: float,
) -> np.ndarray:
    u = u.reshape(-1)
    T = u.shape[0]
    units = W_rec.shape[0]

    v = np.zeros(units)
    s = np.zeros(units)
    H = np.zeros((T, units))

    for t in range(T):
        I = W_in[:, 0] * u[t] + W_rec @ s
        v = alpha_mem * v + (1.0 - alpha_mem) * I
        spikes = (v >= v_th).astype(float)
        v = np.where(spikes > 0.0, v_reset, v)
        s = alpha_syn * s + spikes
        H[t] = s

    return H


class LSMReservoir(BaseReservoir):
    DEFAULT_SCALER = "zscore"

    def _build(self, config: Dict[str, Any]) -> None:
        seed = config.get("seed", 42)
        units = int(config.get("units", 400))
        density = float(config.get("density", 0.05))
        w_rec_scale = float(config.get("w_rec_scale", 1.0))
        input_scale = float(config.get("input_scale", 0.5))

        rng = np.random.default_rng(seed)
        W_rec = rng.normal(0.0, 1.0, (units, units))
        W_rec *= rng.random((units, units)) < density
        W_rec *= w_rec_scale / np.sqrt(units)
        W_in = rng.uniform(-input_scale, input_scale, (units, 1))

        self._W_rec = W_rec
        self._W_in = W_in
        self._v_th = float(config.get("v_th", 1.0))
        self._v_reset = float(config.get("v_reset", 0.0))
        tau_mem = float(config.get("tau_mem", 20.0))
        tau_syn = float(config.get("tau_syn", 10.0))
        dt = float(config.get("dt", 1.0))
        # Precompute decay factors used by both transform() and step()
        self._alpha_mem = float(np.exp(-dt / tau_mem))
        self._alpha_syn = float(np.exp(-dt / tau_syn))

    def transform(self, X: np.ndarray) -> np.ndarray:
        return _lif_run(
            X, self._W_rec, self._W_in,
            self._v_th, self._v_reset,
            self._alpha_mem, self._alpha_syn,
        )

    def sanity_check(self, H: np.ndarray) -> Dict[str, bool]:
        return {"spiking_occurs": bool(H.sum() > 0)}

    # ------------------------------------------------------------------
    # Step API
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        units = self._W_rec.shape[0]
        self._step_v = np.zeros(units)
        self._step_s = np.zeros(units)

    def step(self, x_t: np.ndarray) -> np.ndarray:
        u = float(np.asarray(x_t).reshape(-1)[0])
        v, s = self._step_v, self._step_s
        I = self._W_in[:, 0] * u + self._W_rec @ s
        v = self._alpha_mem * v + (1.0 - self._alpha_mem) * I
        spikes = (v >= self._v_th).astype(float)
        v = np.where(spikes > 0.0, self._v_reset, v)
        s = self._alpha_syn * s + spikes
        self._step_v, self._step_s = v, s
        return s
