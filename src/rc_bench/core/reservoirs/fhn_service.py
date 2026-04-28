import numpy as np
from typing import Dict, Any

from .base import BaseReservoir


def _fhn_deriv(v, w, I_ext, a, b, tau):
    dv = v - (v ** 3) / 3.0 - w + I_ext
    dw = (v + a - b * w) / tau
    return dv, dw


def _fhn_run_rk4(
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
    u = u.reshape(-1)
    T = u.shape[0]
    v, w = v0.copy(), w0.copy()
    H = np.zeros((T, v0.shape[0]))

    for t in range(T):
        I_ext = W_in[:, 0] * u[t] + W_rec @ v
        for _ in range(internal_steps):
            k1v, k1w = _fhn_deriv(v, w, I_ext, a, b, tau)
            k2v, k2w = _fhn_deriv(v + dt / 2 * k1v, w + dt / 2 * k1w, I_ext, a, b, tau)
            k3v, k3w = _fhn_deriv(v + dt / 2 * k2v, w + dt / 2 * k2w, I_ext, a, b, tau)
            k4v, k4w = _fhn_deriv(v + dt * k3v, w + dt * k3w, I_ext, a, b, tau)
            v = v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
            w = w + dt / 6 * (k1w + 2 * k2w + 2 * k3w + k4w)
        H[t] = v

    return H


class FHNReservoir(BaseReservoir):
    DEFAULT_SCALER = "zscore"

    def _build(self, config: Dict[str, Any]) -> None:
        seed = config.get("seed", 42)
        units = int(config.get("units", 200))
        density = float(config.get("density", 0.1))
        sr = float(config.get("sr", 0.9))
        input_scale = float(config.get("input_scale", 0.5))

        rng = np.random.default_rng(seed)
        v0 = rng.uniform(-1.0, 1.0, units)
        w0 = rng.uniform(-1.0, 1.0, units)
        W_rec = rng.normal(0.0, 1.0, (units, units))
        W_rec *= rng.random((units, units)) < density
        max_ev = np.max(np.abs(np.linalg.eigvals(W_rec))) + 1e-8
        W_rec = W_rec / max_ev * sr
        W_in = rng.uniform(-input_scale, input_scale, (units, 1))

        self._v0, self._w0 = v0, w0
        self._W_rec, self._W_in = W_rec, W_in
        self._a = float(config.get("a", 0.7))
        self._b = float(config.get("b", 0.8))
        self._tau = float(config.get("tau", 12.5))
        self._dt = float(config.get("dt", 0.1))
        self._internal_steps = int(config.get("internal_steps", 2))

    def transform(self, X: np.ndarray) -> np.ndarray:
        return _fhn_run_rk4(
            X, self._v0, self._w0,
            self._W_rec, self._W_in,
            self._a, self._b, self._tau,
            self._dt, self._internal_steps,
        )

    def sanity_check(self, H: np.ndarray) -> Dict[str, bool]:
        return {"bounded_oscillation": bool(np.max(np.abs(H)) < 1e6)}

    # ------------------------------------------------------------------
    # Step API
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        self._step_v = self._v0.copy()
        self._step_w = self._w0.copy()

    def step(self, x_t: np.ndarray) -> np.ndarray:
        u = float(np.asarray(x_t).reshape(-1)[0])
        v, w = self._step_v, self._step_w
        I_ext = self._W_in[:, 0] * u + self._W_rec @ v
        dt, a, b, tau = self._dt, self._a, self._b, self._tau
        for _ in range(self._internal_steps):
            k1v, k1w = _fhn_deriv(v, w, I_ext, a, b, tau)
            k2v, k2w = _fhn_deriv(v + dt / 2 * k1v, w + dt / 2 * k1w, I_ext, a, b, tau)
            k3v, k3w = _fhn_deriv(v + dt / 2 * k2v, w + dt / 2 * k2w, I_ext, a, b, tau)
            k4v, k4w = _fhn_deriv(v + dt * k3v, w + dt * k3w, I_ext, a, b, tau)
            v = v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
            w = w + dt / 6 * (k1w + 2 * k2w + 2 * k3w + k4w)
        self._step_v, self._step_w = v, w
        return v
