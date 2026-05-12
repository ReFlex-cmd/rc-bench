"""FitzHugh-Nagumo network reservoir.

Two-variable per node: fast voltage `v` and slow recovery `w`:
    dv/dt = v - v^3 / 3 - w + I_ext
    dw/dt = epsilon * (v + a - b * w)

Network coupling: I_ext_i = W_in_i * u(t) + (W_rec @ v)_i
where W_rec is sparse, scaled to spectral radius `coupling_strength`.

Output (reservoir features): v_i at each time step (fast variable).

Integrator: classical RK4. Per ТЗ §1.5, the time step Δt is fixed at 0.01 in
non-dimensional units and is *not* a hyperparameter — coarser dt produces
numerical artifacts (blow-up of the v^3 term).

Defaults follow the classical excitable regime (FitzHugh 1961, Nagumo 1962):
``a=0.7, b=0.8, epsilon=0.08``.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict

from .base import BaseReservoir


def _fhn_deriv(v, w, I_ext, a, b, epsilon):
    dv = v - (v ** 3) / 3.0 - w + I_ext
    dw = epsilon * (v + a - b * w)
    return dv, dw


def _fhn_rk4_step(v, w, I_ext, a, b, epsilon, dt, internal_steps):
    for _ in range(internal_steps):
        k1v, k1w = _fhn_deriv(v, w, I_ext, a, b, epsilon)
        k2v, k2w = _fhn_deriv(v + dt / 2 * k1v, w + dt / 2 * k1w, I_ext, a, b, epsilon)
        k3v, k3w = _fhn_deriv(v + dt / 2 * k2v, w + dt / 2 * k2w, I_ext, a, b, epsilon)
        k4v, k4w = _fhn_deriv(v + dt * k3v, w + dt * k3w, I_ext, a, b, epsilon)
        v = v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        w = w + dt / 6 * (k1w + 2 * k2w + 2 * k3w + k4w)
    return v, w


def _fhn_run_rk4(
    u: np.ndarray,
    v0: np.ndarray,
    w0: np.ndarray,
    W_rec: np.ndarray,
    W_in: np.ndarray,
    a: float,
    b: float,
    epsilon: float,
    dt: float,
    internal_steps: int,
) -> np.ndarray:
    u = u.reshape(-1)
    T = u.shape[0]
    v, w = v0.copy(), w0.copy()
    H = np.zeros((T, v0.shape[0]))

    for t in range(T):
        I_ext = W_in[:, 0] * u[t] + W_rec @ v
        v, w = _fhn_rk4_step(v, w, I_ext, a, b, epsilon, dt, internal_steps)
        H[t] = v

    return H


class FHNReservoir(BaseReservoir):
    DEFAULT_SCALER = "zscore"

    def _build(self, config: Dict[str, Any]) -> None:
        seed = config.get("seed", 42)
        units = int(config.get("units", 200))
        density = float(config.get("density", 0.1))
        # `coupling_strength` is the canonical name (= spectral radius of W_rec).
        # `sr` accepted for backward compatibility.
        coupling_strength = float(
            config.get("coupling_strength", config.get("sr", 0.9))
        )
        input_scale = float(config.get("input_scale", 0.5))

        rng = np.random.default_rng(seed)
        v0 = rng.uniform(-1.0, 1.0, units)
        w0 = rng.uniform(-1.0, 1.0, units)
        W_rec = rng.normal(0.0, 1.0, (units, units))
        W_rec *= rng.random((units, units)) < density
        max_ev = np.max(np.abs(np.linalg.eigvals(W_rec))) + 1e-8
        W_rec = W_rec / max_ev * coupling_strength
        W_in = rng.uniform(-input_scale, input_scale, (units, 1))

        self._v0, self._w0 = v0, w0
        self._W_rec, self._W_in = W_rec, W_in
        self._a = float(config.get("a", 0.7))
        self._b = float(config.get("b", 0.8))
        # Accept either `epsilon` (preferred) or legacy `tau` (= 1/epsilon).
        if "epsilon" in config:
            self._epsilon = float(config["epsilon"])
        elif "tau" in config:
            self._epsilon = 1.0 / float(config["tau"])
        else:
            self._epsilon = 0.08
        # Fixed integrator step: ТЗ §1.5 — must be ≤ 0.01, NOT a hyperparameter.
        # Reading from config is allowed only for tests; HPO must not tune it.
        self._dt = float(config.get("dt", 0.01))
        self._internal_steps = int(config.get("internal_steps", 1))

    def transform(self, X: np.ndarray) -> np.ndarray:
        return _fhn_run_rk4(
            X, self._v0, self._w0,
            self._W_rec, self._W_in,
            self._a, self._b, self._epsilon,
            self._dt, self._internal_steps,
        )

    def sanity_check(self, H: np.ndarray) -> Dict[str, bool]:
        return {
            "bounded_oscillation": bool(np.max(np.abs(H)) < 1e6),
            "no_nan": bool(np.all(np.isfinite(H))),
        }

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
        v, w = _fhn_rk4_step(
            v, w, I_ext,
            self._a, self._b, self._epsilon,
            self._dt, self._internal_steps,
        )
        self._step_v, self._step_w = v, w
        return v
