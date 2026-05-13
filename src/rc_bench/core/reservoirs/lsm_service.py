"""Liquid State Machine reservoir (LIF neurons with refractory period).

Per ТЗ §1.4: LIF dynamics with absolute refractory period. Membrane potential
integrated via exact solution between events:
    v(t+dt) = alpha_mem * v(t) + (1 - alpha_mem) * I(t),  alpha_mem = exp(-dt/tau_mem)
On spike (v ≥ v_th): emit spike, reset v ← v_reset, enter refractory for
``refractory_steps`` steps during which v is held at v_reset and no spike fires.

Synaptic trace s(t) (low-pass filter of spikes) is the readout feature:
    s(t+dt) = alpha_syn * s(t) + spikes,  alpha_syn = exp(-dt/tau_syn)

Input: continuous u(t) injected as direct current via W_in. (Direct-current
encoding rather than Poisson — a methodological choice; see audit/notes_lsm.md
when written.)
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict

from .base import BaseReservoir


def _lif_run(
    u: np.ndarray,
    W_rec: np.ndarray,
    W_in: np.ndarray,
    v_th: float,
    v_reset: float,
    alpha_mem: float,
    alpha_syn: float,
    refractory_steps: int,
) -> np.ndarray:
    u = u.reshape(-1)
    T = u.shape[0]
    units = W_rec.shape[0]

    v = np.zeros(units)
    s = np.zeros(units)
    refrac = np.zeros(units, dtype=np.int32)
    H = np.zeros((T, units))

    for t in range(T):
        I = W_in[:, 0] * u[t] + W_rec @ s
        not_refrac = refrac == 0
        # Only neurons outside refractory window integrate input
        v = np.where(not_refrac, alpha_mem * v + (1.0 - alpha_mem) * I, v_reset)
        spikes = np.logical_and(v >= v_th, not_refrac).astype(float)
        # Reset spiking neurons and start their refractory countdown
        v = np.where(spikes > 0.0, v_reset, v)
        refrac = np.where(spikes > 0.0, refractory_steps, np.maximum(refrac - 1, 0))
        s = alpha_syn * s + spikes
        H[t] = s

    return H


class LSMReservoir(BaseReservoir):
    DEFAULT_SCALER = "zscore"

    def _build(self, config: Dict[str, Any]) -> None:
        seed = config.get("seed", 42)
        units = int(config.get("units", 400))
        density = float(config.get("density", 0.05))
        # Defaults sized so membrane reaches v_th=1.0 with typical z-scored inputs;
        # see audit/_diag_seed.py — older defaults (0.5, 1.0) gave near-zero spiking.
        w_rec_scale = float(config.get("w_rec_scale", 5.0))
        input_scale = float(config.get("input_scale", 3.0))

        rng = np.random.default_rng(seed)
        W_rec = rng.normal(0.0, 1.0, (units, units))
        W_rec *= rng.random((units, units)) < density
        W_rec *= w_rec_scale / np.sqrt(units)
        W_in = rng.uniform(-input_scale, input_scale, (units, 1))

        self._W_rec = W_rec
        self._W_in = W_in
        self._v_th = float(config.get("v_th", 0.5))
        self._v_reset = float(config.get("v_reset", 0.0))
        tau_mem = float(config.get("tau_mem", 20.0))
        tau_syn = float(config.get("tau_syn", 10.0))
        dt = float(config.get("dt", 1.0))
        # Absolute refractory period in milliseconds → discrete simulation steps.
        # Default 2 ms ≈ 2 steps at dt=1 ms (typical biophysical value).
        t_refractory = float(config.get("t_refractory", 2.0))
        self._refractory_steps = max(1, int(round(t_refractory / dt)))
        # Precompute decay factors used by both transform() and step()
        self._alpha_mem = float(np.exp(-dt / tau_mem))
        self._alpha_syn = float(np.exp(-dt / tau_syn))

    def transform(self, X: np.ndarray) -> np.ndarray:
        return _lif_run(
            X, self._W_rec, self._W_in,
            self._v_th, self._v_reset,
            self._alpha_mem, self._alpha_syn,
            self._refractory_steps,
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
        self._step_refrac = np.zeros(units, dtype=np.int32)

    def step(self, x_t: np.ndarray) -> np.ndarray:
        u = float(np.asarray(x_t).reshape(-1)[0])
        v, s, refrac = self._step_v, self._step_s, self._step_refrac
        I = self._W_in[:, 0] * u + self._W_rec @ s
        not_refrac = refrac == 0
        v = np.where(not_refrac, self._alpha_mem * v + (1.0 - self._alpha_mem) * I, self._v_reset)
        spikes = np.logical_and(v >= self._v_th, not_refrac).astype(float)
        v = np.where(spikes > 0.0, self._v_reset, v)
        refrac = np.where(spikes > 0.0, self._refractory_steps, np.maximum(refrac - 1, 0))
        s = self._alpha_syn * s + spikes
        self._step_v, self._step_s, self._step_refrac = v, s, refrac
        return s
