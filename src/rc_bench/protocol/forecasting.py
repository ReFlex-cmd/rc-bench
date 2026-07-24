"""
Three forecasting protocols for reservoir computing benchmarks.

one_step
    Teacher-forced, H[t] → y[t].  The reservoir is always driven by
    ground-truth inputs.  Targets are returned unchanged.

fixed_horizon
    Teacher-forced, H[t] → y[t + h].  States and targets are trimmed
    so that H[:-h] aligns with y[h:].

closed_loop
    Train identically to one_step (teacher forcing).  At inference, the
    reservoir is warmed up on X_seed (the test-set washout window), then
    the readout's predictions are fed back as reservoir inputs for the
    remaining steps — no ground-truth input is used during rollout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np

if TYPE_CHECKING:
    from rc_bench.core.reservoirs.base import BaseReservoir
    from rc_bench.readout.ridge import RidgeReadout


# ---------------------------------------------------------------------------
# Target builders  (post-washout H and y)
# ---------------------------------------------------------------------------

def build_one_step_targets(
    H: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Identity alignment: H[t] predicts y[t]."""
    return H, y


def build_fixed_horizon_targets(
    H: np.ndarray,
    y: np.ndarray,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Shift targets forward by `horizon` steps: H[t] predicts y[t + horizon].

    Returns (H[:-horizon], y[horizon:]).  Both arrays have length T - horizon.
    """
    if horizon <= 0:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if horizon >= len(y):
        raise ValueError(
            f"horizon={horizon} >= sequence length={len(y)}; no samples remain."
        )
    return H[:-horizon], y[horizon:]


# ---------------------------------------------------------------------------
# Shared per-split target alignment (DEC-012/DEC-013)
# ---------------------------------------------------------------------------

def target_offset(washout: int, horizon: int, mode: str) -> int:
    """Split-local index of the first evaluated target after alignment.

    ``fixed_horizon`` skips ``washout + horizon`` leading positions; ``one_step``
    and ``closed_loop`` skip ``washout``. Reservoir and baseline runners share
    this so every model is evaluated on the same target timestamps.
    """
    if mode == "fixed_horizon":
        return washout + horizon
    if mode in ("one_step", "closed_loop"):
        return washout
    raise ValueError(f"Unknown forecasting_mode: {mode!r}")


def aligned_target_positions(
    observed_mask: np.ndarray,
    washout: int,
    horizon: int,
    mode: str = "fixed_horizon",
) -> np.ndarray:
    """Split-local indices of observed targets after washout/horizon alignment."""
    offset = target_offset(washout, horizon, mode)
    mask = np.asarray(observed_mask, dtype=bool)
    positions = np.arange(offset, mask.size)
    return positions[mask[offset:]]


# ---------------------------------------------------------------------------
# Closed-loop rollout
# ---------------------------------------------------------------------------

def closed_loop_predict(
    readout: "RidgeReadout",
    reservoir: "BaseReservoir",
    X_seed: np.ndarray,
    n_steps: int,
) -> np.ndarray:
    """Autonomous (closed-loop) multi-step rollout.

    1. Warm up the reservoir state on ``X_seed`` using the reservoir's
       ``warmup()`` method (reset_state + sequential step calls).
    2. Starting from the last seed input, feed the readout's prediction
       back as the next reservoir input for ``n_steps`` steps.

    Parameters
    ----------
    readout : fitted RidgeReadout
    reservoir : BaseReservoir — must implement reset_state() and step()
    X_seed : shape [seed_len, input_dim]  — warm-up sequence
    n_steps : number of autonomous steps to generate

    Returns
    -------
    y_pred : np.ndarray, shape [n_steps]
    """
    if n_steps <= 0:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")

    # Warm up: drives reservoir through the seed window. After this call
    # H_warmup[-1] is the reservoir state immediately after seeing X_seed[-1].
    H_warmup = reservoir.warmup(X_seed)

    y_pred = np.empty(n_steps)
    # First prediction comes directly from the last warmup state — no extra
    # step would only re-process X_seed[-1] (off-by-one bug fixed 2026-05-13).
    y_t = float(readout.predict(H_warmup[-1].reshape(1, -1))[0])
    y_pred[0] = y_t

    for t in range(1, n_steps):
        h_t = reservoir.step(np.array([[y_t]]))       # autonomous: feed prediction
        y_t = float(readout.predict(h_t.reshape(1, -1))[0])
        y_pred[t] = y_t

    return y_pred
