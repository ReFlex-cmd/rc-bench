import numpy as np
from typing import Dict, Any

from .base import BaseReservoir


class ESNReservoir(BaseReservoir):
    DEFAULT_SCALER = "none"

    def _build(self, config: Dict[str, Any]) -> None:
        from reservoirpy.nodes import Reservoir
        self._res = Reservoir(
            units=config.get("n_units", 300),
            lr=config.get("lr", 1.0),
            sr=config.get("spectral_radius", 0.9),
            input_scaling=config.get("input_scaling", 0.5),
            rc_connectivity=config.get("rc_connectivity", 0.1),
            input_connectivity=config.get("input_connectivity", 0.1),
            seed=config.get("seed", 42),
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._res.run(X)

    def sanity_check(self, H: np.ndarray) -> Dict[str, bool]:
        return {"bounded_states": bool(np.max(np.abs(H)) < 1e6)}

    # ------------------------------------------------------------------
    # Step API — uses reservoirpy's stateful run() for continuity
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        """Reset reservoirpy node to its initial (zero) state.

        reservoirpy initializes lazily on the first run() call.
        We ensure initialization has happened before calling reset().
        """
        if not self._res.initialized:
            # One silent forward pass to trigger weight initialization
            self._res.run(np.zeros((1, 1)))
        self._res.reset()

    def step(self, x_t: np.ndarray) -> np.ndarray:
        """Run one step; reservoirpy step() expects a 1-D array."""
        return self._res.step(np.asarray(x_t).reshape(-1))

    def warmup(self, X: np.ndarray) -> np.ndarray:
        """Use reservoirpy's native batch run for efficient warmup."""
        self.reset_state()
        return self._res.run(X)
