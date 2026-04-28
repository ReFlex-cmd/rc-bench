from __future__ import annotations

import numpy as np

from rc_bench.core.reservoirs.base import BaseReservoir


class StateCollector:
    """Run a reservoir on raw inputs and apply washout trimming.

    ``collect(X)``  → H[washout:]   (reservoir states, post-washout)
    ``trim(y)``     → y[washout:]   (matching target slice)
    """

    def __init__(self, reservoir: BaseReservoir, washout: int) -> None:
        if washout < 0:
            raise ValueError(f"washout must be >= 0, got {washout}")
        self.reservoir = reservoir
        self.washout = washout

    def collect(self, X: np.ndarray) -> np.ndarray:
        """Transform X through the reservoir and discard the first `washout` states."""
        H = self.reservoir.transform(X)
        return H[self.washout:] if self.washout > 0 else H

    def trim(self, y: np.ndarray) -> np.ndarray:
        """Trim the first `washout` samples from a target array."""
        return y[self.washout:] if self.washout > 0 else y
