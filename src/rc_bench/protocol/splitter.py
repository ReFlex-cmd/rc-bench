from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class Split:
    """One temporal segment of a time series dataset."""
    X: np.ndarray  # shape [T, input_dim]
    y: np.ndarray  # shape [T]


class TimeSeriesSplitter:
    """Split a univariate time series into train / val / test segments.

    Washout trimming is NOT applied here — that is the StateCollector's job.
    """

    def __init__(self, train_frac: float = 0.6, val_frac: float = 0.2) -> None:
        if train_frac + val_frac >= 1.0:
            raise ValueError("train_frac + val_frac must be < 1.0")
        self.train_frac = train_frac
        self.val_frac = val_frac

    def split(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Split]:
        """Return a dict with keys 'train', 'val', 'test'."""
        T = len(y)
        n_train = int(T * self.train_frac)
        n_val = int(T * self.val_frac)

        return {
            "train": Split(X=X[:n_train],              y=y[:n_train]),
            "val":   Split(X=X[n_train:n_train+n_val], y=y[n_train:n_train+n_val]),
            "test":  Split(X=X[n_train+n_val:],        y=y[n_train+n_val:]),
        }
