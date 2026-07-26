from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np


class BaseReservoir(ABC):
    # Subclasses override to apply z-score scaling by default ("zscore" | "none")
    DEFAULT_SCALER: str = "none"

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self._build(config)

    @abstractmethod
    def _build(self, config: Dict[str, Any]) -> None:
        """Initialize reservoir internals from config dict."""
        ...

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Map input sequence X [T] or [T,1] to reservoir states H [T, units].

        Always starts from the reservoir's initial state (stateless between calls).
        """
        ...

    def sanity_check(self, H: np.ndarray) -> Dict[str, bool]:
        """Return dict of {check_name: passed}. Override in subclasses."""
        return {}

    # ------------------------------------------------------------------
    # Step-by-step API — required for closed-loop rollout (Stage 4)
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        """Reset the step-state to initial conditions.

        Must be called before the first ``step()`` call or ``warmup()`` call.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement reset_state()."
            " Required for closed-loop forecasting."
        )

    def step(self, x_t: np.ndarray) -> np.ndarray:
        """Advance the reservoir one step and return the state vector.

        ``x_t`` : shape [1, input_dim] or [input_dim]
        Returns  : shape (units,)
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement step()."
            " Required for closed-loop forecasting."
        )

    def warmup(self, X: np.ndarray) -> np.ndarray:
        """Warm up the reservoir step-state on ``X``; return H matrix.

        Calls ``reset_state()`` then ``step()`` for each row of X.
        After this method, ``step()`` continues from the final state.
        Returns H with shape [len(X), units].
        """
        self.reset_state()
        h_list = []
        for t in range(len(X)):
            h_list.append(self.step(X[t : t + 1]))
        return np.array(h_list)

    def step_operation_counts(self) -> Optional[Dict[str, int]]:
        """Аналитическая стоимость одного ``step()`` в MAC-операциях.

        Возвращает ``None``, если модель не описала свою арифметику: лучше
        честное «недоступно», чем правдоподобная выдумка. Ключи:
        ``reservoir_macs`` (умножения-сложения обновления состояния),
        ``reservoir_nonlinearities`` (число вызовов нелинейности),
        ``reservoir_nonzero_recurrent_weights`` (ненулевые элементы W_rec).

        Конвенция MAC-счёта (что именно входит в ``reservoir_macs``, а что
        нет) описана в docstring модуля ``rc_bench.profiling.activity``.
        """
        return None
