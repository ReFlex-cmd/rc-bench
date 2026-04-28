import numpy as np
from typing import Dict, Any, List, Tuple

from .base import BaseReservoir


class DeepESNReservoir(BaseReservoir):
    """Deep Echo State Network: stacked leaky ESN layers (pure numpy).

    Layer 0 receives the scalar input u(t).
    Layer l receives the state vector of layer l-1 as input.
    Output: concatenation of all layer states  [T, n_layers * units].
    """

    DEFAULT_SCALER = "none"

    def _build(self, config: Dict[str, Any]) -> None:
        self._n_layers = int(config.get("n_layers", 3))
        units = int(config.get("units", 100))
        sr = float(config.get("sr", 0.9))
        self._leak_rate = float(config.get("leak_rate", 0.3))
        input_scaling = float(config.get("input_scaling", 0.5))
        density = float(config.get("density", 0.1))
        seed = config.get("seed", 42)

        self._units = units
        rng = np.random.default_rng(seed)

        self._W_ins: List[np.ndarray] = []
        self._W_recs: List[np.ndarray] = []

        for layer in range(self._n_layers):
            in_dim = 1 if layer == 0 else units
            W_in = rng.uniform(-input_scaling, input_scaling, (units, in_dim))
            W_rec = rng.normal(0.0, 1.0, (units, units))
            W_rec *= rng.random((units, units)) < density
            max_ev = np.max(np.abs(np.linalg.eigvals(W_rec)))
            if max_ev > 1e-8:
                W_rec = W_rec / max_ev * sr
            self._W_ins.append(W_in)
            self._W_recs.append(W_rec)

        self._out_dim = self._n_layers * units

    def _advance(
        self, u_scalar: float, states: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        alpha = self._leak_rate
        new_states: List[np.ndarray] = []
        inp = np.array([u_scalar])  # shape (1,) for layer 0

        for layer in range(self._n_layers):
            h = states[layer]
            pre = np.tanh(self._W_ins[layer] @ inp + self._W_recs[layer] @ h)
            new_h = (1.0 - alpha) * h + alpha * pre
            new_states.append(new_h)
            inp = new_h  # next layer reads current layer's output

        obs = np.concatenate(new_states)
        return new_states, obs

    def transform(self, X: np.ndarray) -> np.ndarray:
        u = X.reshape(-1)
        T = len(u)
        states = [np.zeros(self._units) for _ in range(self._n_layers)]
        H = np.zeros((T, self._out_dim))
        for t in range(T):
            states, H[t] = self._advance(float(u[t]), states)
        return H

    def sanity_check(self, H: np.ndarray) -> Dict[str, bool]:
        checks: Dict[str, bool] = {}
        for layer in range(self._n_layers):
            cols = H[:, layer * self._units:(layer + 1) * self._units]
            checks[f"layer_{layer}_bounded"] = bool(np.max(np.abs(cols)) < 1e6)
            checks[f"layer_{layer}_active"] = bool(cols.std() > 1e-6)
        return checks

    # ------------------------------------------------------------------
    # Step API
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        self._step_states = [np.zeros(self._units) for _ in range(self._n_layers)]

    def step(self, x_t: np.ndarray) -> np.ndarray:
        u = float(np.asarray(x_t).reshape(-1)[0])
        self._step_states, obs = self._advance(u, self._step_states)
        return obs
