import numpy as np
from typing import Dict, Any

from .base import BaseReservoir


class QRCReservoir(BaseReservoir):
    """Quantum Reservoir Computing stub — Ising mean-field simulation.

    Models each virtual qubit as a spin with mean-field magnetization
    x_i ≈ <sigma_z_i> ∈ [-1, 1].

    Dynamics (transverse-field Ising update):
        x(t) = tanh(J @ x(t-1) + h_in * u(t))

    Output uses the "virtual nodes" technique (Fujii & Nakajima 2017):
    after the input-driven step, ``depth - 1`` free-evolution steps are
    applied and all intermediate states are concatenated, yielding a
    feature vector of size n_qubits * depth.

    Parameters
    ----------
    n_qubits : int
        Number of virtual qubits (hidden state size).  Default 50.
    depth    : int
        Number of virtual-node measurement steps per time step.  Default 3.
    coupling : float
        Scale of the random Ising coupling matrix J.  Default 0.5.
    sr       : float
        Target spectral radius of J.  Default 0.9.
    seed     : int
    """

    DEFAULT_SCALER = "none"  # outputs lie in (-1, 1)^(n_qubits * depth)

    def _build(self, config: Dict[str, Any]) -> None:
        n_qubits = int(config.get("n_qubits", 50))
        self._depth = int(config.get("depth", 3))
        coupling = float(config.get("coupling", 0.5))
        sr = float(config.get("sr", 0.9))
        seed = config.get("seed", 42)

        self._n_qubits = n_qubits
        rng = np.random.default_rng(seed)

        # Symmetric Ising coupling matrix (ZZ interactions)
        J = rng.normal(0.0, coupling / np.sqrt(n_qubits), (n_qubits, n_qubits))
        J = (J + J.T) / 2.0
        np.fill_diagonal(J, 0.0)
        max_ev = np.max(np.abs(np.linalg.eigvals(J)))
        if max_ev > 1e-8:
            J = J / max_ev * sr
        self._J = J

        # Transverse input field h_i (one weight per qubit)
        self._h_in = rng.uniform(-1.0, 1.0, n_qubits)

        self._out_dim = n_qubits * self._depth

    def _virtual_nodes(self, x: np.ndarray, u_t: float) -> np.ndarray:
        """One input step + depth-1 free-evolution steps → feature vector."""
        # Input-driven update
        x = np.tanh(self._J @ x + self._h_in * u_t)
        parts = [x]
        # Free-evolution virtual nodes
        for _ in range(self._depth - 1):
            x = np.tanh(self._J @ x)
            parts.append(x)
        return np.concatenate(parts)

    def transform(self, X: np.ndarray) -> np.ndarray:
        u = X.reshape(-1)
        T = len(u)
        x = np.zeros(self._n_qubits)
        H = np.zeros((T, self._out_dim))
        for t in range(T):
            features = self._virtual_nodes(x, float(u[t]))
            x = features[: self._n_qubits]  # carry forward driven state
            H[t] = features
        return H

    def sanity_check(self, H: np.ndarray) -> Dict[str, bool]:
        return {
            "bounded": bool(np.max(np.abs(H)) < 1.0 + 1e-9),
            "non_trivial": bool(H.std() > 0.01),
        }

    # ------------------------------------------------------------------
    # Step API
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        self._step_x = np.zeros(self._n_qubits)

    def step(self, x_t: np.ndarray) -> np.ndarray:
        u = float(np.asarray(x_t).reshape(-1)[0])
        features = self._virtual_nodes(self._step_x, u)
        self._step_x = features[: self._n_qubits]
        return features
