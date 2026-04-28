import numpy as np
from typing import Any, Dict, Tuple
from sklearn.preprocessing import StandardScaler

DATASET_CATALOG: Dict[str, Dict[str, Any]] = {
    "narma10": {
        "description": "Non-linear Auto-Regressive Moving Average, order 10",
        "default_length": 2000,
        "input_dim": 1,
        "output_dim": 1,
    },
    "narma30": {
        "description": "Non-linear Auto-Regressive Moving Average, order 30 (longer memory)",
        "default_length": 3000,
        "input_dim": 1,
        "output_dim": 1,
    },
    "mackey_glass": {
        "description": "Mackey-Glass delay differential equation (tau=17, chaotic regime)",
        "default_length": 5000,
        "input_dim": 1,
        "output_dim": 1,
    },
    "lorenz63": {
        "description": "Lorenz-63 attractor, x-component one-step prediction (RK4, dt=0.01, sub=10)",
        "default_length": 5000,
        "input_dim": 1,
        "output_dim": 1,
    },
}

# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def generate_narma10(T: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 0.5, size=T)
    y = np.zeros(T, dtype=float)
    for t in range(10, T):
        y[t] = (0.3 * y[t - 1]
                + 0.05 * y[t - 1] * np.sum(y[t - 10:t])
                + 1.5 * u[t - 1] * u[t - 10]
                + 0.1)
    return u.reshape(-1, 1), y


def generate_narma30(T: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    # beta scaled by 10/30 so per-element contribution matches NARMA10 and prevents divergence
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 0.5, size=T)
    y = np.zeros(T, dtype=float)
    for t in range(30, T):
        y[t] = (0.3 * y[t - 1]
                + (0.05 / 3.0) * y[t - 1] * np.sum(y[t - 30:t])
                + 1.5 * u[t - 1] * u[t - 30]
                + 0.1)
    return u.reshape(-1, 1), y


def generate_mackey_glass(
    T: int,
    tau: int = 17,
    beta: float = 0.2,
    gamma: float = 0.1,
    n: int = 10,
    dt: float = 1.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mackey-Glass DDE via Euler integration (standard in RC literature).

    Generates T+1 post-warmup points; returns:
        X : x(t),   shape [T, 1]
        y : x(t+1), shape [T]
    """
    rng = np.random.default_rng(seed)
    n_warmup = 500          # discard initial transient
    total = n_warmup + T + 1

    x = np.zeros(total)
    # Perturbed constant initial history
    x[:tau] = 0.9 + rng.uniform(-0.01, 0.01, tau)

    for t in range(tau, total - 1):
        x_tau = x[t - tau]
        x[t + 1] = x[t] + dt * (beta * x_tau / (1.0 + x_tau ** n) - gamma * x[t])

    series = x[n_warmup:]       # length T+1
    return series[:-1].reshape(-1, 1), series[1:]


def generate_lorenz63(
    T: int,
    dt: float = 0.01,
    subsample: int = 10,
    seed: int = 42,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Lorenz-63 attractor via RK4.

    Integration with fine ``dt``, subsampled by ``subsample`` factor.
    Returns x-component one-step-ahead prediction task:
        X : x(t),   shape [T, 1]
        y : x(t+1), shape [T]
    """
    rng = np.random.default_rng(seed)
    n_warmup_sub = 500          # subsampled warmup steps
    total_sub = n_warmup_sub + T + 1
    total_fine = total_sub * subsample

    state = np.array([
        rng.uniform(-15.0, 15.0),
        rng.uniform(-15.0, 15.0),
        rng.uniform(20.0, 40.0),
    ])

    def _rhs(s: np.ndarray) -> np.ndarray:
        sx, sy, sz = s
        return np.array([
            sigma * (sy - sx),
            sx * (rho - sz) - sy,
            sx * sy - beta * sz,
        ])

    x_fine = np.empty(total_fine)
    for i in range(total_fine):
        x_fine[i] = state[0]
        k1 = dt * _rhs(state)
        k2 = dt * _rhs(state + k1 / 2)
        k3 = dt * _rhs(state + k2 / 2)
        k4 = dt * _rhs(state + k3)
        state = state + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    x_sub = x_fine[::subsample]             # length total_sub
    series = x_sub[n_warmup_sub:]           # length T+1
    return series[:-1].reshape(-1, 1), series[1:]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_GENERATORS = {
    "narma10":     generate_narma10,
    "narma30":     generate_narma30,
    "mackey_glass": generate_mackey_glass,
    "lorenz63":    generate_lorenz63,
}


def get_data_for_experiment(
    dataset_name: str,
    length: int = 2000,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    seed: int = 42,
    scaler_name: str = "zscore",
) -> Dict[str, np.ndarray]:
    key = dataset_name.lower()
    gen = _GENERATORS.get(key)
    if gen is None:
        raise ValueError(f"Unknown dataset: {dataset_name!r}. Available: {list(_GENERATORS)}")

    X, y = gen(length, seed=seed)

    n_train = int(length * train_frac)
    n_val = int(length * val_frac)

    X_train, y_train = X[:n_train],              y[:n_train]
    X_val,   y_val   = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test,  y_test  = X[n_train+n_val:],        y[n_train+n_val:]

    if scaler_name.lower() == "zscore":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)
        X_test  = scaler.transform(X_test)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
    }
