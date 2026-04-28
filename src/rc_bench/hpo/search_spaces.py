"""
Per-reservoir Optuna search spaces.

Each entry defines the hyperparameters to tune for one reservoir type.
The "readout_alpha" key is always included and maps to the Ridge penalty.

Format per parameter:
    ("float",     low, high)           – uniform float
    ("float_log", low, high)           – log-uniform float  (low > 0)
    ("int",       low, high)           – uniform int
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import optuna

# ---------------------------------------------------------------------------
# Space definitions
# ---------------------------------------------------------------------------

_SpaceEntry = Tuple[str, float, float]

SEARCH_SPACES: Dict[str, Dict[str, _SpaceEntry]] = {
    "esn": {
        "spectral_radius":   ("float",     0.1,  1.5),
        "input_scaling":     ("float_log", 0.01, 2.0),
        "lr":                ("float",     0.1,  1.0),
        "rc_connectivity":   ("float_log", 0.01, 0.5),
        "readout_alpha":     ("float_log", 1e-4, 10.0),
    },
    "fhn": {
        "sr":                ("float",     0.1,  1.5),
        "input_scale":       ("float_log", 0.01, 2.0),
        "density":           ("float_log", 0.01, 0.5),
        "dt":                ("float_log", 0.01, 0.5),
        "readout_alpha":     ("float_log", 1e-4, 10.0),
    },
    "lsm": {
        "tau_mem":           ("float_log", 5.0,  80.0),
        "tau_syn":           ("float_log", 2.0,  40.0),
        "input_scale":       ("float_log", 0.1,  5.0),
        "density":           ("float_log", 0.01, 0.3),
        "readout_alpha":     ("float_log", 1e-4, 10.0),
    },
    "logistic": {
        "r_min":             ("float",     3.5,  3.95),
        "r_max":             ("float",     3.8,  4.0),
        "input_scale":       ("float_log", 0.005, 0.5),
        "coupling":          ("float_log", 0.01, 0.5),
        "readout_alpha":     ("float_log", 1e-4, 10.0),
    },
}


# ---------------------------------------------------------------------------
# Suggest helpers
# ---------------------------------------------------------------------------

def suggest_params(trial: optuna.Trial, reservoir_type: str) -> Dict[str, Any]:
    """Ask ``trial`` to suggest all hyperparameters for ``reservoir_type``.

    Returns a dict with two keys:
        "reservoir_params" : dict of reservoir config overrides
        "readout_alpha"    : float – Ridge penalty suggested by Optuna
    """
    space = SEARCH_SPACES[reservoir_type.lower()]
    reservoir_params: Dict[str, Any] = {}
    readout_alpha: float = 1.0

    for name, spec in space.items():
        kind, low, high = spec
        if kind == "float":
            value = trial.suggest_float(name, low, high)
        elif kind == "float_log":
            value = trial.suggest_float(name, low, high, log=True)
        elif kind == "int":
            value = trial.suggest_int(name, int(low), int(high))
        else:
            raise ValueError(f"Unknown parameter kind: {kind!r}")

        if name == "readout_alpha":
            readout_alpha = float(value)
        else:
            reservoir_params[name] = value

    # Logistic: r_max must be >= r_min
    if reservoir_type.lower() == "logistic":
        r_min = reservoir_params.get("r_min", 3.5)
        r_max = reservoir_params.get("r_max", 3.8)
        if r_max < r_min:
            reservoir_params["r_max"] = r_min

    return {"reservoir_params": reservoir_params, "readout_alpha": readout_alpha}


def params_from_trial(
    trial_params: Dict[str, Any],
    reservoir_type: str,
) -> Dict[str, Any]:
    """Reconstruct the structured params dict from a completed trial's flat param dict."""
    readout_alpha = trial_params.get("readout_alpha", 1.0)
    reservoir_params = {k: v for k, v in trial_params.items() if k != "readout_alpha"}
    return {"reservoir_params": reservoir_params, "readout_alpha": readout_alpha}
