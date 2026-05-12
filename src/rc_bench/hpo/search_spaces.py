"""
Per-reservoir Optuna search spaces.

Each entry defines the hyperparameters to tune for one reservoir type.
The "readout_alpha" key is always included and maps to the Ridge penalty.

Format per parameter:
    ("float",     low, high)           – uniform float
    ("float_log", low, high)           – log-uniform float  (low > 0)
    ("int",       low, high)           – uniform int

Special case: ``deep_esn`` uses dynamic per-layer parameters (sr_l, leak_rate_l)
and is handled in ``suggest_params`` directly, not by iterating SEARCH_SPACES.
See ТЗ §1.3 / Gallicchio & Micheli 2017.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import optuna

# ---------------------------------------------------------------------------
# Space definitions
# ---------------------------------------------------------------------------

_SpaceEntry = Tuple[str, float, float]

SEARCH_SPACES: Dict[str, Dict[str, _SpaceEntry]] = {
    # Classical Jaeger-style ESN: leak rate fixed to 1.0 in the model (not tuned).
    "esn": {
        "spectral_radius":   ("float",     0.1,  1.5),
        "input_scaling":     ("float_log", 0.01, 2.0),
        "rc_connectivity":   ("float_log", 0.01, 0.20),
        "readout_alpha":     ("float_log", 1e-6, 1e+2),
    },
    # FHN: dt is fixed at 0.01 in fhn_service._build (NOT tuned, see audit/01).
    "fhn": {
        "a":                 ("float",     0.5,  1.0),
        "b":                 ("float",     0.5,  1.0),
        "epsilon":           ("float_log", 0.01, 0.5),
        "coupling_strength": ("float_log", 0.001, 1.0),
        "input_scale":       ("float_log", 0.01, 5.0),
        "density":           ("float_log", 0.01, 0.20),
        "readout_alpha":     ("float_log", 1e-6, 1e+2),
    },
    "lsm": {
        "tau_mem":           ("float_log", 10.0, 50.0),
        "tau_syn":           ("float_log", 2.0,  10.0),
        "v_th":              ("float",     0.5,  1.5),
        "t_refractory":      ("float",     2.0,  5.0),
        "w_rec_scale":       ("float_log", 0.1,  10.0),
        "input_scale":       ("float_log", 0.1,  10.0),
        "density":           ("float_log", 0.05, 0.20),
        "readout_alpha":     ("float_log", 1e-6, 1e+2),
    },
    "logistic": {
        "r_min":             ("float",     3.7,  3.85),
        "r_max":             ("float",     3.9,  3.99),
        "input_scale":       ("float_log", 0.01, 5.0),
        "coupling":          ("float_log", 0.001, 1.0),
        "readout_alpha":     ("float_log", 1e-6, 1e+2),
    },
    "leaky_esn": {
        "sr":                ("float",     0.1,  1.5),
        "leak_rate":         ("float",     0.05, 1.0),
        "input_scaling":     ("float_log", 0.01, 2.0),
        "density":           ("float_log", 0.01, 0.20),
        "readout_alpha":     ("float_log", 1e-6, 1e+2),
    },
    # NOTE: deep_esn entry is a placeholder — actual sampling is done dynamically
    # in suggest_params (per-layer sr and leak_rate). Listed here so consumers
    # like _DEEP_ESN_SHARED below can read shared ranges.
    "deep_esn": {
        "n_layers":          ("int",       2,    5),
        "input_scaling":     ("float_log", 0.01, 2.0),
        "density":           ("float_log", 0.01, 0.20),
        "readout_alpha":     ("float_log", 1e-6, 1e+2),
    },
    # QRC here is a classical mean-field Ising surrogate (see audit/notes_qrc.md),
    # not a true unitary quantum reservoir.
    "qrc": {
        "sr":                ("float",     0.1,  1.5),
        "coupling":          ("float_log", 0.1,  2.0),
        "depth":             ("int",       2,    6),
        "readout_alpha":     ("float_log", 1e-6, 1e+2),
    },
}

# Per-layer ranges for Deep ESN (used by _suggest_deep_esn). Kept separate from
# SEARCH_SPACES because they're sampled n_layers times each.
_DEEP_ESN_PER_LAYER_RANGES: Dict[str, _SpaceEntry] = {
    "sr":         ("float", 0.1, 1.5),
    "leak_rate":  ("float", 0.05, 1.0),
}


# ---------------------------------------------------------------------------
# Suggest helpers
# ---------------------------------------------------------------------------

def _suggest_one(trial: optuna.Trial, name: str, spec: _SpaceEntry) -> float:
    kind, low, high = spec
    if kind == "float":
        return trial.suggest_float(name, low, high)
    if kind == "float_log":
        return trial.suggest_float(name, low, high, log=True)
    if kind == "int":
        return trial.suggest_int(name, int(low), int(high))
    raise ValueError(f"Unknown parameter kind: {kind!r}")


def _suggest_deep_esn(trial: optuna.Trial) -> Dict[str, Any]:
    """Per-layer HPO for Deep ESN (see ТЗ §1.3, Gallicchio & Micheli 2017).

    Samples ``n_layers`` first, then sr_0..sr_{n-1} and leak_rate_0..leak_rate_{n-1}
    independently. Shared params (input_scaling, density) sampled once.
    """
    space = SEARCH_SPACES["deep_esn"]
    n_layers = int(_suggest_one(trial, "n_layers", space["n_layers"]))

    sr_list = [
        _suggest_one(trial, f"sr_{i}", _DEEP_ESN_PER_LAYER_RANGES["sr"])
        for i in range(n_layers)
    ]
    leak_list = [
        _suggest_one(trial, f"leak_rate_{i}", _DEEP_ESN_PER_LAYER_RANGES["leak_rate"])
        for i in range(n_layers)
    ]
    input_scaling = _suggest_one(trial, "input_scaling", space["input_scaling"])
    density = _suggest_one(trial, "density", space["density"])
    readout_alpha = _suggest_one(trial, "readout_alpha", space["readout_alpha"])

    reservoir_params = {
        "n_layers":      n_layers,
        "sr":            sr_list,
        "leak_rate":     leak_list,
        "input_scaling": input_scaling,
        "density":       density,
    }
    return {"reservoir_params": reservoir_params, "readout_alpha": float(readout_alpha)}


def _params_from_trial_deep_esn(trial_params: Dict[str, Any]) -> Dict[str, Any]:
    """Reconstruct Deep ESN structured params from a flat trial.params dict."""
    n_layers = int(trial_params["n_layers"])
    sr_list = [float(trial_params[f"sr_{i}"]) for i in range(n_layers)]
    leak_list = [float(trial_params[f"leak_rate_{i}"]) for i in range(n_layers)]
    return {
        "reservoir_params": {
            "n_layers":      n_layers,
            "sr":            sr_list,
            "leak_rate":     leak_list,
            "input_scaling": float(trial_params["input_scaling"]),
            "density":       float(trial_params["density"]),
        },
        "readout_alpha": float(trial_params["readout_alpha"]),
    }


def suggest_params(trial: optuna.Trial, reservoir_type: str) -> Dict[str, Any]:
    """Ask ``trial`` to suggest all hyperparameters for ``reservoir_type``.

    Returns a dict with two keys:
        "reservoir_params" : dict of reservoir config overrides
        "readout_alpha"    : float – Ridge penalty suggested by Optuna
    """
    rtype = reservoir_type.lower()
    if rtype == "deep_esn":
        return _suggest_deep_esn(trial)

    space = SEARCH_SPACES[rtype]
    reservoir_params: Dict[str, Any] = {}
    readout_alpha: float = 1.0

    for name, spec in space.items():
        value = _suggest_one(trial, name, spec)
        if name == "readout_alpha":
            readout_alpha = float(value)
        else:
            reservoir_params[name] = value

    # Logistic: r_max must be >= r_min
    if rtype == "logistic":
        r_min = reservoir_params.get("r_min", 3.7)
        r_max = reservoir_params.get("r_max", 3.9)
        if r_max < r_min:
            reservoir_params["r_max"] = r_min

    return {"reservoir_params": reservoir_params, "readout_alpha": readout_alpha}


def params_from_trial(
    trial_params: Dict[str, Any],
    reservoir_type: str,
) -> Dict[str, Any]:
    """Reconstruct the structured params dict from a completed trial's flat param dict."""
    if reservoir_type.lower() == "deep_esn":
        return _params_from_trial_deep_esn(trial_params)

    readout_alpha = trial_params.get("readout_alpha", 1.0)
    reservoir_params = {k: v for k, v in trial_params.items() if k != "readout_alpha"}
    return {"reservoir_params": reservoir_params, "readout_alpha": readout_alpha}
