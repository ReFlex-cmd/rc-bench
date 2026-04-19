from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rc_bench.core.data_provider import generate_narma10


def prepare_narma10(
    out_dir: Path,
    length: int = 20000,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    seed: int = 42,
) -> None:
    """
    Генерация и сохранение данных для задачи NARMA10 в общем формате:
      X_train.npy, y_train.npy
      X_val.npy,   y_val.npy
      X_test.npy,  y_test.npy
      spec.json
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y = generate_narma10(length, seed=seed)

    n_train = int(length * train_frac)
    n_val = int(length * val_frac)
    n_test = length - n_train - n_val

    # индексы
    train_slice = slice(0, n_train)
    val_slice = slice(n_train, n_train + n_val)
    test_slice = slice(n_train + n_val, length)

    X_train, y_train = X[train_slice], y[train_slice]
    X_val, y_val = X[val_slice], y[val_slice]
    X_test, y_test = X[test_slice], y[test_slice]

    # сохраняем массивы
    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "X_val.npy", X_val)
    np.save(out_dir / "y_val.npy", y_val)
    np.save(out_dir / "X_test.npy", X_test)
    np.save(out_dir / "y_test.npy", y_test)

    # базовый spec.json — общие настройки для всех моделей
    spec = {
        "task": "narma10",
        "length": length,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "test_frac": n_test / length,
        "washout": 200,
        "scaler": "zscore",
        "readout": "ridge",
        "ridge_alpha_grid": [0.001, 0.01, 0.1, 1.0, 10.0],
        "seed": seed,
    }

    with (out_dir / "spec.json").open("w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=2)
