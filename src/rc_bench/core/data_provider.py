import numpy as np
from typing import Tuple, Dict
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# Генераторы
# -----------------------------------------------------------------------------
def generate_narma10(T: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Генерация временного ряда NARMA10 (чистая математика)."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 0.5, size=T)
    y = np.zeros(T, dtype=float)

    for t in range(10, T):
        y_window = y[t - 10 : t]
        y[t] = (
            0.3 * y[t - 1]
            + 0.05 * y[t - 1] * np.sum(y_window)
            + 1.5 * u[t - 1] * u[t - 10]
            + 0.1
        )
    return u.reshape(-1, 1), y

# -----------------------------------------------------------------------------
# Сервис подготовки данных
# -----------------------------------------------------------------------------
def get_data_for_experiment(
    dataset_name: str, 
    length: int = 2000, # Дефолт для тестов, можно брать из конфига
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    seed: int = 42,
    scaler_name: str = "zscore"
) -> Dict[str, np.ndarray]:
    """
    Фабричный метод: получает имя датасета и возвращает готовые для обучения массивы.
    """
    
    # 1. Генерация (Router)
    if dataset_name.lower() == "narma10":
        X, y = generate_narma10(length, seed=seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # 2. Сплит (Train / Val / Test)
    n_train = int(length * train_frac)
    n_val = int(length * val_frac)
    # n_test остаток

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]

    # 3. Скейлинг
    if scaler_name.lower() == "zscore":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
    
    # Возвращаем словарь, чтобы не путаться в порядке аргументов
    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test
    }