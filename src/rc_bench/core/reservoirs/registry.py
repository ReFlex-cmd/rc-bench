from typing import Dict, Any, Type

from .base import BaseReservoir
from .esn_service import ESNReservoir
from .lsm_service import LSMReservoir
from .fhn_service import FHNReservoir
from .logistic_service import LogisticReservoir

REGISTRY: Dict[str, Type[BaseReservoir]] = {
    "esn": ESNReservoir,
    "lsm": LSMReservoir,
    "fhn": FHNReservoir,
    "logistic": LogisticReservoir,
}


def get_reservoir(reservoir_type: str, config: Dict[str, Any]) -> BaseReservoir:
    cls = REGISTRY.get(reservoir_type.lower())
    if cls is None:
        available = list(REGISTRY)
        raise ValueError(f"Unknown reservoir type {reservoir_type!r}. Available: {available}")
    return cls(config)
