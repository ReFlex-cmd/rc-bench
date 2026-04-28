from typing import Dict, Any, Type

from .base import BaseReservoir
from .esn_service import ESNReservoir
from .lsm_service import LSMReservoir
from .fhn_service import FHNReservoir
from .logistic_service import LogisticReservoir
from .leaky_esn_service import LeakyESNReservoir
from .deep_esn_service import DeepESNReservoir
from .qrc_service import QRCReservoir

REGISTRY: Dict[str, Type[BaseReservoir]] = {
    "esn": ESNReservoir,
    "lsm": LSMReservoir,
    "fhn": FHNReservoir,
    "logistic": LogisticReservoir,
    "leaky_esn": LeakyESNReservoir,
    "deep_esn": DeepESNReservoir,
    "qrc": QRCReservoir,
}


def get_reservoir(reservoir_type: str, config: Dict[str, Any]) -> BaseReservoir:
    cls = REGISTRY.get(reservoir_type.lower())
    if cls is None:
        available = list(REGISTRY)
        raise ValueError(f"Unknown reservoir type {reservoir_type!r}. Available: {available}")
    return cls(config)
