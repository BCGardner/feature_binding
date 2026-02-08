from pathlib import Path
from typing import Any, Dict, Mapping

from .interfaces import INetwork
from ._brian2.network import SpatialNet

__all__ = ["create_network"]


_BACKENDS: Dict[str, Any] = {
    'brian2': SpatialNet,
}


def create_network(model_cfg: str | Path | Mapping, backend: str = 'brian2') -> INetwork:
    if backend not in _BACKENDS:
        raise NotImplementedError()
    model = _BACKENDS[backend]
    return model(model_cfg)
