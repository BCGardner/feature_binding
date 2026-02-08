from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Type

import numpy as np
import numpy.typing as npt

from ..interfaces import IEncoder

__all__ = ["BaseEncoder", "encoder_registry"]

encoder_registry: Dict[str, Type[BaseEncoder]] = {}


class BaseEncoder(ABC, IEncoder):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = cls.__name__.lower()
        encoder_registry[name.removesuffix('encoder')] = cls

    @abstractmethod
    def transform(self, data: np.ndarray) -> npt.NDArray[np.float_]:
        ...
