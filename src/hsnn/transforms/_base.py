from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Type

import numpy as np

__all__ = ["BaseTransform", "transform_registry"]

transform_registry: Dict[str, Type[BaseTransform]] = {}


class BaseTransform(ABC):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        transform_registry[cls.__name__.lower()] = cls

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.transform(image)

    @abstractmethod
    def transform(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
