from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Type

from brian2 import Equations

from ..codeblock import CodeBlock

__all__ = ["PlasticityRule", "plasticity_registry"]

plasticity_registry: Dict[str, Type[PlasticityRule]] = {}


class PlasticityRule(ABC):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = cls.__name__.lower()
        plasticity_registry[name.removesuffix('rule')] = cls

    @property
    @abstractmethod
    def model(self) -> Equations:
        ...

    @property
    @abstractmethod
    def on_pre(self) -> CodeBlock:
        ...

    @property
    @abstractmethod
    def on_post(self) -> CodeBlock:
        ...
