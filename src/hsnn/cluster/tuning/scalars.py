from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Optional, Sequence, Type

import numpy as np
import pandas as pd

from hsnn.core.definitions import Projection
from hsnn.simulation import Simulator
from ._utils import get_rates_db

scalar_registry: Dict[str, Type[BaseScalar]] = {}

__all__ = ['scalar_registry']


class BaseScalar(ABC):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = cls.__name__.lower()
        scalar_registry[name.removesuffix('scalar')] = cls

    def __init__(self, **kwargs) -> None:
        return None

    @abstractmethod
    def __call__(self, simulator: Simulator, data: Sequence[np.ndarray],
                 labels: Optional[np.ndarray] = None) -> float:
        ...


class StateMixin(ABC):
    @property
    @abstractmethod
    def is_built(self) -> bool:
        ...

    @property
    @abstractmethod
    def iteration(self) -> int:
        ...

    @abstractmethod
    def build(self, simulator: Simulator):
        ...


class DeltaWeightScalar(BaseScalar, StateMixin):
    _PROJECTIONS_PLASTIC = (Projection.FF, Projection.E2E, Projection.FB)

    def __init__(self, normalise: bool = True) -> None:
        self.normalise = normalise
        self._ws_prev: pd.Series
        self._is_built = False

    def __call__(self, simulator: Simulator, data: Sequence[np.ndarray],
                 labels: Optional[np.ndarray] = None) -> float:
        deltas = self.get_deltas(simulator, update_state=True)
        return np.mean(list(deltas.values())).item()

    @property
    def is_built(self) -> bool:
        return self._is_built

    @property
    def iteration(self) -> int:
        return self._iteration

    def build(self, simulator: Simulator):
        ws = self._get_weights(simulator)
        self._indices: list[tuple[int, str]] = []
        for index in zip(ws.index.get_level_values('layer'), ws.index.get_level_values('proj')):
            if index not in self._indices:
                self._indices.append(index)
        self._ws_prev = ws
        self._is_built = True
        self._iteration = 0

    def get_deltas(self, simulator: Simulator, update_state: bool = True) -> dict:
        if not self.is_built:
            raise RuntimeError("not yet built on initial network state")
        ws = self._get_weights(simulator)
        deltas = {}
        for index in self._indices:
            deltas[index] = np.linalg.norm(ws.xs(index) - self._ws_prev.xs(index))
            if self.normalise:
                deltas[index] /= np.linalg.norm(self._ws_prev.xs(index))
        if update_state:
            self._ws_prev = ws
            self._iteration += 1
        return deltas

    def _get_weights(self, simulator: Simulator) -> pd.Series:
        ws = simulator.network.get_syn_params(return_delays=False,
                                              projections=self._PROJECTIONS_PLASTIC)
        if ws is None:
            raise ValueError()
        else:
            return ws['w']


class RatesPercentileScalar(BaseScalar):
    def __init__(self, percentile: float = 90.0, layer: Optional[Any] = None,
                 subset: Optional[Iterable] = None, duration: float = 150,
                 duration_relax: float = 0, offset: float = 50, reps: int = 1,
                 conditional_fired: bool = True, interpolation: str = 'higher') -> None:
        self.percentile = percentile
        self.layer = layer
        self.duration = duration
        self.duration_relax = duration_relax
        self.offset = offset
        self.subset = subset
        self.reps = reps
        self.conditional_fired = conditional_fired
        self.interpolation = interpolation

    def __call__(self, simulator: Simulator, data: Sequence[np.ndarray],
                 labels: Optional[np.ndarray] = None) -> float:
        data = [data[i] for i in self.subset] if self.subset is not None else data
        rates_db = get_rates_db(simulator.network, data, self.duration,
                                self.duration_relax, self.offset, self.reps,
                                self.conditional_fired)
        if self.layer is not None:
            rs = rates_db['rate'].sel(layer=self.layer).values
        else:
            rs = rates_db['rate'].values
        rs = rs[rs > 0]
        if len(rs):
            return float(np.percentile(rs, self.percentile, method='higher').item())
        else:
            return 0.0
