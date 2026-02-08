from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import numpy.typing as npt

__all__ = ["PNG"]


@dataclass
class PNG:
    layers: npt.NDArray[np.int_]
    nrns: npt.NDArray[np.int_]
    lags: npt.NDArray[np.float_]
    times: npt.NDArray[np.float_] = field(repr=False)
    num_occ: int = field(init=False)
    imgs: Optional[npt.NDArray[np.int_]] = field(default=None, repr=False)
    reps: Optional[npt.NDArray[np.int_]] = field(default=None, repr=False)
    times_rel: Optional[npt.NDArray[np.float_]] = field(default=None, repr=False)

    def __post_init__(self):
        self.num_occ = len(self.times)

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, PNG):
            return hash(self) == hash(__value)
        return False

    def __hash__(self) -> int:
        indices = np.lexsort((self.nrns, self.layers, self.lags))
        attrs = [tuple(elem[indices]) for elem in [self.layers, self.nrns]]
        return hash(tuple(attrs))


class Refine(ABC):
    @abstractmethod
    def __call__(self, pngs: Sequence[PNG]) -> Sequence[PNG]:
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
