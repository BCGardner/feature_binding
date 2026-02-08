from typing import Optional

import numpy as np
import numpy.typing as npt
from brian2 import Synapses
from brian2.units import Quantity

from ._base import BaseConnector
from . import samplers as S

__all__ = ["PatchConnector", "GaussianConnector", "DenseConnector"]


def _assert_spatial(synapses: Synapses):
    for group in (synapses.source, synapses.target):
        for attribute in ('x', 'y'):
            assert hasattr(group, attribute), f"{group.name} missing '{attribute}'"


class PatchConnector(BaseConnector):
    def sample(self, points: npt.NDArray[np.float_], point_ref: npt.NDArray[np.float_],
               **kwargs) -> npt.NDArray[np.int_]:
        return S.uniform(points, **kwargs)

    def connect(self, synapses: Synapses, **kwargs) -> None:
        p_conn: float = kwargs.get('p_conn', 1.0)
        num_conn: Optional[int] = kwargs.get('num_conn', None)
        fan_in: float = kwargs['fan_in']
        radius: float = fan_in * (synapses.source.x[1] - synapses.source.x[0])
        i, j = self._get_indices(synapses.source, synapses.target, radius, p_conn=p_conn,
                                 num_conn=num_conn)
        synapses.connect(i=i, j=j)


class GaussianConnector(BaseConnector):
    def sample(self, points: npt.NDArray[np.float_], point_ref: npt.NDArray[np.float_],
               **kwargs) -> npt.NDArray[np.int_]:
        return S.gaussian(points, point_ref, **kwargs)

    def connect(self, synapses: Synapses, **kwargs) -> None:
        p_conn: float = kwargs.get('p_conn', 1.0)
        num_conn: Optional[int] = kwargs.get('num_conn', None)
        stdev: float = kwargs['fan_in'] * (synapses.source.x[1] - synapses.source.x[0])
        cutoff: float = 3 * stdev
        i, j = self._get_indices(synapses.source, synapses.target, cutoff,
                                 stdev=stdev, p_conn=p_conn, num_conn=num_conn)
        synapses.connect(i=i, j=j)


class DenseConnector(BaseConnector):
    def sample(self, points: npt.NDArray[np.float_], point_ref: npt.NDArray[np.float_],
               **kwargs) -> npt.NDArray[np.int_]:
        raise NotImplementedError()

    def connect(self, synapses: Synapses, **kwargs) -> None:
        p_conn: float = kwargs.get('p_conn', 1.0)
        n: int = kwargs.get('n', 1)
        synapses.connect(p=p_conn, n=n)


def connect_spatial(synapses: Synapses, p_conn: float, fan_in: float) -> None:
    _assert_spatial(synapses)
    radius: Quantity = fan_in * (synapses.source.x[1] - synapses.source.x[0])
    synapses.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) <= radius',
                     p=p_conn, namespace={'radius': radius})
