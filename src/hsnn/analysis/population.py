from itertools import product
from typing import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from scipy.spatial import KDTree

from hsnn.ops import conversion
from hsnn.core import SpikeRecord
from hsnn.core.logger import get_logger
from hsnn.core.types import Recording

__all__ = ["Population", "get_coords"]

logger = get_logger(__name__)


class Population:
    def __init__(self, duration: float, bin_size: int = 2,
                 spatial_dims: tuple[int, int] = (64, 64)) -> None:
        self._duration = duration
        self._bin_size = bin_size
        self._build_kdtree(spatial_dims)

    def _build_kdtree(self, spatial_dims: tuple[int, int]):
        self.spatial_dims = spatial_dims
        xs, ys = get_coords(range(self._max_nrns), spatial_dims)
        self._kdtree_all = KDTree(np.vstack((xs, ys)).T)

    @property
    def spatial_dims(self):
        return self._spatial_dims

    @property
    def time_points(self):
        return np.arange(0, self._duration, self._bin_size)

    @spatial_dims.setter
    def spatial_dims(self, value: tuple[int, int]):
        self._spatial_dims = value
        self._max_nrns = int(np.prod(value))

    def global_activity(self, recording: Recording) -> npt.NDArray[np.float64]:
        spike_events = conversion.as_spike_events(recording)
        return _binned_activity(spike_events[1], self._duration, self._max_nrns,
                                self._bin_size)

    def local_activity(self, recording: Recording, nrn_id: int, radius: float):
        spike_events = conversion.as_spike_events(recording)
        elig_ids = self.get_local_ids(nrn_id, radius)
        mask = np.isin(spike_events[0], elig_ids)  # retain recorded spikes of eligible neurons
        logger.info(f"{len(set(spike_events[0][mask]))} / {len(elig_ids)}")
        return _binned_activity(spike_events[1][mask], self._duration,
                                len(elig_ids), self._bin_size)

    def get_local_ids(self, nrn_id: int, radius: float) -> list[int]:
        xs, ys = get_coords([nrn_id], self._spatial_dims)
        return self._kdtree_all.query_ball_point((xs[0], ys[0]), r=radius, p=2.0)


def get_activity_array(records: xr.DataArray, nrn_id: int, layer: int,
                       radius: int = 10, duration: float | None = None,
                       bin_size: int = 2, spatial_dims: tuple[int, int] = (64, 64)) -> xr.DataArray:
    num_reps = len(records['rep'])
    num_imgs = len(records['img'])
    _duration: float = duration if duration is not None else records.item(0).duration
    pop = Population(_duration, bin_size, spatial_dims)

    data = np.full((len(pop.time_points), num_reps, num_imgs), np.nan)
    for rep, img in product(range(num_reps), range(num_imgs)):
        record: SpikeRecord = records.sel(rep=rep, img=img, layer=3, nrn_cls='EXC').item()
        activity_local = pop.local_activity(record.spike_events, nrn_id, radius)
        data[:, rep, img] = activity_local
    return xr.DataArray(
        data,
        dims=['t', 'rep', 'img'],
        coords=dict(
            t=pop.time_points,
            rep=range(num_reps),
            img=range(num_imgs)
        ),
        attrs={
            'unit': 'hertz',
            'description': "Population activity (localised)",
            'layer': layer,
            'nrn': nrn_id
        }
    )


def get_coords(
    nrn_ids: Sequence[int], spatial_dims: tuple[int, int] = (64, 64)
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    ys, xs = np.unravel_index(nrn_ids, spatial_dims)
    return xs, ys


def _binned_activity(spike_times: npt.NDArray[np.float64], duration: float,
                     max_nrns: int, bin_size: int = 2) -> npt.NDArray[np.float64]:
    bins = np.arange(0, duration + bin_size, bin_size)
    population_activity, _ = np.histogram(spike_times, bins=bins)
    return population_activity / (bin_size * max_nrns) * 1E3
