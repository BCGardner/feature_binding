from copy import deepcopy
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from hsnn.core import SpikeRecord
from hsnn.core.types import SpikeTrains
from .conversion import spike_events_to_trains
from .filtering import select_nrns, mask_recording

__all__ = [
    "concat_records",
    "concatenate_spike_trains",
    "hstack_spike_trains"
]


def concat_records(records: Sequence[xr.DataArray], new_dim: str,
                   dim_vals: Optional[Iterable] = None,
                   attrs: Optional[Mapping] = None) -> xr.DataArray:
    coords = range(len(records)) if dim_vals is None else dim_vals
    records_: xr.DataArray = \
        xr.concat(records, dim=pd.Index(coords, name=new_dim), fill_value=None)
    if attrs is None:
        attrs = {}
    attrs_ = deepcopy(records_.attrs)
    attrs_.update(**attrs)
    return records_.assign_attrs(attrs_)


def concatenate_spike_trains(spike_records: xr.DataArray, nrns: Optional[npt.ArrayLike] = None,
                             duration: Optional[float] = None, offset: float = 0,
                             separation: float = 0) -> SpikeTrains:
    _valid_dims = {'img', 'rep'}
    if not set(spike_records.dims).issubset(_valid_dims):
        raise ValueError(f"valid dim(s) include: {_valid_dims}")
    spike_records = spike_records.transpose(..., 'rep')
    duration_: float = spike_records.item(0).duration if duration is None else duration
    idxs, times =  np.array([], dtype=np.int_), np.array([], dtype=np.float64)
    for idx, rec in enumerate(spike_records.values.flat):
        assert isinstance(rec, SpikeRecord)
        t_shift = (duration_ + separation) * idx
        spike_events = select_nrns(rec.spike_events, nrns) if nrns is not None else rec.spike_events
        spike_events = mask_recording(spike_events, duration_, offset, relative_times=True)
        idxs = np.concatenate([idxs, spike_events[0]])
        times = np.concatenate([times, spike_events[1] + t_shift])
    spike_trains = spike_events_to_trains((idxs, times))
    if nrns is not None:
        spike_trains = select_nrns(spike_trains, nrns)
    return spike_trains


def hstack_spike_trains(spike_trains1: SpikeTrains, spike_trains2: SpikeTrains,
                        duration: float) -> SpikeTrains:
    """Stack spike trains by appending spike times in `spike_trains2`.

    Args:
        spike_trains1 (SpikeTrains): Initial spike trains.
        spike_trains2 (SpikeTrains): Spike trains to append.
        duration (float): Duration of initial spike trains.

    Returns:
        SpikeTrains: Concatenated spike trains, containing the common set of nrn IDs.
    """
    common_ids = sorted(set(spike_trains1.keys()) | (spike_trains2.keys()))
    spike_trains = {i: np.array([], dtype=np.float64) for i in common_ids}
    for i, spike_times in spike_trains1.items():
        spike_trains[i] = spike_times.copy()
    for i, spike_times in spike_trains2.items():
        spike_trains[i] = np.concatenate([spike_trains[i], spike_times + duration])
    return spike_trains
