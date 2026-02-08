from copy import deepcopy
from typing import Dict, List, Optional, Sequence

import numpy as np
import numpy.typing as npt

from .types import SpikeEvents, SpikeTrains, FiringRates

__all__ = ["SpikeRecord", "StateRecord"]


def _isvalid_indices(idxs: npt.ArrayLike, num_nrns: Optional[int]) -> bool:
    if num_nrns is None or not np.any(idxs):
        return True
    return np.max(idxs) < num_nrns


def _events_to_trains(idxs: npt.ArrayLike, times: npt.ArrayLike,
                      num_nrns: Optional[int] = None) -> SpikeTrains:
    if not _isvalid_indices(idxs, num_nrns):
        raise ValueError(f"insufficient number of neurons: '{num_nrns}'")
    idxs = np.asarray(idxs, dtype=np.int_)
    times = np.asarray(times, dtype=np.float_)
    idxs_unique = np.unique(idxs)
    if num_nrns is not None:
        keys = np.arange(num_nrns, dtype=np.int_)
    else:
        keys = idxs_unique
    spike_trains: SpikeTrains = {}
    for key in keys:
        if key in idxs_unique:
            args = np.flatnonzero(idxs == key)
            spike_trains[key] = times[args]
        else:
            spike_trains[key] = np.array([], dtype=np.float_)
        spike_trains[key] = np.sort(spike_trains[key])
    return spike_trains


class SpikeRecord:
    def __init__(self, num_nrns: int, duration: float, idxs: npt.ArrayLike,
                 times: npt.ArrayLike, name: Optional[str] = None) -> None:
        idxs_ = np.asarray(idxs, dtype=np.int_)
        times_ = np.asarray(times, dtype=np.float_)
        if len(times_) and max(times_) >= duration:
            raise ValueError("spike time(s) out of bounds")
        if len(idxs_) and max(idxs_) >= num_nrns:
            raise ValueError(f"insufficient size for num_nrns")
        self.num_nrns = num_nrns
        self.duration = duration
        self.name = name
        self._idxs: npt.NDArray[np.int_] = idxs_
        self._times: npt.NDArray[np.float_] = times_

    @property
    def spike_events(self) -> SpikeEvents:
        return self._idxs, self._times

    @property
    def spike_trains(self) -> SpikeTrains:
        return _events_to_trains(self._idxs, self._times, self.num_nrns)

    @property
    def rates(self) -> FiringRates:
        rates = np.zeros(self.num_nrns, dtype=np.float_)
        nrn_ids, counts = np.unique(self.spike_events[0], return_counts=True)
        rates[nrn_ids] = counts / self.duration * 1E3
        return np.arange(self.num_nrns, dtype=np.int_), rates


class StateRecord:
    def __init__(self, num_nrns: int, duration: float, name: Optional[str] = None,
                 **kwargs) -> None:
        self.num_nrns = num_nrns
        self.duration = duration
        self.name = name
        self._states = dict(**kwargs)
        for key, val in self._states.items():
            setattr(self, key, val)

    @property
    def variables(self) -> List[str]:
        return list(self._states.keys())

    def get_states(self) -> Dict[str, npt.ArrayLike]:
        return deepcopy(self._states)
