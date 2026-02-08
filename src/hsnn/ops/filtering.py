from typing import Optional

import numpy as np
import numpy.typing as npt

from ._base import assert_recording
from .conversion import as_spike_events, spike_events_to_trains, get_rates
from ..core.types import Recording, SpikeEvents, SpikeTrains, FiringRates

__all__ = [
    "get_submask",
    "mask_recording",
    "mask_rates",
    "select_nrns"
]


def get_submask(spike_times: npt.ArrayLike, duration: float, t_start: float = 0) -> npt.NDArray[np.bool_]:
    spike_times = np.asarray(spike_times)
    return (spike_times >= t_start) & (spike_times < t_start + duration)


def mask_recording(recording: Recording, duration: float, t_start: float = 0,
                   relative_times: bool = False, num_nrns: Optional[int] = None) -> Recording:
    spike_ids, spike_times = as_spike_events(recording)
    mask = get_submask(spike_times, duration, t_start)
    spike_ids, spike_times = spike_ids[mask], spike_times[mask]
    if relative_times:
        spike_times = spike_times - t_start
    if isinstance(recording, dict):
        return spike_events_to_trains((spike_ids, spike_times), num_nrns)
    return spike_ids, spike_times


def mask_rates(recording: SpikeEvents | SpikeTrains, duration: float,
               t_start: float = 0, num_nrns: Optional[int] = None) -> FiringRates:
    spike_events = mask_recording(as_spike_events(recording), duration, t_start, num_nrns=num_nrns)
    return get_rates(spike_events, duration, num_nrns)


def select_nrns(recording: Recording, nrn_ids: npt.ArrayLike) -> Recording:
    assert_recording(recording)
    nrn_ids = np.asarray(np.sort(nrn_ids), dtype=np.int_)
    if isinstance(recording, dict):
        return {nrn_id: recording.get(nrn_id, np.array([], np.float64)) for nrn_id in nrn_ids}
    idxs, vals = recording
    mask = np.isin(idxs, nrn_ids)
    return idxs[mask], vals[mask]
