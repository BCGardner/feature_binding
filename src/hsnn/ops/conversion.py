from typing import List, Optional

import numpy as np
import numpy.typing as npt
import xarray as xr
import sparse

from ._base import assert_recording
from hsnn.core import SpikeRecord
from hsnn.core.types import SpikeEvents, SpikeTrains, FiringRates

__all__ = [
    "spike_trains_to_events",
    "spike_events_to_trains",
    "as_spike_events",
    "as_spike_trains",
    "get_rates",
    "create_spike_record",
    "to_sparse_xarray",
    "sparse_to_spike_trains",
    "layers_to_proj"
]


def spike_events_to_trains(spike_events: SpikeEvents, num_nrns: Optional[int] = None) -> SpikeTrains:
    if not _isvalid_indices(spike_events[0], num_nrns):
        raise ValueError(f"insufficient number of neurons: '{num_nrns}'")
    idxs = np.asarray(spike_events[0], dtype=np.int_)
    times = np.asarray(spike_events[1], dtype=np.float_)
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


def spike_trains_to_events(spike_trains: SpikeTrains) -> SpikeEvents:
    if len(spike_trains) == 0:
        return np.array([], dtype=np.int_), np.array([], dtype=np.float_)
    idxs: List[np.int_] = []
    times: List[np.float_] = []
    for idx, spike_train in spike_trains.items():
        if len(spike_train):
            idxs.extend([idx] * len(spike_train))
            times.extend(spike_train)
    idxs_ = np.asarray(idxs).astype(np.int_)
    times_ = np.asarray(times).astype(np.float_)
    args = np.lexsort((idxs_, times_))
    return idxs_[args], times_[args]


def as_spike_events(recording: SpikeEvents | SpikeTrains) -> SpikeEvents:
    assert_recording(recording)
    if isinstance(recording, dict):
        return spike_trains_to_events(recording)
    return np.asarray(recording[0], dtype=np.int_), np.asarray(recording[1], dtype=np.float_)


def as_spike_trains(recording: SpikeEvents | SpikeTrains, num_nrns: Optional[int] = None) -> SpikeTrains:
    assert_recording(recording)
    if isinstance(recording, dict):
        if num_nrns is None:
            return recording
        else:
            return spike_events_to_trains(spike_trains_to_events(recording), num_nrns=num_nrns)
    return spike_events_to_trains(recording, num_nrns=num_nrns)


def get_rates(recording: SpikeEvents | SpikeTrains, duration: float, num_nrns: Optional[int] = None) -> FiringRates:
    spike_events = as_spike_events(recording)
    nrn_ids, counts = np.unique(spike_events[0], return_counts=True)
    rates = counts / duration * 1E3
    if num_nrns is None:
        return nrn_ids, rates
    else:
        rates_ = np.zeros(num_nrns, dtype=np.float_)
        rates_[nrn_ids] = rates
        return np.arange(num_nrns, dtype=np.int_), rates_


def create_spike_record(num_nrns: int, duration: float, recording: SpikeEvents | SpikeTrains,
                        name: Optional[str] = None) -> SpikeRecord:
    recording_ = as_spike_events(recording)
    return SpikeRecord(num_nrns, duration, recording_[0], recording_[1], name)


def to_sparse_xarray(records: xr.DataArray) -> xr.DataArray:
    dims = ('img', 'rep', 'layer')
    if set(records.dims) != set(dims):
        raise ValueError(f"Expected dimensions: {dims}, got: {records.dims}")
    stacked = records.stack(all_dims=dims)
    spike_record: SpikeRecord = stacked.item(0)
    num_nrns = spike_record.num_nrns
    duration = records.attrs['duration']
    dt = records.attrs['dt']

    matrices = []
    for spike_record in stacked.values:
        s = sparse.COO(_as_binary_matrix(
            spike_record, dt=dt, num_nrns=num_nrns, duration=duration
        ))
        matrices.append(s)
    data = sparse.stack(matrices)

    sparse_array = xr.DataArray(
        data,
        dims=('all_dims', 'nrn', 'time'),
        coords=dict(
            all_dims=stacked.all_dims,
            nrn=np.arange(num_nrns),
            time=np.arange(round(duration / dt)) * dt
        )
    ).unstack("all_dims").transpose('img', 'rep', 'layer', 'nrn', 'time')
    sparse_array.attrs = records.attrs.copy()
    return sparse_array


def sparse_to_spike_trains(sparse_array: xr.DataArray) -> SpikeTrains:
    matrix = sparse_array.transpose('nrn', 'time').to_numpy()
    nrn_ids = sparse_array.coords['nrn'].values
    rows, cols = np.nonzero(matrix)
    spike_times = cols * sparse_array.attrs['dt']

    spike_trains = {}
    for i in set(rows):
        mask = rows == i
        spike_trains[nrn_ids[i]] = spike_times[mask]
    return spike_trains


def layers_to_proj(layer_post: int, layer_pre: int) -> str:
    """Gets the projection name connecting the given layers.

    Args:
        layer_post (int): Postsynaptic layer.
        layer_pre (int): Presynaptic layer

    Raises:
        ValueError: Raised if no projection exists between these layers.

    Returns:
        str: Projection name.
    """
    if layer_post == layer_pre:
        return 'E2E'
    elif layer_post == layer_pre + 1:
        return 'FF'
    elif layer_post == layer_pre - 1:
        return 'FB'
    else:
        raise ValueError(f"No projection between layers {layer_pre} and {layer_post}")


def _isvalid_indices(idxs: npt.ArrayLike, num_nrns: Optional[int]) -> bool:
    if num_nrns is None or not np.any(idxs):
        return True
    return np.max(idxs) < num_nrns


def _as_binary_matrix(
    record: SpikeRecord, dt: float = 0.1,
    num_nrns: int | None = None, duration: float | None = None
) -> npt.NDArray[np.int8]:
    if num_nrns is not None:
        assert record.num_nrns == num_nrns
    if duration is not None:
        assert record.duration == duration
    dims = (record.num_nrns, round(record.duration / dt))
    spikes_array = np.zeros(dims, dtype=np.int8)
    nrn_ids = record.spike_events[0]
    spike_indices = np.round(record.spike_events[1] / dt).astype(int)
    indices = np.ravel_multi_index([nrn_ids, spike_indices], dims=dims)
    spikes_array.flat[indices] = 1
    return spikes_array
