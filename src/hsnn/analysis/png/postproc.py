"""Postprocessing for general analysis of detected PNGs.
"""

import numpy as np
import numpy.typing as npt
import xarray as xr

from hsnn.core.record import SpikeRecord
from hsnn.core.types import SpikeTrains
from hsnn import ops
from .base import PNG

__all__ = [
    "get_spike_trains",
    "get_polygrp_trains",
    "gather_recordings",
    "get_closest_indices"
]


def get_spike_trains(polygrp: PNG, img: int, rep: int, records: xr.DataArray,
                     duration: float | None = None, t_start: float = 0.0,
                     relative_times: bool = False) -> SpikeTrains:
    """Retrieve original spike train recordings associated with `polygrp` for a
    given (`img`, `rep`). The returned indices correspond to polygrp indices.

    Args:
        polygrp (PNG): Query PNG containing `nrns` and `layers`.
        img (int): Target image ID.
        rep (int): Target repetition ID.
        records (xr.DataArray): Original spike records, dims (`rep`, `img`, `layer`, `nrn_cls`).
        duration (float | None, optional): Retrieve spikes recorded for this duration. Defaults to None.
        t_start (float, optional): Retrieve spikes recorded after this value. Defaults to 0.0.
        relative_times (bool, optional): Adjust spike timings as relative to `t_start`. Defaults to False.

    Returns:
        SpikeTrains: Original spike trains associated with polygrp.
    """
    spike_seqs = gather_recordings(records, img, rep, polygrp)
    spike_trains = {np.int64(i): spike_times for i, spike_times in enumerate(spike_seqs)}
    _duration: float = records.item(0).duration if duration is None else duration
    return ops.filtering.mask_recording(spike_trains, _duration, t_start,
                                        relative_times, len(spike_trains))


def get_polygrp_trains(polygrp: PNG, img: int, rep: int, records: xr.DataArray,
                       duration: float | None = None, t_start: float = 0.0,
                       relative_times: bool = False) -> SpikeTrains:
    """Retrieve original spike train recordings that are specific to the activations
    of `polygrp` for a given (`img`, `rep`). The returned indices correspond to polygrp indices.

    Args:
        polygrp (PNG): Query PNG containing `nrns` and `layers`.
        img (int): Target image ID.
        rep (int): Target repetition ID.
        records (xr.DataArray): Original spike records, dims (`rep`, `img`, `layer`, `nrn_cls`).
        duration (float | None, optional): Retrieve spikes recorded for this duration. Defaults to None.
        t_start (float, optional): Retrieve spikes recorded after this value. Defaults to 0.0.
        relative_times (bool, optional): Adjust spike timings as relative to `t_start`. Defaults to False.

    Returns:
        SpikeTrains: Original spike trains corresponding to polygrp activations.
    """
    # Neuronal spike trains
    spike_seqs = gather_recordings(records, img, rep, polygrp)
    spike_trains = {np.int64(i): spike_times for i, spike_times in enumerate(spike_seqs)}
    # PNG spike trains
    indices = get_closest_indices(spike_seqs, img, rep, polygrp)
    polygrp_trains = {np.int64(i): spike_trains[i][indices[i]] for i in spike_trains.keys()}
    _duration: float = records.item(0).duration if duration is None else duration
    return ops.filtering.mask_recording(polygrp_trains, _duration, t_start,
                                        relative_times, len(polygrp_trains))


def gather_recordings(records: xr.DataArray, img: int, rep: int,
                      polygrp: PNG) -> list[npt.NDArray[np.float64]]:
    """Selects spike recordings corresponding to `layers` and `nrns` in polygrp
    for a specific `img` and `rep`. The returned spike times are in the original
    time frame of the recorded experiments.

    Args:
        records (xr.DataArray): Spike records, dims (`rep`, `img`, `layer`, `nrn_cls`).
        img (int): Target image ID.
        rep (int): Target repetition ID.
        polygrp (PNG): Query PNG containing `nrns` and `layers`.

    Returns:
        list[npt.NDArray[np.float64]]: List containing spike time sequences, in same order as polygrp.
    """
    spike_seqs = []
    for layer in np.unique(polygrp.layers):
        record: SpikeRecord = records.sel(rep=rep, img=img, layer=layer, nrn_cls='EXC').item()
        mask = polygrp.layers == layer
        for nrn in polygrp.nrns[mask]:
            spike_seqs.append(record.spike_trains[nrn])
    return spike_seqs


def get_closest_indices(spike_seqs: list[npt.NDArray[np.float64]], img: int, rep: int,
                        polygrp: PNG) -> list[npt.NDArray[np.int_]]:
    """Gets indices of spike times that are closest to the SPADE-approximated times.

    Args:
        spike_seqs (list[npt.NDArray[np.float64]]): List of spike time sequences.
        img (int): Target image ID.
        rep (int): Target repetition ID.
        polygrp (PNG): Query PNG containing `nrns`, `layers` and `times_rel`.

    Returns:
        list[npt.NDArray[np.int_]]: List of closest recording indices, ordered according to polygrp.
    """
    assert polygrp.times_rel is not None
    mask = (polygrp.imgs == img) & (polygrp.reps == rep)
    png_spikes = polygrp.times_rel[mask] + polygrp.lags[:, np.newaxis]  # shape (num_nrns, num_occ)
    spike_args = []
    for spike_times, times_ref in zip(spike_seqs, png_spikes):
        spike_args.append(_argclose_all(spike_times, times_ref))
    return spike_args


def argclose(times: npt.NDArray[np.float64], times_ref: npt.NDArray[np.float64],
             tol: float = 3.0) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    """Returns the indices in `times` and `times_ref` that are within `tol` of each other.

    Args:
        times (npt.NDArray[np.float64]): Sequence of times, shape (num_times,).
        times_ref (npt.NDArray[np.float64]): Sequence of ref. times, shape (num_ref,).
        tol (float, optional): Maximum separation between paired timings. Defaults to 3.0.

    Returns:
        tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]: Indices of paired timings.
    """
    idxs_pre, idxs_post = [], []
    for i, t_pre in enumerate(times):
        diffs = np.abs(t_pre - times_ref)
        j = np.argmin(diffs)
        if diffs[j] <= tol and j not in idxs_post:
            idxs_pre.append(i)
            idxs_post.append(j)
    pre_idxs = np.asarray(idxs_pre)
    post_idxs = np.asarray(idxs_post)
    assert len(pre_idxs) == len(post_idxs)
    return pre_idxs, post_idxs


def _argclose_all(spike_times: npt.NDArray[np.float64], times_ref: npt.NDArray[np.float64],
                  tol: float = 3.0) -> npt.NDArray[np.int_]:
    """Returns the indices of spikes in `spike_times` paired with those in `times_ref`.
    All spikes in `times_ref` must be paired with those in `spike_times` within `tol`.

    Args:
        spike_times (npt.NDArray[np.float64]): Sequence of times, shape (num_spikes,).
        times_ref (npt.NDArray[np.float64]): Sequence of ref. times, shape (num_occ,)
        tol (float, optional): Maximum separation between paired spikes. Defaults to 3.0.

    Raises:
        ValueError: Not all spikes in `times_ref` can be paired.

    Returns:
        npt.NDArray[np.int_]: Indices of paired spikes in `spike_times`.
    """
    if len(times_ref) == 0:
        return np.array([], dtype=np.int_)
    delta_ts = np.abs(spike_times - times_ref[:, np.newaxis])
    if max(delta_ts.min(axis=1)) > tol:
        raise ValueError(f"Max difference '{delta_ts.max()}' exceeds `tol`")
    return np.argmin(delta_ts, axis=1)
