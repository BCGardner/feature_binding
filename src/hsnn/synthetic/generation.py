from typing import Any, Callable, List

import numpy as np
import numpy.typing as npt

from hsnn.core.types import SpikeTrains
from hsnn import ops
from .patterns import SpatioTemporalPattern, PolyChronGroup

__all__ = ["generate_poisson_train",
           "generate_uniform_train",
           "generate_poisson_pattern",
           "generate_pngs",
           "repeat_spike_trains"]


def generate_poisson_train(duration: float, rate: float, min_spacing: float = 1, precision: int = 1,
                           max_len: int = 1000, seed: Any = None) -> npt.NDArray[np.float_]:
    rng = np.random.default_rng(seed)
    if rate == 0:
        scale = np.inf
    else:
        scale = 1 / (rate * 1E-3)
    times: list[float] = []
    last_time = -min_spacing
    while len(times) < max_len:
        time = (last_time + min_spacing) + rng.exponential(scale)
        if 0 <= time < duration:
            times.append(time)
            last_time = times[-1]
        elif time >= duration:
            break
    return np.array(times, dtype=np.float_).round(precision)


def generate_uniform_train(duration: float, num_spikes: int, min_spacing: float = 1,
                           precision: int = 1, seed: Any = None) -> npt.NDArray[np.float_]:
    if min_spacing > duration / num_spikes:
        raise ValueError("min_spacing too large")
    rng = np.random.default_rng(seed)
    while True:
        times = np.sort(rng.uniform(0, duration, size=num_spikes))
        if len(times) > 1:
            if np.diff(times).min() >= min_spacing:
                break
        else:
            break
    return np.array(times, dtype=np.float_).round(precision)


def generate_poisson_pattern(duration: float, rate: float, num_nrns: int,
                             seed: Any = None, **kwargs) -> SpatioTemporalPattern:
    rng = np.random.default_rng(seed)
    spike_trains = {}
    for nrn_id in range(num_nrns):
        spike_trains[nrn_id] = generate_poisson_train(duration, rate, seed=rng, **kwargs)
    return SpatioTemporalPattern(duration, num_nrns, spike_trains)


def generate_pngs(max_grps: int, num_nrns: int, nrn_choices: Any, replace: bool = True,
                  distr_func: Callable = generate_poisson_train, seed: Any = None,
                  **distr_kwargs) -> List[PolyChronGroup]:
    rng = np.random.default_rng(seed)
    num_choices = max_grps * num_nrns
    nrn_id_sets = np.split(rng.choice(nrn_choices, size=num_choices, replace=replace), max_grps)
    pngs = []
    for nrn_ids in nrn_id_sets:
        spike_trains = {nrn_id: distr_func(seed=rng, **distr_kwargs) for nrn_id in nrn_ids}
        png = PolyChronGroup(spike_trains, seed=rng)
        if len(png.nrn_ids) > 1:
            pngs.append(png)
    return pngs


def repeat_spike_trains(spike_trains: dict, duration: float, min_spacing: float,
                        distr_func: Callable = generate_poisson_train,
                        seed: Any = None, **kwargs) -> SpikeTrains:
    rng = np.random.default_rng(seed)
    spike_idxs, spike_times = ops.spike_trains_to_events(spike_trains)
    pattern_span = np.max(spike_times) - np.min(spike_times)
    min_spacing += float(pattern_span)
    onset_times = distr_func(duration - pattern_span, min_spacing=min_spacing,
                             seed=rng, **kwargs)
    shifted_times = spike_times[:, np.newaxis] + onset_times
    ret = {}
    for nrn_id, spike_train in zip(spike_idxs, shifted_times):
        ret[nrn_id] = spike_train
    return ret
