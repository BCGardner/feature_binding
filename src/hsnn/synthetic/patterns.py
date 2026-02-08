from typing import Any, Callable, Optional

import numpy as np
import numpy.typing as npt

from hsnn.core.types import SpikeTrains
from hsnn.core.logger import get_logger
from hsnn import ops

__all__ = ["SpatioTemporalPattern", "PolyChronGroup"]

logger = get_logger(__name__)


class SpatioTemporalPattern:
    def __init__(self, duration: float, num_nrns: int, spike_trains: Optional[dict] = None,
                 precision: int = 1):
        self._precision = precision
        self.duration = duration
        self.num_nrns = num_nrns
        self.spike_trains = spike_trains if spike_trains is not None else {}

    @property
    def spike_trains(self) -> SpikeTrains:
        return self._spike_trains

    @spike_trains.setter
    def spike_trains(self, value: dict[int, npt.ArrayLike]):
        nrns = np.arange(self.num_nrns, dtype=np.int_)
        if not set(value.keys()).issubset(nrns):
            raise ValueError("mismatched neuron index")
        self._spike_trains: SpikeTrains = {}
        for nrn in nrns:
            spike_times = np.array(value.get(nrn, []), dtype=np.float_).round(self._precision)
            if np.max(spike_times) >= self.duration:
                logger.warning(f"spike train value(s) clipped")
                mask = ops.get_submask(spike_times, self.duration)
                self._spike_trains[nrn] = spike_times[mask]
            else:
                self._spike_trains[nrn] = spike_times

    def embed_spike_trains(self, spike_trains: dict, inplace: bool = False) -> Optional[dict]:
        spike_ids, spike_times = ops.spike_trains_to_events(self.spike_trains)
        spike_ids_, spike_times_ = ops.spike_trains_to_events(spike_trains)
        if np.max(spike_times_) >= self.duration:
            raise ValueError("embedding spike time(s) exceed the pattern duration")
        spike_ids = np.append(spike_ids, spike_ids_)
        spike_times = np.append(spike_times, spike_times_)
        spike_trains = ops.spike_events_to_trains((spike_ids, spike_times), self.num_nrns)
        if inplace:
            self.spike_trains = spike_trains
            return None
        return spike_trains


class PolyChronGroup:
    def __init__(self, spike_trains: dict, relative_timings: bool = True,
                 precision: int = 1, seed: Any = None) -> None:
        self._rng = np.random.default_rng(seed)
        self._precision = precision
        self.relative_timings = relative_timings
        self.spike_trains = spike_trains

    def __repr__(self) -> str:
        return f'PNG(spike_trains={self.spike_trains}, span={self.span})'

    @property
    def spike_trains(self) -> SpikeTrains:
        return self._spike_trains

    @spike_trains.setter
    def spike_trains(self, value: SpikeTrains):
        spike_ids, spike_times = ops.spike_trains_to_events(value)
        if len(spike_times):
            if self.relative_timings:
                spike_times -= spike_times[0]
            spike_times = spike_times.round(self._precision)
            self._span = round(float(np.max(spike_times) - np.min(spike_times)), self._precision)
            self._spike_trains = ops.spike_events_to_trains((spike_ids, spike_times))
            self._nrn_ids = np.unique(spike_ids)
        else:
            self._span = 0.0
            self._spike_trains = {}
            self._nrn_ids = np.array([], dtype=np.int_)

    @property
    def nrn_ids(self) -> npt.NDArray[np.int_]:
        return self._nrn_ids

    @property
    def span(self) -> float:
        return self._span

    def generate_repeats(self, duration: float, min_spacing: float,
                         distr_func: Callable, **kwargs) -> tuple[npt.NDArray[np.float_], SpikeTrains]:
        nrn_ids, spike_times = ops.spike_trains_to_events(self.spike_trains)
        min_spacing += self.span
        onset_times = distr_func(duration - self.span, min_spacing=min_spacing,
                                 seed=self._rng, **kwargs)
        nrn_ids = np.repeat(nrn_ids, len(onset_times))
        shifted_times = np.ravel(spike_times[:, np.newaxis] + onset_times).round(self._precision)
        return onset_times, ops.spike_events_to_trains((nrn_ids, shifted_times))
