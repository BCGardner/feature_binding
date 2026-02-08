from copy import copy
from typing import Any, Optional, Sequence

import neo
import numpy as np
import pandas as pd
import quantities as pq
from elephant.spade import spade

from hsnn.core.logger import get_logger
from hsnn.core.types import SpikeEvents, SpikeTrains, SpikePatterns
from hsnn.ops import conversion
from .base import PNG, Refine
from ._utils import PrintManager

__all__ = ["SpadeMethod"]

logger = get_logger(__name__)

_DEFAULT_KWARGS = {
    'bin_size': 1,
    'winlen': 20,
    'min_spikes': 3,
    'min_occ': 3,
    'max_spikes': 3,
    'min_neu': 3,
    'n_surr': 0,
    'dither': 5,
    'spectrum': '3d#',
    'alpha': None,
    'psr_param': None,
    'output_format': 'patterns'
}


class SpadeMethod:
    def __init__(self, verbose: bool = False, refine: Optional[Refine] = None,
                 **spade_kwargs) -> None:
        self._verbose = verbose
        self._default_kwargs = copy(_DEFAULT_KWARGS)
        self.refine = refine
        self.default_kwargs = spade_kwargs

    @property
    def default_kwargs(self):
        return copy(self._default_kwargs)

    @default_kwargs.setter
    def default_kwargs(self, kwargs: dict):
        _update_kwargs(self._default_kwargs, **kwargs)

    def __call__(self, patterns: SpikePatterns, duration: float,
                 **spade_kwargs) -> Sequence[PNG]:
        neo_map = self._get_neo_map(patterns)
        spike_trains_neo = self._convert_patterns(patterns, duration)
        spade_kwargs_ = self.default_kwargs
        _update_kwargs(spade_kwargs_, **spade_kwargs)
        bin_size = spade_kwargs_.pop('bin_size')
        dither = spade_kwargs_.pop('dither', 5) * pq.ms
        output_format = spade_kwargs_['output_format']
        # results: layers, neurons, lags, times
        # target spade kwargs: bin_size (ms), max_delay (ms), min/max spikes, min_occ / fraction of reps
        num_active_neu = sum([len(spike_train) > 0 for spike_train in spike_trains_neo])
        if num_active_neu > 1 and num_active_neu >= spade_kwargs_['min_neu']:
            with PrintManager(self._verbose):
                results = spade(spike_trains_neo, bin_size=bin_size*pq.ms,
                                dither=dither, **spade_kwargs_)[output_format] # type: ignore
                results = _drop_invalid_patterns(results, spade_kwargs_)
        else:
            results = []
        pngs = self._convert_results(results, neo_map)
        if self.refine is None:
            return pngs
        else:
            return self.refine(pngs)

    def _convert_patterns(self, patterns: SpikePatterns, duration: float) -> list[neo.SpikeTrain]:
        spike_trains_neo: list[neo.SpikeTrain] = []
        for layer in patterns.keys():
            spike_trains_neo.extend(_as_neo_spike_trains(patterns[layer], duration))
        return spike_trains_neo

    def _get_neo_map(self, patterns: SpikePatterns) -> pd.DataFrame:
        data: dict[str, list] = {'layer': [], 'nrn': []}
        for layer in patterns.keys():
            for nrn in patterns[layer].keys():
                data['layer'].append(layer)
                data['nrn'].append(nrn)
        return pd.DataFrame(data)

    def _convert_results(self, results: list[dict], neo_map: pd.DataFrame) -> list[PNG]:
        results_: list[PNG] = []
        for result in results:
            df = neo_map.loc[result['neurons']]
            layers, nrns = df['layer'].values, df['nrn'].values
            results_.append(
                PNG(layers, nrns, np.concatenate([[0], np.array(result['lags'])]),
                    np.array(result['times']))
            )
        return sorted(results_, key=lambda x: len(x.times), reverse=True)


def _as_neo_spike_trains(recording: SpikeEvents | SpikeTrains, duration: float) -> list[neo.SpikeTrain]:
    spike_trains = conversion.as_spike_trains(recording)
    return [neo.SpikeTrain(spike_train*pq.ms, duration*pq.ms)
            for spike_train in spike_trains.values()]


def _update_kwargs(dst: dict[str, Any], **kwargs) -> None:
    for key, val in kwargs.items():
        if key in dst.keys():
            dst[key] = val
        else:
            raise KeyError(f"invalid key '{key}', valid choices include: {list(dst.keys())}")


def _drop_invalid_patterns(patterns: list[dict], spade_kwargs: dict) -> list[dict]:
    min_spikes = spade_kwargs['min_spikes']
    min_occ = spade_kwargs['min_occ']
    max_spikes = spade_kwargs['max_spikes']
    min_neu = spade_kwargs['min_neu']

    def isvalid(pattern: dict) -> bool:
        num_spikes = len(pattern['neurons'])
        if len(pattern['times']) < min_occ:
            return False
        if num_spikes < min_spikes:
            return False
        if max_spikes is not None and num_spikes > max_spikes:
            return False
        if len(set(pattern['neurons'])) < min_neu:
            return False
        return True

    patterns_ = []
    for pattern in patterns:
        if isvalid(pattern):
            patterns_.append(pattern)
        else:
            logger.warn(f"Invalid SPADE result: {pattern}")
    return patterns_
