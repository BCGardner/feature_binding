from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from brian2 import CodeRunner, Network, NeuronGroup, Synapses, SpikeMonitor, StateMonitor
from brian2.units import msecond

from ..groups import GroupFactory
from .. import symbols
from ...definitions import NeuronClass, Projection
from ...interfaces import ILayer, IStimulus
from ...record import SpikeRecord, StateRecord
from ...types import SynParams

__all__ = ["BaseLayer", "stimulus_registry"]

stimulus_registry: Dict[str, Type[StimulusLayer]] = {}


def _get_submask(times: npt.ArrayLike, duration: float, t_rel: float = 0) -> npt.NDArray[np.bool_]:
    times = np.asarray(times, dtype=np.float_)
    return np.logical_and((times >= t_rel), (times < t_rel + duration))


def _get_spike_record(spike_mon: SpikeMonitor, duration: float, t_rel: float = 0) -> SpikeRecord:
    mask = _get_submask(spike_mon.t / msecond, duration, t_rel)
    idxs = spike_mon.i[:][mask]
    times = spike_mon.t[:][mask] / msecond - t_rel
    return SpikeRecord(spike_mon.source.N, duration, idxs, times, spike_mon.source.name) # type: ignore


def _get_state_record(state_mon: StateMonitor, duration: float, t_rel: float = 0) -> StateRecord:
    t = state_mon.t / msecond
    mask = _get_submask(t, duration, t_rel)
    states = {'t': t[mask]}
    for key in state_mon.recorded_variables.keys():
        unit = symbols.retrieve_unit(key)
        if unit is not None:
            val = getattr(state_mon, key) / unit
            states[key] = val[..., mask]
    return StateRecord(state_mon.source.N, duration, state_mon.source.name, **states)


class BaseLayer(ABC, ILayer):
    def __init__(self, name: str, group_shapes: Dict[NeuronClass, Tuple],
                 network: Network) -> None:
        self.name = name
        self.group_shapes = group_shapes
        self._neurons: Dict[NeuronClass, NeuronGroup] = {}
        self._synapses: Dict[Projection, Synapses] = {}
        self._spike_mons: Dict[NeuronClass, SpikeMonitor] = {}
        self._state_mons: Dict[NeuronClass, StateMonitor] = {}
        self._clampers: Dict[NeuronClass, CodeRunner] = {}
        self._network = network

    @property
    def monitor_spikes(self) -> bool:
        return all([spike_mon.active for spike_mon in self._spike_mons.values()])

    @monitor_spikes.setter
    def monitor_spikes(self, value: bool):  # type: ignore
        for spike_mon in self._spike_mons.values():
            spike_mon.active = value

    @property
    def monitor_states(self) -> bool:
        return all([state_mon.active for state_mon in self._state_mons.values()])

    @monitor_states.setter
    def monitor_states(self, value: bool):  # type: ignore
        for state_mon in self._state_mons.values():
            state_mon.active = value

    @property
    def clamp_voltages(self) -> bool:
        return all([clamper.active for clamper in self._clampers.values()])

    @clamp_voltages.setter
    def clamp_voltages(self, value: bool):  # type: ignore
        for clamper in self._clampers.values():
            clamper.active = value

    @property
    def projections(self) -> Iterable[Projection]:
        return tuple(self._synapses.keys())

    def get_spikes(self, duration: float, t_rel: float = 0) -> xr.DataArray:
        records = np.array([_get_spike_record(spike_mon, duration, t_rel)
                            for spike_mon in self._spike_mons.values()], dtype=object)
        nrn_classes = [nrn_cls.name for nrn_cls in self._spike_mons.keys()]
        return xr.DataArray(data=records, coords=dict(nrn_cls=nrn_classes), dims=['nrn_cls'],
                            attrs=dict(unit='msecond'))

    def get_states(self, duration: float, t_rel: float = 0) -> xr.DataArray:
        records = np.array([_get_state_record(state_mon, duration, t_rel)
                            for state_mon in self._state_mons.values()], dtype=object)
        nrn_classes = [nrn_cls.name for nrn_cls in self._state_mons.keys()]
        return xr.DataArray(data=records, coords=dict(nrn_cls=nrn_classes), dims=['nrn_cls'],
                            attrs=dict(unit='msecond'))

    def get_syn_params(
        self, return_delays: bool = True,
        projections: Optional[Iterable[Projection]] = None
    ) -> Optional[SynParams]:
        projections = self.projections if projections is None else projections
        syn_params: List[pd.DataFrame] = []
        proj_keys: List[str] = []

        for proj in projections:
            if proj in self._synapses:
                synapses = self._synapses[proj]
            else:
                continue
            if synapses is not None:
                data = {'w': synapses.w[:], 'pre': synapses.i[:], 'post': synapses.j[:]}
                if return_delays:
                    data['delay'] = synapses.delay[:] / msecond
                syn_params.append(pd.DataFrame(data).set_index(['pre', 'post']).astype(np.float_))
                proj_keys.append(proj.name)
        try:
            return pd.concat(syn_params, keys=proj_keys, names=['proj']).sort_index(level=0)
        except ValueError:
            return None

    def get_delays(self, projection: Projection) -> npt.NDArray[np.float64]:
        self._check_projection_exists(projection)
        return self._synapses[projection].delay / msecond

    def set_delays(self, projection: Projection, delays: npt.ArrayLike) -> None:
        self._check_projection_exists(projection)
        _delays = np.asarray(delays, dtype=float)
        self._synapses[projection].delay = _delays * msecond

    def _build_spike_mons(self, group_factory: GroupFactory, vars_key: str,
                          record: Any = True, active: bool = False) -> None:
        for nrn_cls, group in self._neurons.items():
            name = f'{self.name}_{nrn_cls.name}_spikemon'
            spike_mon = group_factory.create_spikemon(group, vars_key, record, active, name=name)
            self._spike_mons[nrn_cls] = spike_mon

    def _build_state_mons(self, group_factory: GroupFactory, vars_key: str,
                          record: Any = True, active: bool = False) -> None:
        for nrn_cls, group in self._neurons.items():
            name = f'{self.name}_{nrn_cls.name}_statemon'
            state_mon = group_factory.create_statemon(group, vars_key, record, active, name=name)
            self._state_mons[nrn_cls] = state_mon

    def _check_projection_exists(self, projection: Projection) -> None:
        if projection not in self.projections:
            raise ValueError(f"Projection {projection.name} not found in layer {self.name}.")


class StimulusLayer(BaseLayer, IStimulus):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = cls.__name__.lower()
        stimulus_registry[name.removesuffix('layer')] = cls

    def __init__(self, name: str, group_shapes: Dict[NeuronClass, Tuple],
                 group_factory: GroupFactory) -> None:
        super().__init__(name, group_shapes, group_factory.network)

    @abstractmethod
    def set_activations(self, activations: npt.ArrayLike) -> None:
        ...
