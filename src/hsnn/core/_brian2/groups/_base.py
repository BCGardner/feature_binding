from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional, Sequence, Type

import numpy as np
from brian2 import CodeRunner, Network, Group, NeuronGroup, SpikeMonitor, \
    StateMonitor, Synapses, Equations
from brian2.units import hertz

from . import helper as hp
from .connectors import connector_registry
from .plasticity import plasticity_registry
from ...definitions import NeuronClass, SynapseClass

__all__ = ["GroupFactory", "group_registry"]

group_registry: Dict[str, Type[GroupFactory]] = {}


class GroupFactory(ABC):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = cls.__name__.lower()
        group_registry[name.removesuffix('factory')] = cls

    def __init__(self, namespaces: Dict[str, Dict[Enum, dict]], connector_id: str, plasticity_id: str,
                 network: Network, integ_method: str = 'rk4', spatial_span: float = 128) -> None:
        self._namespaces = hp.process_namespaces(namespaces)
        self._connector = connector_registry[connector_id]()
        self._plasticity = plasticity_registry[plasticity_id]()
        self._network = network
        self._integ_method = integ_method
        self._spatial_span = spatial_span  # Aligns spatial layers with differing numbers of neurons
        self.event_records = {
            'poisson': None,
            'neurons': None,
        }
        self.state_records = {
            'poisson': 'rate',
            'neurons': True,
            'synapses': True,
        }
        self.__post_init__()

    def __post_init__(self) -> None:
        return None

    @property
    def network(self) -> Network:
        return self._network

    @property
    def spatial_eqs(self) -> Equations:
        """All created NeuronGroups must be spatial.
        """
        return Equations('''
        x : 1
        y : 1
        ''')

    @property
    def synapses_model(self) -> Equations:
        """The base model which must be implemented by all created Synapses.
        """
        return Equations("w : 1")

    @abstractmethod
    def create_neurons(self, num_nrns: Sequence, identifier: NeuronClass,
                       name: str = 'neurongroup*', **kwargs) -> NeuronGroup:
        ...

    @abstractmethod
    def create_synapses(self, source: NeuronGroup, target: NeuronGroup,
                        identifier: SynapseClass, name: str = 'synapses*', **kwargs) -> Synapses:
        ...

    @abstractmethod
    def create_clamper(self, group: Group, active: bool = False, name: Optional[Any] = None) -> CodeRunner:
        ...

    def create_poisson(self, num_nrns: Sequence, identifier: NeuronClass,
                       name: str = 'poissongroup*', **kwargs) -> NeuronGroup:
        assert len(num_nrns) == 3, "shape must match: (CxHxW)"
        attrs_init = kwargs.get('attrs_init', {})
        equations = Equations("rate : Hz") + self.spatial_eqs + Equations("z : 1")
        group = NeuronGroup(
            np.prod(num_nrns), equations, self._integ_method, threshold='rand() >= exp(-rate*dt)',
            namespace={}, name=name, **kwargs
        )
        attrs_init = hp.process_kwargs({'rate': 0 * hertz}, **attrs_init)
        xs, ys, zs = hp.get_spatial_coords(num_nrns[1:], num_nrns[0], self._spatial_span)
        hp.set_group_attr(group, identifier=identifier, x=xs, y=ys, z=zs, **attrs_init)
        self._update_network(group)
        return group

    def create_spikemon(self, source: Group, key: str, record: Any = True,
                        active: bool = False, name: str = 'spikemonitor*') -> SpikeMonitor:
        spike_mon = SpikeMonitor(source, self.event_records[key], record, name=name)
        spike_mon.active = active
        self._update_network(spike_mon)
        return spike_mon

    def create_statemon(self, source: Group, key: str, record: Any = True,
                        active: bool = False, name: str = 'statemonitor*') -> StateMonitor:
        state_mon = StateMonitor(source, self.state_records[key], record, name=name)
        state_mon.active = active
        self._update_network(state_mon)
        return state_mon

    def _create_coderunner(self, group: Group, code: Any, active: bool = False,
                           name: Optional[Any] = None) -> CodeRunner:
        code_runner = group.run_regularly(code, name=name)
        code_runner.active = active
        return code_runner

    def _update_network(self, group: Group):
        if group.name in self._network:
            raise ValueError(f"'{group.name}' already exists")
        self._network.add(group)
