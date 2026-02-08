from typing import Dict, Tuple

from brian2 import NeuronGroup

from ._base import BaseLayer
from ..groups import GroupFactory
from ...definitions import NeuronClass, SynapseClass, Projection

__all__ = ["SpatialLayer"]


_PROJECTION_SYNCLS_MAP = {
    Projection.FF:  SynapseClass.PLASTIC,
    Projection.E2I: SynapseClass.FIXED,
    Projection.I2E: SynapseClass.FIXED,
    Projection.FB:  SynapseClass.PLASTIC,
    Projection.E2E: SynapseClass.PLASTIC
}


class SpatialLayer(BaseLayer):
    def __init__(self, name: str, group_shapes: Dict[NeuronClass, Tuple],
                 proj_params: Dict[Projection, Dict[str, dict]],
                 group_factory: GroupFactory, layer_prev: BaseLayer) -> None:
        super().__init__(name, group_shapes, group_factory.network)
        self.layer_prev = layer_prev
        self._build_neurons(group_shapes, group_factory)
        self._build_synapses(proj_params, group_factory)
        self._build_spike_mons(group_factory, 'neurons')
        self._build_state_mons(group_factory, 'neurons')
        self._build_clampers(group_factory)

    def _build_neurons(self, group_shapes: Dict[NeuronClass, Tuple],
                       group_factory: GroupFactory) -> None:
        for nrn_cls, group_size in group_shapes.items():
            name = f'{self.name}_{nrn_cls.name}'
            neuron_group = group_factory.create_neurons(group_size, nrn_cls, name=name)
            self._neurons[nrn_cls] = neuron_group

    def _build_synapses(self, proj_params: Dict[Projection, Dict[str, dict]],
                        group_factory: GroupFactory) -> None:
        for proj, params in proj_params.items():
            if proj == Projection.FF:
                source = self.layer_prev._neurons[NeuronClass.EXC]
                target = self._neurons[NeuronClass.EXC]
            elif proj == Projection.E2I:
                source = self._neurons[NeuronClass.EXC]
                target = self._neurons[NeuronClass.INH]
            elif proj == Projection.I2E:
                source = self._neurons[NeuronClass.INH]
                target = self._neurons[NeuronClass.EXC]
            elif proj == Projection.FB:
                source = self._neurons[NeuronClass.EXC]
                target = self.layer_prev._neurons[NeuronClass.EXC]
            elif proj == Projection.E2E:
                source = self._neurons[NeuronClass.EXC]
                target = self._neurons[NeuronClass.EXC]
            else:
                raise ValueError(f"Unknown projection: {proj}")
            self._build_projection(source, target, proj, group_factory, **params)

    def _build_projection(self, source: NeuronGroup, target: NeuronGroup,
                          proj: Projection, group_factory: GroupFactory, **kwargs) -> None:
        syn_cls = _PROJECTION_SYNCLS_MAP[proj]
        name = f'{source.name}_{target.name}'
        self._synapses[proj] = \
            group_factory.create_synapses(source, target, syn_cls, name, **kwargs)

    def _build_clampers(self, group_factory: GroupFactory) -> None:
        for nrn_cls, group in self._neurons.items():
            name = f'{self.name}_{nrn_cls.name}_clamper'
            self._clampers[nrn_cls] = group_factory.create_clamper(group, name=name)
