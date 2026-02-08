from typing import Dict, Sequence, Tuple

import numpy.typing as npt
from brian2.units import hertz

from ._base import StimulusLayer
from ..groups import GroupFactory
from ...definitions import NeuronClass

__all__ = ["PoissonLayer"]


class PoissonLayer(StimulusLayer):
    def __init__(self, name: str, group_shapes: Dict[NeuronClass, Tuple],
                 group_factory: GroupFactory) -> None:
        super().__init__(name, group_shapes, group_factory)
        self._build_neurons(group_shapes[NeuronClass.EXC], group_factory)
        self._build_spike_mons(group_factory, 'poisson')
        self._build_state_mons(group_factory, 'poisson')

    def set_activations(self, activations: npt.ArrayLike) -> None:
        self._neurons[NeuronClass.EXC].rate = activations * hertz

    def _build_neurons(self, group_size: Sequence, group_factory: GroupFactory) -> None:
        name = f'{self.name}_{NeuronClass.EXC.name}'
        neuron_group = group_factory.create_poisson(group_size, NeuronClass.EXC,
                                                    name=name)
        self._neurons[NeuronClass.EXC] = neuron_group
