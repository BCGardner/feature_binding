from typing import Sequence

from ._base import AbstractSNN
from ..groups import group_registry, GroupFactory
from ..layer import stimulus_registry, SpatialLayer
from ...config import ModelParams
from ...definitions import Projection

__all__ = ["SpatialNet"]

_TOPOLOGY_ID = 'spatial'


class SpatialNet(AbstractSNN):
    def __post_init__(self, model_params: ModelParams) -> None:
        stimulus_id = model_params.network['stimulus']
        neuron_model = model_params.network.get('neuron_model', 'coba')
        factory: GroupFactory = group_registry[neuron_model](
            model_params.namespaces,
            model_params.network.get('connector', 'patch'),
            model_params.network.get('plasticity', 'stdp'),
            self._network,
            self.integ_method
        )
        self._build_layers(stimulus_id, model_params.topology, model_params.projections, factory)

    def _build_layers(self, stimulus_id: str, topology: dict, projections: Sequence[dict],
                      factory: GroupFactory) -> None:
        self._layers = []
        # Input layer
        layer = stimulus_registry[stimulus_id]('L0', topology[stimulus_id], factory)
        self._stimulus = layer
        self._layers.append(layer)
        # Spatial layers
        group_shapes = topology[_TOPOLOGY_ID]
        for idx in range(self.num_hidden + 1):
            proj_params = projections[idx]
            if idx == 0 and Projection.FB in proj_params:  # No feedback to input layer
                del proj_params[Projection.FB]
            self._layers.append(
                SpatialLayer(f'L{idx+1}', group_shapes, proj_params, factory, self._layers[-1])
            )
