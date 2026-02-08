import pytest

import numpy as np
from brian2 import Network, seed

from hsnn.core import NeuronClass
from hsnn.core.config import ModelParams
from hsnn.core._brian2.groups import COBAFactory
from hsnn.core._brian2.layer import PoissonLayer, SpatialLayer


@pytest.fixture(autouse=True)
def set_seed():
    seed(42)


@pytest.fixture
def coba_factory(model_params: ModelParams) -> COBAFactory:
    return COBAFactory(model_params.namespaces, 'patch', 'stdp', Network())


@pytest.fixture
def poisson_layer(model_params: ModelParams, coba_factory: COBAFactory) -> PoissonLayer:
    return PoissonLayer('L0', model_params.topology['poisson'], coba_factory)


def test_poisson_layer(poisson_layer: PoissonLayer):
    neurons = poisson_layer._neurons[NeuronClass.EXC]
    assert neurons.N == np.prod(poisson_layer.group_shapes[NeuronClass.EXC])


def test_spatial_layer(model_params: ModelParams, coba_factory: COBAFactory,
                       poisson_layer: PoissonLayer):
    group_sizes = model_params.topology['spatial']
    proj_params = model_params.projections[0]
    layer = SpatialLayer('L1', group_sizes, proj_params, coba_factory, poisson_layer)
    for nrn_cls, group_size in group_sizes.items():
        neurons = layer._neurons[nrn_cls]
        assert neurons.N == np.prod(group_size)
    elems = [obj.name for obj in layer._network.objects]
    assert len(elems) == len(set(elems))
