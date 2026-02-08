from typing import Dict
import pytest

import numpy as np
from brian2 import Network, seed
from brian2.units import msecond, mvolt, nsiemens, hertz

from hsnn.core import NeuronClass, SynapseClass, Projection
from hsnn.core.config import ModelParams
from hsnn.core._brian2.groups import COBAFactory


@pytest.fixture(autouse=True)
def set_seed():
    seed(42)


class TestCobaFactory:
    @pytest.fixture(autouse=True)
    def setup_method(self, model_params: ModelParams):
        self.factory = COBAFactory(model_params.namespaces, 'patch', 'stdp', Network())

    def test_create_neurons(self, model_params: ModelParams):
        namespace_cfg = model_params.namespaces['neurons']
        for nrn_cls in NeuronClass:
            num_nrns = (2, 2)
            spacing = self.factory._spatial_span / 2
            name = f'{nrn_cls.name}'
            nrn_grp = self.factory.create_neurons(num_nrns, nrn_cls, name=name)
            assert np.allclose(nrn_grp.x, np.array([0, 1, 0, 1]) * spacing)
            assert np.allclose(nrn_grp.y, np.array([0, 0, 1, 1]) * spacing)
            assert nrn_grp._N == np.prod(num_nrns)
            assert nrn_grp.identifier == nrn_cls
            assert nrn_grp.name == name
            assert np.all(nrn_grp.v[:] / mvolt == namespace_cfg[nrn_cls]['V_0'])
            assert np.all(nrn_grp.g_e[:] / nsiemens == 0)
            assert np.all(nrn_grp.g_i[:] / nsiemens == 0)
        self.factory.network.run(10*msecond, namespace={})

    def test_create_synapses(self, model_params: ModelParams):
        namespace_cfg = model_params.namespaces['synapses']
        connect_kwargs = model_params.projections[0][Projection.FF]['connect_kwargs']
        neuron_group_1 = self.factory.create_neurons((32, 32), NeuronClass.EXC, name='nrn_grp_1')
        neuron_group_2 = self.factory.create_neurons((64, 64), NeuronClass.EXC, name='nrn_grp_2')
        for syn_cls in SynapseClass:
            attrs_init: Dict[str, str] = {}
            if syn_cls == SynapseClass.FIXED:
                attrs_init.update(**{
                    'w':        'w_max',
                    'delay':    'delay_max'
                })
            name = syn_cls.name
            synapses = self.factory.create_synapses(
                neuron_group_1, neuron_group_2, syn_cls, name=name,
                connect_kwargs=connect_kwargs, attrs_init=attrs_init
            )
            assert synapses.identifier == syn_cls
            assert synapses.name == name
            if syn_cls == SynapseClass.FIXED:
                assert np.all(synapses.w[:] == namespace_cfg[syn_cls]['w_max'])
                assert np.all(synapses.delay[:] / msecond == namespace_cfg[syn_cls]['delay_max'])
            else:
                assert np.all((synapses.w[:] >= 0) & (synapses.w[:] <= 1))
                assert np.all(synapses.delay[:] / msecond <= namespace_cfg[syn_cls]['delay_max'])
        self.factory.network.run(10*msecond, namespace={'rho': 0})

    def test_create_poisson(self):
        num_nrns = (2, 2, 2)
        spacing_x = self.factory._spatial_span / num_nrns[2]
        spacing_y = self.factory._spatial_span / num_nrns[1]
        nrn_cls = NeuronClass.EXC
        name = f'{nrn_cls.name}'
        nrn_grp = self.factory.create_poisson(num_nrns, nrn_cls, name=name)
        assert np.allclose(nrn_grp.x, np.array([0, 1, 0, 1, 0, 1, 0, 1]) * spacing_x)
        assert np.allclose(nrn_grp.y, np.array([0, 0, 1, 1, 0, 0, 1, 1]) * spacing_y)
        assert np.array_equal(nrn_grp.z, np.array([0, 0, 0, 0, 1, 1, 1, 1]))
        assert nrn_grp._N == np.prod(num_nrns)
        assert nrn_grp.identifier == nrn_cls
        assert nrn_grp.name == name
        assert np.all(nrn_grp.rate[:] / hertz == 0)
        self.factory.network.run(10*msecond, namespace={})
