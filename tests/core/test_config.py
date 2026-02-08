from typing import Sequence
import pytest

from hsnn.core.config import ModelParams
from hsnn.core.definitions import NeuronClass, SynapseClass, Projection

_NAMESPACES_CLASS_MAPPING = {
    'neurons': NeuronClass,
    'synapses': SynapseClass
}


class TestModelParams:
    @pytest.fixture(autouse=True)
    def setup_method(self, config_path: str):
        self.model_params = ModelParams(config_path)

    def test_attrs(self):
        for attr in ('network', 'encoder', 'topology', 'projections', 'namespaces'):
            assert hasattr(self.model_params, attr)

    def test_network(self):
        assert isinstance(self.model_params.network, dict)

    def test_encoder(self):
        assert isinstance(self.model_params.encoder, dict)

    def test_topology(self):
        topology_cfg = self.model_params.topology
        for _, kwargs in topology_cfg.items():
            for nrn_cls, spatial_dims in kwargs.items():
                assert nrn_cls in NeuronClass
                assert isinstance(spatial_dims, tuple)

    def test_projections(self):
        num_hidden = self.model_params.network['num_hidden']
        projections_cfg = self.model_params.projections
        assert isinstance(projections_cfg, Sequence)
        assert len(projections_cfg) == num_hidden + 1
        for idx in range(len(projections_cfg)):
            projections = projections_cfg[idx]
            for proj, proj_kwargs in projections.items():
                assert proj in Projection
                assert 'connect_kwargs' in proj_kwargs.keys()

    def test_namespaces(self):
        namespaces_cfg = self.model_params.namespaces
        for key, symbol_tables in namespaces_cfg.items():
            grp_cls = _NAMESPACES_CLASS_MAPPING[key]
            for grp_type in symbol_tables.keys():
                assert grp_type in grp_cls
