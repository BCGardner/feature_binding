import pytest

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from hsnn.core import NeuronClass, SynapseClass, Projection
from hsnn.core._brian2.network import SpatialNet
from hsnn.core._brian2.testing import assert_netstate_equal
from hsnn import pipeline


def test_spnet_restore(config_path: str, tmp_path):
    states_path = tmp_path / 'spnet.pkl'
    model_cfg = OmegaConf.load(config_path)
    assert isinstance(model_cfg, DictConfig)
    model_cfg.network.seed_val = 42
    # Run network and store initial / advanced states
    spnet = SpatialNet(model_cfg)
    spnet.store('init', file_path=states_path)
    state_init = spnet._network.get_states()
    spnet.simulate(10)
    spnet.store('advanced', file_path=states_path)
    state_advanced = spnet._network.get_states()
    with pytest.raises(AssertionError):
        assert_netstate_equal(state_advanced, state_init)
    # Assert successful restore
    spnet_ = SpatialNet(model_cfg)
    spnet_.restore('advanced', file_path=states_path)
    assert_netstate_equal(spnet_._network.get_states(), state_advanced)
    # Assert failed restore
    model_cfg.network.seed_val = 41
    spnet_ = SpatialNet(model_cfg)
    with pytest.raises(ValueError):
        spnet_.restore('advanced', file_path=states_path)


def test_spnet_set_delays(config_path: str, tmp_path):
    syn_params_path = tmp_path / 'syn_params.parquet'
    model_cfg = OmegaConf.load(config_path)
    assert isinstance(model_cfg, DictConfig)
    model_cfg.network.seed_val = 42
    jitter_rng = np.random.default_rng(43)

    #  === Instantiate a new network and jitter synaptic delays === #

    # Initial state
    spnet = SpatialNet(model_cfg)
    projections = [proj for proj in spnet.projections
                   if proj not in {Projection.I2E, Projection.E2I}]
    syn_params_init = spnet.get_syn_params(
        return_delays=True, projections=projections)
    assert syn_params_init is not None

    # Randomly jitter synaptic delays
    pipeline.jitter_synapse_delays(
        spnet.layers[1:], sigma=0.4, projections=projections, seed=jitter_rng)
    syn_params_jittered = spnet.get_syn_params(
        return_delays=True, projections=projections)
    assert syn_params_jittered is not None
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(syn_params_jittered, syn_params_init)

    # Store/restore synaptic parameters for target projections
    syn_params_jittered.to_parquet(syn_params_path)
    syn_params_restored = pd.read_parquet(syn_params_path)

    #  === Instantiate a new network and set to restored delays === #

    # Create new network identical to initial one
    spnet_new = SpatialNet(model_cfg)
    syn_params_new_init = spnet_new.get_syn_params(
        return_delays=True, projections=projections)
    assert syn_params_new_init is not None
    pd.testing.assert_frame_equal(syn_params_new_init, syn_params_init)

    # Restore jittered synaptic delays
    spnet_new.set_delays(syn_params_restored)
    syn_params_new_restored = spnet_new.get_syn_params(
        return_delays=True, projections=projections)
    assert syn_params_new_restored is not None
    pd.testing.assert_frame_equal(syn_params_new_restored, syn_params_jittered)


class TestSpatialNetwork:
    @pytest.fixture(autouse=True)
    def setup_class(self, model_cfg):
        self.spnet = SpatialNet(model_cfg)

    def test_attrs(self, model_cfg: DictConfig):
        assert len(self.spnet.layers) == model_cfg.network.num_hidden + 2
        assert self.spnet.seed_val == model_cfg.network.seed_val
        assert self.spnet.dt == model_cfg.network.dt
        assert self.spnet.integ_method == model_cfg.network.integ_method
        assert set(self.spnet.projections) == set([Projection[name] for name in model_cfg.projections.keys()])

    def test_topology(self, model_cfg: DictConfig):
        print(id(self.spnet))
        input_layer = self.spnet._layers[0]
        assert NeuronClass.EXC in input_layer._neurons
        assert NeuronClass.INH not in input_layer._neurons
        assert input_layer._neurons[NeuronClass.EXC]._N == np.prod(model_cfg.topology.poisson.EXC)
        for layer in self.spnet._layers[1:]:
            for nrn_cls in NeuronClass:
                assert nrn_cls in layer._neurons
                assert layer._neurons[nrn_cls]._N == \
                    np.prod(model_cfg.topology.spatial[nrn_cls.name])

    def test_projections(self):
        assert len([proj for proj in self.spnet._layers[0].projections]) == 0
        for idx, layer in enumerate(self.spnet._layers[1:]):
            if idx == 0:
                projections = set(self.spnet.projections)
                projections.discard(Projection.FB)
                assert set(layer.projections) == projections
            else:
                assert set(layer.projections) == set(self.spnet.projections)
