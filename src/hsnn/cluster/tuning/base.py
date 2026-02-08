from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, MutableMapping, Mapping, Optional, Sequence

import numpy as np
from omegaconf import OmegaConf
from ray import tune

from hsnn.utils import io
from hsnn.simulation import Simulator

__all__ = ["traverse_dict", "override_config", "BaseTrainable"]


_TUNE_SAMPLER_MAPPING = {
    "uniform": tune.uniform,
    "quniform": tune.quniform,
    "loguniform": tune.loguniform,
    "qloguniform": tune.qloguniform,
    "randn": tune.randn,
    "qrandn": tune.qrandn,
    "randint": tune.randint,
    "qrandint": tune.qrandint,
    "lograndint": tune.lograndint,
    "qlograndint": tune.qlograndint,
    "choice": tune.choice,
    "grid": tune.grid_search,
}


def traverse_dict(cfg: Mapping, keys: Iterable) -> Any:
    """Returns a reference to either a nested dictionary or end value.
    """
    ptr = cfg
    for k in keys:
        ptr = ptr[k]
    return ptr


def get_nested(key_path: str, cfg: Mapping):
    return traverse_dict(cfg, key_path.split('.'))


def set_nested(value, key_path: str, cfg: MutableMapping) -> None:
    subkeys = key_path.split('.')
    ptr = traverse_dict(cfg, subkeys[:-1])
    ptr[subkeys[-1]] = value


def override_config(cfg: MutableMapping, hyper_params: Mapping) -> Dict[str, Any]:
    """Returns a copy of the config updated with ray tune sample spaces specified by hyper_params.

    Args:
        cfg (MutableMapping): Experiment config.
        hyper_params (Mapping): Hyperparameters: mapping key paths to ray tune sampler names.

    Returns:
        Dict[str, Any]: Copy of modified experiment config.
    """
    cfg_ = deepcopy(io.as_dict(cfg))
    for path, kwargs in hyper_params.items():
        sampler = _TUNE_SAMPLER_MAPPING[kwargs['search']](*kwargs['args'])
        set_nested(sampler, path, cfg)
        keys = path.split('.')
        ptr = traverse_dict(cfg_, keys[:-1])
        ptr[keys[-1]] = _TUNE_SAMPLER_MAPPING[kwargs['search']](*kwargs['args'])
    return cfg_


def _as_primitives(cfg: MutableMapping, hyper_params: Mapping) -> None:
    """Casts non-primitive config values to fundamental Python types (e.g. int or float).
    """
    for path in hyper_params.keys():
        val = get_nested(path, cfg)
        if isinstance(val, np.number):
            set_nested(val.item(), path, cfg)


class BaseTrainable(tune.Trainable):
    def setup(self, config: Dict, data: Sequence[np.ndarray], labels: np.ndarray):  # type: ignore[override]
        self.data = data
        self.labels = labels
        self.sim = Simulator.from_config(config)
        training_cfg: dict = config['training']
        self.sim.network.lrate = training_cfg['lrate']
        self.num_epochs_iter: int = training_cfg.get('num_epochs_iter', 1)
        self.shuffle: bool = training_cfg.get('shuffle', False)
        self._save_config(config)

    def save_checkpoint(self, checkpoint_dir: str) -> Optional[str | dict]:
        store_path = Path(checkpoint_dir) / 'netstate.pkl'
        self.sim.network.store(file_path=store_path)
        return {'store': str(store_path)}

    def load_checkpoint(self, checkpoint: dict | str):
        if isinstance(checkpoint, dict):
            store_path = Path(checkpoint['store']) / 'netstate.pkl'
        elif isinstance(checkpoint, str):
            store_path = Path(checkpoint) / 'netstate.pkl'
        else:
            raise ValueError(f'invalid checkpoint: {checkpoint}')
        cfg_path = Path(self.logdir) / 'config.yaml'
        self.sim = Simulator.from_config_restore(cfg_path, store_path)
        self.sim.network.seed_val = None

    def _save_config(self, config: Dict) -> None:
        config_ = deepcopy(config)
        config_['network']['seed_val'] = self.sim.network.seed_val
        _as_primitives(config_, config_['tuning'].pop('hyper_params', {}))
        OmegaConf.save(config_, Path(self.logdir) / 'config.yaml')
