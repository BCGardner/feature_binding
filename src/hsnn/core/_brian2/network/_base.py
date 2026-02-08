import logging
from pathlib import Path
import pickle
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from brian2 import defaultclock, seed, start_scope, Network, Synapses
from brian2.units import msecond

from ..layer import BaseLayer
from ..._base import BaseNetwork
from ...logger import get_logger

logger = get_logger(__name__)


class AbstractSNN(BaseNetwork):
    def __init__(self, model_cfg: str | Path | Mapping) -> None:
        start_scope()
        self._network = Network()
        self._namespace: Dict[str, Any] = {'rho': 0}
        self._layers: List[BaseLayer] = []
        super().__init__(model_cfg)

    @property
    def seed_val(self):
        return self._seed_val

    @seed_val.setter
    def seed_val(self, value: Optional[Any]):
        if value is None:
            rng = np.random.default_rng(None)
            value = int(rng.integers((1<<32 - 1), dtype=np.int32))
        seed(value)
        self._seed_val = value
        logger.info(f'Network seed: {value}')

    @property
    def dt(self) -> float:
        return self._dt

    @dt.setter
    def dt(self, value: float):
        defaultclock.dt = value * msecond
        self._dt = value

    @property
    def t(self) -> float:
        return self._network.t / msecond

    @property
    def layers(self) -> Sequence[BaseLayer]:
        return self._layers

    @property
    def lrate(self) -> float:
        return self._namespace['rho']

    @lrate.setter
    def lrate(self, value: float):
        self._namespace['rho'] = value

    @property
    def store_names(self) -> Tuple:
        return tuple(self._network._stored_state.keys())

    def simulate(self, duration: float) -> None:
        if logger.getEffectiveLevel() == logging.INFO:
            report = 'stdout'
            logger.info('Network simulation:')
        else:
            report = None
        self._network.run(duration * msecond, report=report, namespace=self._namespace)

    def store(self, name: str = 'default', file_path: Optional[str | Path] = None) -> None:
        if file_path is not None:
            file_path = str(Path(file_path).resolve())
        self._network.store(name, filename=file_path)

    def restore(self, name: str = 'default', file_path: Optional[str | Path] = None) -> None:
        if file_path is not None:
            if not _is_compatible(self._network, name, file_path):
                raise ValueError("incompatible restored state")
        self._network.restore(name, filename=file_path)

    def remove_store(self, name: str) -> None:
        del self._network._stored_state[name]


def _is_compatible(network: Network, name: str, file_path: str | Path) -> bool:
    with open(file_path, "rb") as f:
        state: Dict[str, Any] = pickle.load(f)[name]
    syn_names = {obj.name for obj in network.objects if isinstance(obj, Synapses)}
    syn_states = {k: v for k, v in state.items() if k in syn_names}
    for syn_name in syn_names:
        syn_state = syn_states[syn_name]
        if not all([
            np.array_equal(syn_state['_synaptic_pre'][0], network[syn_name].i),
            np.array_equal(syn_state['_synaptic_post'][0], network[syn_name].j)
        ]):
            return False
    return True
