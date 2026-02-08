from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from .config import ModelParams
from .encoders import encoder_registry
from .interfaces import IEncoder, IStimulus, ILayer, INetwork
from .definitions import Projection
from .types import SynParams


class BaseNetwork(ABC, INetwork):
    def __init__(self, model_cfg: str | Path | Mapping) -> None:
        model_params = ModelParams(model_cfg)
        # Network attrs
        self.num_hidden = model_params.network['num_hidden']
        self.seed_val = model_params.network.get('seed_val', None)
        self.dt = model_params.network.get('dt', 0.1)
        self.integ_method = model_params.network.get('integ_method', 'rk4')
        self._projections: Iterable[Projection] = tuple(model_params.projections[0].keys())
        encoder_id = model_params.network['encoder']
        self._encoder: IEncoder = encoder_registry[encoder_id](**model_params.encoder)
        self._stimulus: IStimulus
        self.__post_init__(model_params)
        if not hasattr(self, "_stimulus"):
            raise AttributeError("stimulus is unassigned")

    @abstractmethod
    def __post_init__(self, model_params: ModelParams) -> None:
        ...

    @property
    @abstractmethod
    def t(self) -> float:
        ...

    @property
    @abstractmethod
    def layers(self) -> Sequence[ILayer]:
        ...

    @property
    @abstractmethod
    def store_names(self) -> Tuple:
        ...

    @property
    @abstractmethod
    def lrate(self) -> float:
        ...

    @lrate.setter
    @abstractmethod
    def lrate(self, value: float):  # type: ignore
        ...

    @abstractmethod
    def simulate(self, duration: float) -> None:
        ...

    @abstractmethod
    def store(self, name: str = 'default', file_path: Optional[str | Path] = None) -> None:
        ...

    @abstractmethod
    def restore(self, name: str = 'default', file_path: Optional[str | Path] = None) -> None:
        ...

    @abstractmethod
    def remove_store(self, name: str) -> None:
        ...

    @property
    def projections(self) -> Iterable[Projection]:
        return self._projections

    @property
    def monitor_spikes(self) -> bool:
        return all([layer.monitor_spikes for layer in self.layers])

    @monitor_spikes.setter
    def monitor_spikes(self, value: bool):  # type: ignore
        for layer in self.layers:
            layer.monitor_spikes = value

    @property
    def monitor_states(self) -> bool:
        return all([layer.monitor_states for layer in self.layers])

    @monitor_states.setter
    def monitor_states(self, value: bool):  # type: ignore
        for layer in self.layers:
            layer.monitor_states = value

    @property
    def clamp_voltages(self) -> bool:
        return all([layer.clamp_voltages for layer in self.layers])

    @clamp_voltages.setter
    def clamp_voltages(self, value: bool):  # type: ignore
        for layer in self.layers:
            layer.clamp_voltages = value

    def get_spikes(self, duration: float, t_rel: float = 0, slicer=slice(None)) -> xr.DataArray:
        records = [layer.get_spikes(duration, t_rel) for layer in self.layers[slicer]]
        coords = {'layer': self._get_layer_ids(slicer)}
        return xr.concat(records, dim='layer', fill_value=None).assign_coords(coords)

    def get_states(self, duration: float, t_rel: float = 0, slicer=slice(None)) -> xr.DataArray:
        records = [layer.get_states(duration, t_rel) for layer in self.layers[slicer]]
        coords = {'layer': self._get_layer_ids(slicer)}
        return xr.concat(records, dim='layer', fill_value=None).assign_coords(coords)

    def get_syn_params(
        self, return_delays: bool = True,
        projections: Optional[Iterable[Projection]] = None,
        slicer=slice(None)
    ) -> Optional[SynParams]:
        layer_ids = self._get_layer_ids(slicer)
        syn_states = [self.layers[idx].get_syn_params(return_delays, projections)
                      for idx in layer_ids]
        try:
            return pd.concat(syn_states, keys=layer_ids, names=['layer']).sort_index(level=0)  # type: ignore[arg-type]
        except ValueError:
            return None

    def set_delays(self, syn_params: SynParams) -> None:
        _check_syn_params(syn_params, check_delay=True)
        for layer_id, layer_params in syn_params.groupby('layer', observed=False):
            for proj, proj_params in layer_params.groupby('proj', observed=False):
                layer = self.layers[int(layer_id)]  # type: ignore
                layer.set_delays(Projection[str(proj)], proj_params['delay'].to_numpy())

    def encode(self, data: np.ndarray) -> None:
        self._stimulus.set_activations(self._encoder.transform(data))

    def clear_input(self) -> None:
        self._stimulus.set_activations(0)

    def _get_layer_ids(self, slicer: slice) -> List[int]:
        return list(range(*slicer.indices(len(self.layers))))


def _check_syn_params(syn_params: SynParams, check_delay: bool = True) -> None:
    assert isinstance(syn_params.index, pd.MultiIndex)
    midx_expected = ['layer', 'proj', 'pre', 'post']
    if syn_params.index.names != midx_expected:
        raise ValueError(f"Expected index names: {midx_expected}, "
                         "but got: {syn_params.index.names}")
    if check_delay:
        if 'delay' not in syn_params.columns:
            raise ValueError("Delay column not found in syn_params")
