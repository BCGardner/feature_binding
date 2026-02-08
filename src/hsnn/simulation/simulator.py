from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from . import functional as fn
from ..core.backends import create_network
from ..core.interfaces import INetwork
from ..utils import io

__all__ = ["get_sim_kwargs", "Simulator"]


_DEFAULT_KWARGS = {
    'duration': 100,
    'duration_relax': 50
}


def get_sim_kwargs(sim_cfg: Mapping) -> dict:
    sim_kwargs = copy(_DEFAULT_KWARGS)
    for k, v in sim_cfg.items():
        if k in sim_kwargs:
            sim_kwargs[k] = v
    return sim_kwargs


class Simulator:
    def __init__(self, network: INetwork, duration: float, duration_relax: float) -> None:
        self._network = network
        self.duration = duration
        self.duration_relax = duration_relax

    @property
    def network(self) -> INetwork:
        return self._network

    @classmethod
    def from_config(cls, model_cfg: str | Path | Mapping, backend: str = 'brian2') -> Simulator:
        model_cfg_ = io.as_dict(model_cfg)
        sim_kwargs = get_sim_kwargs(model_cfg_.get('simulation', {}))
        network = create_network(model_cfg, backend)
        return cls(network, **sim_kwargs)

    @classmethod
    def from_config_restore(cls, model_cfg: str | Path | Mapping, file_path: str | Path,
                            name: str = 'default', backend: str = 'brian2') -> Simulator:
        model_cfg_ = io.as_dict(model_cfg)
        if model_cfg_['network']['seed_val'] is None:
            raise ValueError(f"invalid seed_val: {None}")
        file_path = Path(file_path)
        if not file_path.is_file():
            raise FileNotFoundError()
        simulator = cls.from_config(model_cfg_, backend)
        simulator.restore(file_path, name)
        return simulator

    def restore(self, file_path: str | Path, name: str = 'default') -> None:
        self._network.restore(name, file_path)

    def restore_delays(self, file_path: Path) -> None:
        syn_params = pd.read_parquet(file_path)
        self._network.set_delays(syn_params)

    def flush(self) -> None:
        return fn.flush(self._network, self.duration_relax)

    def present(self, images: Sequence[np.ndarray]) -> None:
        return fn.batch(self._network, images, self.duration, self.duration_relax)

    def infer(self, images: Sequence[np.ndarray], slicer=slice(None)) -> xr.DataArray:
        return fn.infer_batch(self._network, images, self.duration, self.duration_relax, slicer)

    def infer_reps(self, images: Sequence[np.ndarray], slicer=slice(None), reps: int = 1) -> xr.DataArray:
        return fn.infer_batch_reps(self._network, images, self.duration, self.duration_relax,
                                   slicer, reps)
