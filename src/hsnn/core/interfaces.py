from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Protocol, Tuple

import numpy as np
import numpy.typing as npt
import xarray as xr

from .definitions import NeuronClass, Projection
from .types import SynParams

__all__ = ["IEncoder", "IStimulus", "ILayer", "INetwork"]


class IEncoder(Protocol):
    def transform(self, data: np.ndarray) -> npt.NDArray[np.float_]:
        ...


class IStimulus(Protocol):
    def set_activations(self, activations: npt.ArrayLike) -> None:
        ...


class ILayer(Protocol):
    name: str
    group_shapes: Dict[NeuronClass, Tuple]

    monitor_spikes: bool
    monitor_states: bool
    clamp_voltages: bool

    @property
    def projections(self) -> Iterable[Projection]:
        ...

    def get_spikes(self, duration: float, t_rel: float = 0) -> xr.DataArray:
        ...

    def get_states(self, duration: float, t_rel: float = 0) -> xr.DataArray:
        ...

    def get_syn_params(
        self, return_delays: bool = True,
        projections: Optional[Iterable[Projection]] = None
    ) -> Optional[SynParams]:
        ...

    def get_delays(self, projection: Projection) -> npt.NDArray[np.float64]:
        ...

    def set_delays(self, projection: Projection, delays: npt.ArrayLike) -> None:
        ...


class INetwork(Protocol):
    num_hidden: int
    seed_val: Any
    dt: float
    integ_method: str
    lrate: float

    monitor_spikes: bool
    monitor_states: bool
    clamp_voltages: bool

    @property
    def t(self) -> float:
        ...

    @property
    def layers(self) -> Sequence[ILayer]:
        ...

    @property
    def projections(self) -> Iterable[Projection]:
        ...

    @property
    def store_names(self) -> Tuple:
        ...

    def get_spikes(self, duration: float, t_rel: float = 0, slicer=slice(None)) -> xr.DataArray:
        ...

    def get_states(self, duration: float, t_rel: float = 0, slicer=slice(None)) -> xr.DataArray:
        ...

    def get_syn_params(
        self, return_delays: bool = True,
        projections: Optional[Iterable[Projection]] = None,
        slicer=slice(None)
    ) -> Optional[SynParams]:
        ...

    def set_delays(self, syn_params: SynParams) -> None:
        ...

    def encode(self, data: np.ndarray) -> None:
        ...

    def simulate(self, duration: float) -> None:
        ...

    def clear_input(self) -> None:
        ...

    def store(self, name: str = 'default', file_path: Optional[str | Path] = None) -> None:
        ...

    def restore(self, name: str = 'default', file_path: Optional[str | Path] = None) -> None:
        ...

    def remove_store(self, name: str) -> None:
        ...
