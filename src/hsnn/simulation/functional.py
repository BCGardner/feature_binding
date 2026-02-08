from typing import List, Sequence

import numpy as np
import xarray as xr

from .managers import MonitorContext, StateContext, ClampContext
from ..core.interfaces import INetwork
from .. import ops


def flush(network: INetwork, duration: float):
    network.clear_input()
    if duration > 0:
        with ClampContext(network):
            network.simulate(duration)


def present(network: INetwork, image: np.ndarray, duration: float,
            duration_relax: float = 0) -> None:
    flush(network, duration_relax)
    network.encode(image)
    network.simulate(duration)


def batch(network: INetwork, images: Sequence[np.ndarray], duration: float,
          duration_relax: float = 0) -> None:
    for image in images:
        present(network, image, duration, duration_relax)


def infer_batch(network: INetwork, images: Sequence[np.ndarray], duration: float,
                duration_relax: float = 0, slicer=slice(None)) -> xr.DataArray:
    lrate_prior = network.lrate
    network.lrate = 0
    with StateContext(network):
        flush(network, duration_relax)
        with MonitorContext(network, monitor_spikes=True):
            records: List[xr.DataArray] = []
            for image in images:
                with StateContext(network):
                    t_rel = network.t
                    present(network, image, duration)
                    records.append(
                        network.get_spikes(duration, t_rel, slicer)
                    )
    network.lrate = lrate_prior
    attrs = {
        'description': 'Spike recordings (inference)'
    }
    return ops.concat_records(records, 'img', attrs=attrs)


def infer_batch_reps(network: INetwork, images: Sequence[np.ndarray], duration: float,
                     duration_relax: float = 0, slicer=slice(None), reps: int = 1) -> xr.DataArray:
    records: List[xr.DataArray] = [infer_batch(network, images, duration, duration_relax, slicer)
                                   for _ in range(reps)]
    return ops.concat_records(records, 'rep')
