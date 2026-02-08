"""Provides functions for applying jitter to synaptic delays in HSNN layers.

This module contains utilities for perturbing synaptic delays, which can be
useful for introducing noise or variability during simulations or training
of spiking neural networks. This includes setting synaptic delays from a
pandas dataframe.
"""

from argparse import Namespace
from typing import Iterable

import numpy as np
import pandas as pd

from hsnn.core import ILayer, Projection
from hsnn.core.logger import get_logger
from hsnn import ops
from hsnn.simulation import Simulator
from hsnn.utils import handler
from hsnn.utils.handler import TrialView

__all__ = [
    "jitter_synapse_delays",
    "restore_jittered_network"
]

_DEFAULT_PROJS = (Projection.FF, Projection.E2E, Projection.FB)

logger = get_logger(__name__)


def jitter_synapse_delays(
    layers: Iterable[ILayer],
    sigma: float,
    projections: Iterable[Projection] = _DEFAULT_PROJS,
    dt: float = 0.1,
    seed: np.random.Generator | None = None
) -> None:
    rng = np.random.default_rng(seed)
    for layer in layers:
        _projections = [projection for projection in projections
                        if projection in layer.projections]
        for projection in _projections:
            new_delays = ops.multiplicative_jitter(
                layer.get_delays(projection), sigma, dt, rng
            )
            layer.set_delays(projection, new_delays)
            logger.info(
                f"Jittered synaptic delays for {layer.name} ({projection})"
            )


def restore_jittered_network(opt: Namespace, trial: TrialView, sim: Simulator) -> None:
        syn_params_path = handler.get_artifact_path(
            'syn_params', trial, opt.chkpt, subdir=opt.subdir,
            delay_jitter=opt.delay_jitter, noise=opt.noise, ext='.parquet'
        )
        if not syn_params_path.exists():
            raise FileNotFoundError(
                f"Synaptic parameters file '{syn_params_path}' not found. "
                "Run the simulation with delay jitter first."
            )
        syn_params_jittered = pd.read_parquet(syn_params_path)
        sim.network.set_delays(syn_params_jittered)
        logger.info(f"Restored delays from '{syn_params_path}'")
