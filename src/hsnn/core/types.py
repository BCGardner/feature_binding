from typing import Dict, Tuple, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd

__all__ = [
    "SpikeEvents",
    "SpikeTrains",
    "FiringRates",
    "Recording",
    "SpikePatterns",
    "SynParams",
]


SpikeEvents: TypeAlias = Tuple[npt.NDArray[np.int_], npt.NDArray[np.float_]]
"""Spike events: 2-tuple of (neuron IDs, spike times)"""

SpikeTrains: TypeAlias = Dict[np.int_, npt.NDArray[np.float_]]
"""Spike trains: mapping from neuron ID to spike times"""

FiringRates: TypeAlias = Tuple[npt.NDArray[np.int_], npt.NDArray[np.float_]]
"""Firing rates: 2-tuple of (neuron IDs, firing rates)"""

Recording = TypeVar("Recording", SpikeEvents, SpikeTrains)

SpikePatterns: TypeAlias = Dict[np.int_, SpikeTrains]
"""Spatiotemporal spike patterns: mapping from id to spike trains"""

SynParams: TypeAlias = pd.DataFrame
"""Synaptic parameters DataFrame, with column(s): w[, delay],
multilevel-index: [layer,] proj, pre, post"""
