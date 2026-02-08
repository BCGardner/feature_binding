"""Utilities for computing PNG counts from detection databases.

This module provides shared functionality for counting PNGs across trials,
used by both standard experiment analysis and hyperparameter sweep scripts.
"""

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Iterable, Sequence

import numpy as np

import hsnn.analysis.png.db as polydb
from hsnn.analysis.png import PNG
from hsnn.utils import handler

__all__ = [
    "load_detections_trials",
    "get_png_counts_per_layer",
    "get_polygrps_per_layer",
    "get_num_polygrps",
    "combine_counts",
]


def load_detections_trials(
    trials: Sequence[handler.TrialView],
    state: str | tuple[str, ...] = ("pre", "post"),
    **kwargs,
) -> list[dict[str, polydb.PNGDatabase]]:
    """Load PNG databases for multiple trials.

    Args:
        trials: Sequence of trial views to load detections for.
        state: Network state(s) to load ('pre', 'post', or both).
        **kwargs: Additional arguments passed to handler.load_detections.

    Returns:
        List of dictionaries mapping state to PNGDatabase for each trial.
    """
    loader_fn = partial(handler.load_detections, state=state, **kwargs)
    with ThreadPoolExecutor() as executor:
        return list(executor.map(loader_fn, trials))


def get_png_counts_per_layer(
    database: polydb.PNGDatabase,
    layers: Iterable[int],
    nrn_ids: Iterable[int] = range(4096),
    index: int = 1,
) -> np.ndarray:
    """Get PNG counts for each layer from a database.

    Args:
        database: PNG database to query.
        layers: Layer indices to query.
        nrn_ids: Neuron IDs to query.
        index: PNG position index (default 1 = second-firing neuron).

    Returns:
        Array of counts per layer.
    """
    counts = []
    for layer in layers:
        polygrps = database.get_pngs(layer, nrn_ids, index)
        counts.append(len(polygrps))
    return np.array(counts)


def get_polygrps_per_layer(
    database_dict: dict[str, polydb.PNGDatabase],
    layers: Iterable[int] = range(1, 5),
    nrn_ids: Iterable[int] = range(4096),
    index: int = 1,
) -> dict[str, list[list[PNG]]]:
    """Get all detected PNGs for each state per layer.

    Args:
        database_dict: Dictionary mapping state to PNGDatabase.
        layers: Layer indices to query.
        nrn_ids: Neuron IDs to query.
        index: PNG position index (default 1 = second-firing neuron).

    Returns:
        Dictionary mapping state to list of PNG lists (one per layer).
    """
    polygrps: dict[str, list[list[PNG]]] = {state: [] for state in database_dict.keys()}
    for layer in layers:
        for state in polygrps.keys():
            polygrps[state].append(database_dict[state].get_pngs(layer, nrn_ids, index))  # type: ignore
    return polygrps


def get_num_polygrps(
    polygrps_dict: dict[str, list[list[PNG]]],
) -> dict[str, np.ndarray]:
    """Count PNGs in each layer, indexed by state.

    Args:
        polygrps_dict: Dictionary mapping state to list of PNG lists per layer.

    Returns:
        Dictionary mapping state to array of counts per layer.
    """
    num_polygrps = {}
    for state, polygrps_l in polygrps_dict.items():
        num_polygrps[state] = np.array([len(polygrps) for polygrps in polygrps_l])
    return num_polygrps


def combine_counts(
    trial_counts: list[dict[str, np.ndarray]],
    states: tuple[str, ...] = ("post",),
) -> dict[str, np.ndarray]:
    """Stack counts from multiple trials into arrays.

    Args:
        trial_counts: List of per-trial count dicts.
        states: States to combine.

    Returns:
        Dict mapping state -> stacked array of shape (n_trials, n_layers).
    """
    dst = {}
    for state in states:
        arrays = [counts[state] for counts in trial_counts if state in counts]
        if arrays:
            dst[state] = np.vstack(arrays)
    return dst
