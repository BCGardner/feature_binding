"""Common utilities for computing information measures across trials.

This module provides shared functionality for computing stimulus-specific
information measures from inference recordings, used by both standard
experiment analysis and hyperparameter sweep analysis scripts.
"""

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Optional, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from hsnn.analysis import measures
from hsnn.utils import handler

__all__ = [
    "load_results_trials",
    "compute_measures_single",
    "compute_trial_measures",
    "rank_measures",
    "combine_measures",
    "get_nested_config",
]


def load_results_trials(
    trials: Sequence[handler.TrialView],
    state: str | tuple[str, ...] = ("pre", "post"),
    **kwargs,
) -> list[dict[str, xr.DataArray]]:
    """Load inference results for multiple trials in parallel.

    Args:
        trials: Sequence of trial views to load.
        state: Training state(s) to load ('pre', 'post', or both).
        **kwargs: Additional arguments passed to handler.load_results.

    Returns:
        List of dicts mapping state -> recordings DataArray.
    """
    loader_fn = partial(handler.load_results, state=state, **kwargs)
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(loader_fn, trials))
    return results


def compute_measures_single(args: tuple) -> tuple:
    """Compute measures for a single (trial, state) pair.

    Args:
        args: Tuple of (trial_idx, state, records, labels, target, duration, offset, layer).

    Returns:
        Tuple of (trial_idx, state, result DataFrame).
    """
    trial_idx, state, records, labels, target, duration, offset, layer = args
    result = measures.get_combined_measures(
        records,
        labels=labels,
        target=target,
        duration=duration,
        offset=offset,
        layer=layer,
    )
    return trial_idx, state, result


def compute_trial_measures(
    results: list[dict[str, xr.DataArray]],
    labels: pd.DataFrame,
    target: int,
    duration: float,
    offset: float,
    layer: int = 4,
    states: tuple[str, ...] = ("pre", "post"),
) -> list[dict[str, pd.DataFrame]]:
    """Compute information measures for all trials and states.

    Args:
        results: List of loaded inference results per trial.
        labels: Dataset annotations.
        target: Target feature value (e.g., 1 for convex).
        duration: Observation period.
        offset: Start of observation period.
        layer: Network layer to analyse.
        states: Training states to compute.

    Returns:
        List of dicts mapping state -> measures DataFrame per trial.
    """
    work_items = [
        (i, state, records, labels, target, duration, offset, layer)
        for i, records_dict in enumerate(results)
        for state, records in records_dict.items()
        if state in states
    ]

    with ProcessPoolExecutor() as executor:
        computed = list(executor.map(compute_measures_single, work_items))

    # Reconstruct structure
    specific_measures: list[dict[str, pd.DataFrame]] = [{} for _ in results]
    for trial_idx, state, result in computed:
        specific_measures[trial_idx][state] = result

    return specific_measures


def rank_measures(
    specific_measures: list[dict[str, pd.DataFrame]],
    side: Optional[str] = None,
) -> list[dict[str, npt.NDArray[np.float64]]]:
    """Rank measures by information content.

    Args:
        specific_measures: List of per-trial measure dicts.
        side: Specific side to rank, or None for max across sides.

    Returns:
        List of dicts mapping state -> sorted measure arrays.
    """
    if side is None:
        # Compute max measures across all sides
        ranked = [
            {
                state: np.array(
                    ISR.max(axis=1).sort_values(ascending=False), dtype=float
                )
                for state, ISR in measures_dict.items()
            }
            for measures_dict in specific_measures
        ]
    else:
        # Compute side-specific measures
        ranked = [
            {
                state: np.array(
                    ISR[side].sort_values(ascending=False).values, dtype=float
                )
                for state, ISR in measures_dict.items()
            }
            for measures_dict in specific_measures
        ]
    return ranked


def combine_measures(
    specific_measures: list[dict[str, npt.NDArray[np.float64]]],
    states: tuple[str, ...] = ("pre", "post"),
) -> dict[str, npt.NDArray[np.float64]]:
    """Stack measures from multiple trials into arrays.

    Args:
        specific_measures: List of per-trial measure dicts.
        states: States to combine.

    Returns:
        Dict mapping state -> stacked array of shape (n_trials, n_neurons).
    """
    dst = {}
    for key in states:
        arrays = [vals[key] for vals in specific_measures if key in vals]
        if arrays:
            dst[key] = np.vstack(arrays)
    return dst


def get_nested_config(cfg: dict, key_path: str):
    """Extract a value from a nested config dict using a slash-separated path.

    Args:
        cfg: Configuration dictionary.
        key_path: Slash-separated path (e.g., 'training/lrate').

    Returns:
        Value at the specified path.
    """
    parts = key_path.split("/")
    value = cfg[parts[0]]
    for part in parts[1:]:
        value = value[part]
    return value
