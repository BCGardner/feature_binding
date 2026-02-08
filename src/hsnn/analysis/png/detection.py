"""User functional API for running PNG analysis using SPADE as backend method.
"""

from itertools import combinations
from typing import Any, Optional, Iterable, Sequence

import numpy as np
import pandas as pd

from hsnn import ops
from hsnn.core.logger import get_logger
from ..base import get_midx
from ..spikesdb import ResultsDatabase
from .base import PNG
from .mining import SpadeMethod
from . import refinery, filters

__all__ = [
    "get_record_coords",
    "get_record_occurrences",
    "get_summary"
]

pidx = pd.IndexSlice
logger = get_logger(__name__)


def sorted_pngs(pngs: Iterable[PNG]) -> Sequence[PNG]:
    return sorted(pngs, key=lambda x: len(x.times), reverse=True)


def _select_coord(record_coords: dict[str, Any], img: int, rep: int) -> dict[str, Any]:
    indexers = dict(imgs=img, reps=rep)
    masks = []
    for k, v in indexers.items():
        masks.append(record_coords[k] == v)
    mask = np.all(masks, axis=0)
    return {k: v[mask] for k, v in record_coords.items()}


def _unique_pngs(pngs: Iterable[PNG], issorted: bool = True) -> Sequence[PNG]:
    if not issorted:
        pngs = sorted_pngs(pngs)
    return sorted_pngs(set(pngs))


def get_record_coords(png: PNG, results_db: ResultsDatabase) -> dict[str, Any]:
    """Retrieve (img, rep) coords from which this PNG has been detected, along
    with relative PNG onset times.

    Args:
        png (PNG): Detected PNG.
        results_db (ResultsDatabase): ResultsDB on which detection was run.

    Returns:
        dict: containing `imgs`, `reps`, `times_rel`.
    """
    dims = results_db.dims[1:]
    if dims != ('img', 'rep'):
        raise ValueError(f"invalid dims {dims}")
    chunk_size = results_db.duration + results_db.separation  # Duration per individual, concatenated spike train (chunk)
    indices = (png.times // chunk_size).astype(int)  # Flat indices
    # Convert flat indices to iloc[img, rep]
    imgs = results_db.data['img'].unique()
    records_ = results_db._records.sel(img=imgs)
    reps = records_['rep'].values
    coords_ = np.unravel_index(indices, records_.shape[1:])  # (img, rep)
    return {
        'imgs':   imgs[coords_[0]],
        'reps':   reps[coords_[1]],
        'times_rel': (png.times % chunk_size) + results_db.offset
    }


def run_spade(
    results_db: ResultsDatabase,
    refine: Optional[refinery.Refine] = None,
    spade_kwargs: Optional[dict] = None,
    attach_coords: bool = True,
) -> Sequence[PNG]:
    """Runs the SPADE method by combining all spike patterns present in `results_db`.

    Args:
        results_db (ResultsDatabase): The database of spike recordings to run detection on.
        refine (Optional[refinery.Refine], optional): Post-detection filtering steps. Defaults to None.
        spade_kwargs (Optional[dict], optional): Key-value pairs to pass to SPADE method. Defaults to None.
        attach_coords (bool, optional): Attaches `imgs`, `reps`, `times_rel` to detected PNGs. Defaults to True.

    Returns:
        Sequence[PNG]: Sequence of mined, and optionally refined, PNGs.
    """
    patterns = results_db.get_patterns()
    duration_max = results_db.get_max_duration()
    spade_kwargs = spade_kwargs or {}
    spade = SpadeMethod(refine=refine, **spade_kwargs)
    polygrps = spade(patterns, duration_max)
    if attach_coords:
        for polygrp in polygrps:
            coords = get_record_coords(polygrp, results_db)
            polygrp.imgs = coords['imgs']
            polygrp.reps = coords['reps']
            polygrp.times_rel = coords['times_rel']
    return polygrps


def mine_selected(
    nrn: int,
    layer: int,
    targets: dict,
    results_db: ResultsDatabase,
    w_min: float = 0.5,
    proj: tuple = ('FF', 'E2E'),
    refine: Optional[refinery.Refine] = refinery.DropRepeating(),
    spade_kwargs: Optional[dict] = None,
) -> Sequence[PNG]:
    """Detects PNGs from a collection of spike recordings linked to a select `nrn` in `layer`.
    Spike records from `results_db` are filtered to those images with the desired `targets`.
    Only presynaptic neurons projecting to `nrn` with weights greater than `w_min` are included.

    Args:
        nrn (int): The select neuron linked to the collection of mined spike recordings.
        layer (int): The layer in which the select neuron resides.
        targets (dict): Filters spike recordings to those for images containing these target features.
        results_db (ResultsDatabase): The mined database of spike recordings.
        w_min (float, optional): The minimum weights of afferent synapses on `nrn`. Defaults to 0.5.
        proj (tuple, optional): Projections between neurons to filter by. Defaults to ('FF', 'E2E').
        refine (Optional[refinery.Refine], optional): Post-detection filtering steps. Defaults to refinery.DropRepeating().
        spade_kwargs (Optional[dict], optional): Key-value pairs to pass to SPADE method. Defaults to None.

    Returns:
        list[PNG]: Sequence of mined, and optionally refined, PNGs.
    """
    midx = filters.get_presyn_indices(nrn, layer, results_db.syn_params, w_min=w_min, proj=proj)
    results_db = results_db.filter_indices(midx).filter_targets(**targets)
    return run_spade(results_db, refine, spade_kwargs)


def mine_structural(
    nrn: int,
    layer: int,
    results_db: ResultsDatabase,
    targets: Optional[dict] = None,
    w_min: float = 0.5,
    projs: tuple = ('FF', 'E2E'),
    tol: float = 3.0,
    merge: bool = True,
    return_unique: bool = True,
    spade_kwargs: Optional[dict] = None,
) -> Sequence[PNG]:
    """Mine structural PNGs: specifically three-neuron HFB circuits.

    Args:
        nrn (int): Second-firing focal neuron.
        layer (int): Layer of focal neuron.
        results_db (ResultsDatabase): Database of SpikeRecords.
        targets (Optional[dict], optional): Filter to these recorded images. Defaults to None.
        w_min (float, optional): Minimum synaptic weight of all connections. Defaults to 0.5.
        projs (tuple, optional): Afferent projections all neurons must have. Defaults to ('FF', 'E2E').
        tol (float, optional): Tolerance window to determine causal connections. Defaults to 3.0.
        merge (bool, optional): Merges mined structural PNGs with `tol`. Defaults to True.
        return_unique (bool, optional): Apply set logic to only return unique PNGs. Defaults to True.
        spade_kwargs (Optional[dict], optional): Key-values to pass to SPADE. Defaults to None.

    Returns:
        Sequence[PNG]: Sequence of HFBs.
    """
    index = (layer, nrn)
    syn_params = results_db.syn_params
    targets = targets or {}
    refine = [
        refinery.DropRepeating(),
        refinery.FilterLayers([layer - 1, layer, layer]),
        refinery.FilterIndex(index, position=1),
        refinery.Constrained(syn_params, w_min, tol)
    ]
    if merge:
        refine.extend(
            [refinery.Merge(tol=tol, strategy='mean'),
             refinery.Constrained(syn_params, w_min, tol)]
        )
    midx = filters.get_structural_indices(nrn, layer, syn_params, w_min, tol)
    results_db = results_db.filter_indices(midx).filter_targets(**targets)
    polygrps = run_spade(results_db, refinery.Compose(refine), spade_kwargs)
    if return_unique:
        return _unique_pngs(polygrps)
    return polygrps


def mine_triads(
    nrn: int,
    layer: int,
    results_db: ResultsDatabase,
    targets: Optional[dict] = None,
    w_min: float = 0.5,
    tol: float = 3.0,
    merge: bool = True,
    return_unique: bool = True,
    spade_kwargs: Optional[dict] = None
) -> Sequence[PNG]:
    """Mine structural PNGs (HFB circuits) with synaptic constraints.

    Detects three-neuron PNGs with [L-1, L, L] layer structure where all
    synaptic connections satisfy weight and delay constraints.

    Args:
        nrn: Second-firing focal neuron (high-level).
        layer: Layer of focal neuron.
        results_db: Database of SpikeRecords.
        targets: Filter to these recorded images.
        w_min: Minimum synaptic weight of all connections.
        tol: Tolerance window for causal delay alignment.
        merge: Merge similar PNGs.
        return_unique: Return only unique PNGs.
        spade_kwargs: Key-values to pass to SPADE.

    Returns:
        Sequence of HFBs satisfying structural constraints.
    """
    index = (layer, nrn)
    refine = [
        refinery.DropRepeating(),
        refinery.FilterLayers([layer - 1, layer, layer]),
        refinery.FilterIndex(index, position=1),
        refinery.Constrained(results_db.syn_params, w_min, tol)
    ]
    if merge:
        refine.extend(
            [refinery.Merge(tol=tol, strategy='mean'),
             refinery.Constrained(results_db.syn_params, w_min, tol)]
        )
    refine_compose = refinery.Compose(refine)
    midxs = filters.get_triad_indexes(nrn, layer, results_db.syn_params, w_min, tol)
    results_db = results_db.filter_targets(**(targets or {}))

    polygrps: list[PNG] = []
    for midx in midxs:
        results_db_ = results_db.filter_indices(midx)
        polygrps.extend(run_spade(results_db_, refine_compose, spade_kwargs))
    if return_unique:
        return _unique_pngs(polygrps, issorted=False)
    return polygrps


def mine_triads_unconstrained(
    nrn: int,
    layer: int,
    results_db: ResultsDatabase,
    targets: Optional[dict] = None,
    tol: float = 3.0,
    merge: bool = True,
    return_unique: bool = True,
    spade_kwargs: Optional[dict] = None
) -> Sequence[PNG]:
    """Mine triplet PNGs with [L-1, L, L] layer structure WITHOUT synaptic constraints.

    Detects all three-neuron PNGs where:
    - First neuron is in layer L-1 (low-level)
    - Second neuron is `nrn` in layer L (high-level)
    - Third neuron is in layer L (binding)

    No weight or delay constraints are enforced - only layer structure and
    connectivity existence are required.

    Args:
        nrn: Second-firing focal neuron (high-level).
        layer: Layer of focal neuron.
        results_db: Database of SpikeRecords.
        targets: Filter to these recorded images.
        tol: Tolerance for merging similar PNGs (if merge=True).
        merge: Merge similar PNGs.
        return_unique: Return only unique PNGs.
        spade_kwargs: Key-values to pass to SPADE.

    Returns:
        Sequence of triplet PNGs with [L-1, L, L] structure (unconstrained).
    """
    index = (layer, nrn)
    # Refinery WITHOUT Constrained filter
    refine = [
        refinery.DropRepeating(),
        refinery.FilterLayers([layer - 1, layer, layer]),
        refinery.FilterIndex(index, position=1),
    ]
    if merge:
        refine.append(refinery.Merge(tol=tol, strategy='mean'))
    refine_compose = refinery.Compose(refine)

    # Get all possible triads without weight/delay constraints
    midxs = filters.get_triad_indexes_unconstrained(nrn, layer, results_db.syn_params)
    results_db = results_db.filter_targets(**(targets or {}))

    polygrps: list[PNG] = []
    for midx in midxs:
        results_db_ = results_db.filter_indices(midx)
        polygrps.extend(run_spade(results_db_, refine_compose, spade_kwargs))
    if return_unique:
        return _unique_pngs(polygrps, issorted=False)
    return polygrps


def issignificant(
    polygrp: PNG,
    results_db: ResultsDatabase,
    tol: float = 2.0,
    isstructural: bool = False,
    projs: tuple = ('FF', 'E2E'),
    spade_kwargs: Optional[dict] = None
) -> bool:
    spade_kwargs_ = {
        'min_occ': 3,
        'n_surr': 100,
        'alpha': 0.05,
        'bin_size': 2,
        'winlen': 10
    }
    spade_kwargs = spade_kwargs or {}
    spade_kwargs_.update(**spade_kwargs)
    syn_params = results_db.syn_params
    if isstructural:
        refine = refinery.Compose([
            refinery.Match(tuple(zip(polygrp.layers, polygrp.nrns))),
            refinery.Constrained(syn_params, w_min=0.5, tol=tol)
        ])
    else:
        refine = None
    midx = get_midx(polygrp.layers, polygrp.nrns)
    for index_ in midx:
        if index_ not in results_db.index:
            raise IndexError(f"'{index_}' not in ResultsDatabase")
    results_db = results_db.filter_indices(midx)
    polygrps_ = run_spade(results_db, refine, spade_kwargs_)
    for polygrp_ in polygrps_:
        if _isclose_polygrps(polygrp_, polygrp, tol):
            return True
    return False


def get_record_occurrences(
    img: int,
    rep: int,
    png: PNG,
    record_coords: dict[str, Any],
) -> pd.DataFrame:
    """Gets PNG occurrences within a given (img, rep) recording.

    Args:
        img (int): ImageID.
        rep (int): Repetition index.
        png (PNG): Detected PNG.
        record_coords (dict[str, Any]): Record coords containing `imgs`, `reps`, `times_rel`.

    Returns:
        pd.DataFrame: MultiIndexed by `(layer, nrn)`, containing originally recorded occurrence times.
    """
    coord = _select_coord(record_coords, img, rep)
    index = pd.MultiIndex.from_arrays([png.layers, png.nrns], names=['layer', 'nrn'])
    cols = pd.RangeIndex(0, len(coord['times_rel']), name='occ')
    return pd.DataFrame(data=png.lags[:, np.newaxis] + coord['times_rel'], index=index, columns=cols)


def get_summary(polygrp: PNG, syn_params: pd.DataFrame) -> pd.DataFrame:
    combs = list(combinations(range(len(polygrp.nrns)), 2))
    midx = []
    data = {'w': [], 'delay': [], 'lag': []}
    for j, i in combs:
        layer_pre, layer_post = polygrp.layers[j], polygrp.layers[i]
        nrn_pre, nrn_post = polygrp.nrns[j], polygrp.nrns[i]
        proj = ops.layers_to_proj(layer_post, layer_pre)
        syn_params_ = syn_params.loc[(layer_post, proj, nrn_pre, nrn_post)] # type: ignore
        data['w'].append(syn_params_['w'])
        data['delay'].append(syn_params_['delay'])
        data['lag'].append(polygrp.lags[i] - polygrp.lags[j])
        midx.append(
            (layer_post, proj, nrn_pre, nrn_post)
        )
    midx = pd.MultiIndex.from_tuples(midx, names=['layer', 'proj', 'pre', 'post'])
    df = pd.DataFrame(data, index=midx)
    return df


def _isclose_polygrps(polygrp: PNG, polygrp_ref: PNG, tol: float) -> bool:
    indices = tuple(zip(polygrp.layers, polygrp.nrns))
    indices_ref = tuple(zip(polygrp_ref.layers, polygrp_ref.nrns))
    if len(indices) == len(indices_ref) and indices == indices_ref:
        if np.abs(polygrp.lags - polygrp_ref.lags).max() <= tol:
            return True
    return False
