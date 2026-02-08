# type: ignore
import os
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import ray
import xarray as xr
from joblib import Parallel, delayed

import hsnn.analysis.png.db as polydb
from hsnn import analysis
from hsnn.core.logger import logging
from hsnn import ops, utils
from hsnn.analysis import png, ResultsDatabase
from ._base import SimulatorActor, setup_ray
from . import _utils

__all__ = [
    "run_inference",
    "run_inference_chunked",
    "detect_structures",
    "detect_structures_unconstrained",
    "get_or_detect",
    "test_significance",
    "get_sensitivities"
]

logger = logging.getLogger()


def run_inference(cfg: str | Path | Mapping, imageset: Sequence[np.ndarray],
                  store_path=None, syn_params_path: Optional[Path] = None,
                  reps: int = 10, debug=False, use_ray: bool = False,
                  **sim_kwargs) -> xr.DataArray:
    cfg = deepcopy(utils.io.as_dict(cfg))
    cfg['simulation'].update(**sim_kwargs)

    if use_ray:
        setup_ray()
        sims = _get_actors(cfg, store_path, syn_params_path, reps, debug=debug)
        records: xr.DataArray = ops.concat_records(
            ray.get([sim.infer.remote(imageset) for sim in sims]), 'rep'
        )
        ray.shutdown()
    else:
        rep_records = Parallel(n_jobs=reps, backend="loky")(
            delayed(_local_infer)(cfg, imageset, store_path, syn_params_path)
            for _ in range(reps)
        )
        records: xr.DataArray = ops.concat_records(rep_records, 'rep')

    if debug:
        _assert_unique(records)
    return records


def run_inference_chunked(
    cfg: str | Path | Mapping, imageset: Sequence[np.ndarray],
    store_path=None, syn_params_path: Optional[Path] = None,
    workers: int | None = None, debug=False, use_ray: bool = False,
    **sim_kwargs
) -> xr.DataArray:
    cfg = deepcopy(utils.io.as_dict(cfg))
    cfg['simulation'].update(**sim_kwargs)
    data = list(imageset)

    if use_ray:
        setup_ray()
        n_workers = int(max(ray.available_resources().get("CPU", 1), 1)) if workers is None else workers
        n_actors = min(n_workers, len(data))
        logging.info(f"Running inference with {n_actors} actors on {len(data)} images.")
        img_offsets, sublists = zip(*_utils.chunks(data, n_actors))
        if len(sublists) != n_actors:
            raise ValueError(
                f"Number of actors ({n_actors}) does not match number of chunks ({len(sublists)})."
            )
        actors = _get_actors(cfg, store_path, syn_params_path, n_actors, debug=debug)
        futures = []
        for actor, subarr in zip(actors, sublists, strict=True):
            futures.append(actor.infer.remote(subarr))
        partials = ray.get(futures)
        ray.shutdown()
    else:
        n_workers = (os.cpu_count() or 1) if workers is None else workers
        n_chunks = min(n_workers, len(data))
        logging.info(f"Running inference with {n_chunks} chunks on {len(data)} images (local).")
        img_offsets, sublists = zip(*_utils.chunks(data, n_chunks))
        if len(sublists) != n_chunks:
            raise ValueError(
                f"Number of chunks ({n_chunks}) does not match expected ({len(sublists)})."
            )
        partials = Parallel(n_jobs=n_chunks, backend="loky")(
            delayed(_local_infer_chunk)(cfg, subarr, store_path, syn_params_path)
            for subarr in sublists
        )

    partials = [da.assign_coords(img=np.arange(img_offset, img_offset + da.sizes["img"]))
                for da, img_offset in zip(partials, img_offsets)]
    return xr.concat(partials, dim="img").sortby("img").expand_dims({'rep': [0]})


def detect_structures(
    nrn_ids: Iterable[int],
    layer: int,
    results_db: ResultsDatabase,
    targets: Optional[dict] = None,
    w_min: float = 0.5,
    tol: float = 3.0,
    merge: bool = True,
    return_unique: bool = True,
    num_workers: int | None = None,
    use_ray: bool = False,
    spade_kwargs: Optional[dict] = None,
) -> Sequence[png.PNG]:
    """Detect structural PNGs for a set of neurons.

    Args:
        nrn_ids: Neuron identifiers whose projections are evaluated.
        layer: Layer index used to query PNG production.
        results_db: Database providing simulation outputs for detection.
        targets: Optional target configuration passed to the detector.
        w_min: Minimum weight threshold for inclusion in detections.
        projs: Projection types to consider during detection.
        tol: Spatial tolerance parameter for merging detections.
        merge: Whether to merge overlapping detections.
        return_unique: Whether to keep only unique detections per neuron.
        num_workers: Number of worker processes to use locally.
        use_ray: Whether to run detection via Ray remote actors.
        spade_kwargs: Additional keyword arguments forwarded to Spade detection.

    Returns:
        Sequence[png.PNG]: Sorted list of detected PNG structures.
    """
    if not use_ray:
        # --- joblib backend (local processes) ---
        n_jobs = os.cpu_count() if num_workers is None else num_workers
        detections: Sequence[Sequence[png.PNG]] = Parallel(
            n_jobs=n_jobs, backend="loky"
        )(
            delayed(png.detection.mine_triads)(
                nrn_id, layer, results_db, targets, w_min, tol, merge,
                return_unique, spade_kwargs
            )
            for nrn_id in nrn_ids
        )
    else:
        # --- Ray backend ---
        setup_ray(num_cpus=num_workers)
        mine_structural = ray.remote(png.detection.mine_triads)
        futures = []
        for nrn_id in nrn_ids:
            pngs = mine_structural.remote(
                nrn_id, layer, results_db, targets, w_min, tol, merge,
                return_unique, spade_kwargs
            )
            futures.append(pngs)
        detections: Sequence[Sequence[png.PNG]] = ray.get(futures)
        ray.shutdown()

    # _utils.log_png_hash_collisions(layer, nrn_ids, detections)

    ret = []
    for pngs in detections:
        ret.extend(pngs)
    return png.detection.sorted_pngs(ret)


def detect_structures_unconstrained(
    nrn_ids: Iterable[int],
    layer: int,
    results_db: ResultsDatabase,
    targets: Optional[dict] = None,
    tol: float = 3.0,
    merge: bool = True,
    return_unique: bool = True,
    num_workers: int | None = None,
    use_ray: bool = False,
    spade_kwargs: Optional[dict] = None,
) -> Sequence[png.PNG]:
    """Detect triplet PNGs with [L-1, L, L] structure WITHOUT synaptic constraints.

    Args:
        nrn_ids: Neuron identifiers whose projections are evaluated.
        layer: Layer index used to query PNG production.
        results_db: Database providing simulation outputs for detection.
        targets: Optional target configuration passed to the detector.
        tol: Tolerance parameter for merging detections.
        merge: Whether to merge overlapping detections.
        return_unique: Whether to keep only unique detections per neuron.
        num_workers: Number of worker processes to use locally.
        use_ray: Whether to run detection via Ray remote actors.
        spade_kwargs: Additional keyword arguments forwarded to Spade detection.

    Returns:
        Sequence[png.PNG]: Sorted list of detected PNG structures (unconstrained).
    """
    if not use_ray:
        # --- joblib backend (local processes) ---
        n_jobs = os.cpu_count() if num_workers is None else num_workers
        detections: Sequence[Sequence[png.PNG]] = Parallel(
            n_jobs=n_jobs, backend="loky"
        )(
            delayed(png.detection.mine_triads_unconstrained)(
                nrn_id, layer, results_db, targets, tol, merge,
                return_unique, spade_kwargs
            )
            for nrn_id in nrn_ids
        )
    else:
        # --- Ray backend ---
        setup_ray(num_cpus=num_workers)
        mine_unconstrained = ray.remote(png.detection.mine_triads_unconstrained)
        futures = []
        for nrn_id in nrn_ids:
            pngs = mine_unconstrained.remote(
                nrn_id, layer, results_db, targets, tol, merge,
                return_unique, spade_kwargs
            )
            futures.append(pngs)
        detections: Sequence[Sequence[png.PNG]] = ray.get(futures)
        ray.shutdown()

    # Deduplicate across all detections (same PNG may be detected for different focal neurons)
    ret = set()
    for pngs in detections:
        ret.update(pngs)
    return png.detection.sorted_pngs(ret)


def get_or_detect(
    nrn_ids: Iterable[int],
    layer: int,
    index: int,
    png_db: polydb.PNGDatabase,
    results_db: ResultsDatabase,
    method: str = 'structural',
    num_workers: int | None = None,
    **detect_kwargs,
) -> Sequence[png.PNG]:
    """Retrieve stored PNGs and/or run detection for missing neurons.

    Args:
        nrn_ids: Neuron indices to fetch or detect PNGs for.
        layer: Layer identifier used to group PNGs.
        index: Run index that partitions PNG results within the database.
        png_db: PNG metadata/database interface.
        results_db: Simulation results database used for detection.
        method: Detection strategy: 'structural' (constrained) or 'unconstrained'.
        num_workers: Worker count to forward to detection logic.
        **detect_kwargs: Additional keyword arguments passed to detection function.

    Raises:
        ValueError: If the provided method is unsupported.

    Returns:
        The PNGs associated with `nrn_ids` from the database.
    """
    detect_nrn_ids = set(nrn_ids) - set(png_db.get_run_nrns(layer, index))
    if len(detect_nrn_ids):
        nrn_ids_sorted = sorted(detect_nrn_ids)
        if method == 'structural':
            polygrps = detect_structures(
                nrn_ids_sorted, layer, results_db, num_workers=num_workers, **detect_kwargs
            )
        elif method == 'unconstrained':
            polygrps = detect_structures_unconstrained(
                nrn_ids_sorted, layer, results_db, num_workers=num_workers, **detect_kwargs
            )
        else:
            raise ValueError(f"invalid method: '{method}'")
        png_db.insert_pngs(polygrps)
        png_db.insert_runs(nrn_ids_sorted, layer, index)
        logging.info(f"Inserted neurons: '{nrn_ids}'")
    return png_db.get_pngs(layer, nrn_ids, index=index)


def test_significance(
    polygrps: Sequence[png.PNG],
    results_db: ResultsDatabase,
    num_workers: int | None = None,
    use_ray: bool = False,
    **kwargs,
) -> Sequence[bool]:
    """Test PNGs for statistical significance.

    Args:
        polygrps: PNG structures to evaluate.
        results_db: Database that stores simulation results for comparison.
        num_workers: Number of worker processes to spawn.
            Defaults to None (uses os.cpu_count()).
        use_ray: Whether to dispatch significance tests through Ray remote
            actors.
        **kwargs: Forwarded to `png.detection.issignificant`.

    Returns:
        Sequence[bool]: Significance result per PNG.
    """
    if not use_ray:
        # --- joblib backend (local processes) ---
        n_jobs = os.cpu_count() if num_workers is None else num_workers

        significance_tests: Sequence[bool] = Parallel(
            n_jobs=n_jobs,
            backend="loky",
        )(
            delayed(png.detection.issignificant)(
                polygrp, results_db, **kwargs
            )
            for polygrp in polygrps
        )
        return list(significance_tests)

    # --- Ray backend ---
    setup_ray(num_cpus=num_workers)
    issignificant = ray.remote(png.detection.issignificant)
    futures = []
    for polygrp in polygrps:
        future = issignificant.remote(
            polygrp, results_db, **kwargs
        )
        futures.append(future)
    significance_tests: Sequence[bool] = ray.get(futures)
    ray.shutdown()
    return significance_tests


def get_sensitivities(nrn_id: int, layer: int, duration: float, offset: float,
                      records: xr.DataArray, syn_params: pd.DataFrame,
                      input_shape: tuple = (8, 128, 128)) -> pd.DataFrame:
    """Gets sensitivities of neuron for each of the images in `imageset` with respect
    to each (X, Y) input position, based on averaged neuronal firing rates. The neuron's
    sensitivities are averaged over all the channels at each input position.
    TODO: Move this closer to analysis code (this doesn't use Ray / tasks / cluster logic)

    Args:
        nrn_id (int): Focal neuron.
        layer (int): Neuron layer.
        duration (float): Duration to infer neuronal responses over.
        offset (float): Offset to apply when inferring neuronal responses.
        records (xr.DataArray): Spike recordings, with dims (`rep`, `img`, `layer`, `nrn_cls`).
        syn_params (pd.DataFrame): Synaptic parameters containing `weights`.
        input_shape (tuple, optional): Input layer shape (CxHxW). Defaults to (8, 128, 128).

    Raises:
        ValueError: Invalid spike record dims.

    Returns:
        pd.DataFrame: Neuronal sensitivites for each image, indexed by X, Y.
    """
    _DIMS = {'rep', 'img', 'layer', 'nrn_cls'}
    if set(records.dims) != _DIMS:
        raise ValueError(f"Invalid records dims: {set(records.dims)}")
    partial_fn = partial(_sensitivities_task, nrn_id, layer, duration, offset,
                         records, syn_params, input_shape)
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(partial_fn, records['img'].values))
    return pd.concat(results, axis=1, join='outer')


def _make_local_simulator(cfg, store_path=None, syn_params_path=None,
                          randomised_state=True):
    """Create a local Simulator instance from config, optionally restoring state.

    When ``randomised_state`` is True (the default) the network seed is
    re-drawn at random so that each worker produces independent stochastic
    results — matching the behaviour of :class:`SimulatorActor`.
    """
    from hsnn.simulation import Simulator
    sim = Simulator.from_config(cfg)
    if randomised_state:
        sim.network.seed_val = None
    if store_path is not None:
        sim.restore(store_path)
    if syn_params_path is not None:
        sim.restore_delays(syn_params_path)
    return sim


def _local_infer(cfg, imageset, store_path, syn_params_path):
    """Run a single inference rep locally (picklable for joblib)."""
    sim = _make_local_simulator(cfg, store_path, syn_params_path)
    return sim.infer(imageset)


def _local_infer_chunk(cfg, chunk, store_path, syn_params_path):
    """Run inference on an image chunk locally (picklable for joblib)."""
    sim = _make_local_simulator(cfg, store_path, syn_params_path)
    return sim.infer(list(chunk))


def _assert_unique(records: xr.DataArray):
    """Assert recordings are unique for debugging.
    """
    hs = []
    for rec in records.sel(img=0, layer=4, nrn_cls='EXC').values.flat:
        hs.append(hash(tuple(rec.spike_events[0])))
    assert len(set(hs)) == len(hs)


def _get_actors(cfg, store_path=None, syn_params_path: Optional[Path] = None,
                num_actors: int = 10, debug: bool = False) -> Sequence[SimulatorActor]:
    sims = []
    for i in range(num_actors):
        sim = SimulatorActor.remote(cfg)
        if store_path is not None:
            sim.restore.remote(store_path)
        if syn_params_path is not None:
            sim.restore_delays.remote(syn_params_path)
        if debug:
            syn_params = ray.get(sim.get_syn_params.remote())
            base_dir = syn_params_path.parent / '_actors'
            base_dir.mkdir(parents=True, exist_ok=True)
            syn_params.to_parquet(base_dir / f"syn_params_actor_{i}.parquet")
        sims.append(sim)
    return sims


def _sensitivities_task(nrn_id, layer, duration, offset, records,
                        syn_params, input_shape, idx) -> pd.DataFrame:
    rates_sel = analysis.infer_rates(
        records.sel(img=idx, layer=slice(1, layer), nrn_cls='EXC'),
        duration, offset
    )
    surrogates = rates_sel.mean(dim='rep')
    # TODO: Remove all refs to `analysis.get_sensitivites`
    # deltas = analysis.get_sensitivities([nrn_id], surrogates, syn_params)
    deltas = analysis.get_exact_sensitivities(
        [nrn_id], surrogates, syn_params, np.prod(input_shape)
    )
    sensitivities = analysis.reduce_sensitivities(deltas, input_shape)
    return sensitivities.to_frame(idx)
