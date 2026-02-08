#!/usr/bin/env python3
"""Detects three-neuron PNGs without synaptic constraints.

This script detects all triplet PNGs with layer structure [L-1, L, L] where:
- First neuron is in layer L-1
- Second and third neurons are in layer L
- Second-firing neuron is used as the index (for DB compatibility)
- NO synaptic weight or delay constraints are enforced

This is used to compute the fraction of detected PNGs that correspond to
structurally valid HFB circuits (i.e., those satisfying synaptic constraints).
"""

from argparse import ArgumentParser, Namespace
import logging
import os
from pathlib import Path
from time import time

import pandas as pd
import xarray as xr

from hsnn.utils import io, handler
from hsnn import analysis, pipeline
from hsnn.analysis.png import _utils as png_utils
import hsnn.analysis.png.db as polydb
from hsnn.simulation import Simulator
from hsnn.core.logger import get_logger

logger = get_logger(__name__)


def compute_hfb_fraction(
    db_unconstrained: polydb.PNGDatabase,
    syn_params: pd.DataFrame,
    layers: list[int] | None = None,
    position: int = 1,
    w_min: float = 0.5,
    tol: float = 3.0,
) -> dict:
    """Compute fraction of unconstrained PNGs that are valid HFBs.

    Args:
        db_unconstrained: Database containing unconstrained PNGs.
        syn_params: Network synaptic parameters.
        layers: Layers to analyze (None = all).
        position: Firing order index used for DB retrieval.
        w_min: Minimum weight for HFB constraint.
        tol: Tolerance for delay alignment.

    Returns:
        Dictionary with per-layer and overall statistics.
    """
    layer_ids = sorted(syn_params.index.unique('layer'))
    if layers is not None:
        layer_ids = [layer_id for layer_id in layer_ids if layer_id in layers]

    results = {'layers': {}, 'total_unconstrained': 0, 'total_hfb': 0}

    for layer in layer_ids:
        try:
            nrn_ids = sorted(syn_params.xs(layer, level='layer').index.unique('post'))
        except KeyError:
            continue
        polygrps = db_unconstrained.get_pngs(layer, nrn_ids, index=position)

        n_unconstrained = len(polygrps)
        n_hfb = sum(1 for p in polygrps if png_utils.isconstrained(p, syn_params, w_min, tol))

        results['total_unconstrained'] += n_unconstrained
        results['total_hfb'] += n_hfb
        results['layers'][layer] = {
            'unconstrained': n_unconstrained,
            'hfb': n_hfb,
            'fraction': n_hfb / n_unconstrained if n_unconstrained > 0 else 0.0
        }

        if n_unconstrained > 0:
            frac = n_hfb / n_unconstrained * 100
            logger.info(f"Layer {layer}: {n_hfb}/{n_unconstrained} = {frac:.1f}% HFB-consistent")
        else:
            logger.info(f"Layer {layer}: No PNGs detected")

    if results['total_unconstrained'] > 0:
        results['overall_fraction'] = results['total_hfb'] / results['total_unconstrained']
        overall_pct = results['overall_fraction'] * 100
        logger.info(f"\nOverall: {results['total_hfb']}/{results['total_unconstrained']} = {overall_pct:.1f}% HFB-consistent")
    else:
        results['overall_fraction'] = 0.0

    return results


def main(opt: Namespace):
    # === Setup === #

    logger.info(opt)
    expt = handler.ExperimentHandler(Path.cwd() / opt.expt_dir)
    trial = expt[opt.trial]
    cfg = trial.config

    # Get previous spike recordings
    results_path = handler.get_results_path(trial, opt.chkpt, subdir=opt.subdir,
                                            delay_jitter=opt.delay_jitter, noise=opt.noise)
    records: xr.DataArray = io.load_pickle(results_path)
    logger.info(f"Loaded spike recordings: '{results_path}'")

    # Get network synaptic parameters
    if opt.chkpt is not None:
        store_path = trial.checkpoints[opt.chkpt].store_path
    else:
        store_path = None
    sim = Simulator.from_config(cfg)
    if store_path is not None:
        sim.restore(store_path)
        logger.info(f"Restored model from '{store_path}'")

    if opt.delay_jitter > 0.0:
        pipeline.restore_jittered_network(opt, trial, sim)
    syn_params = sim.network.get_syn_params(return_delays=True)
    assert syn_params is not None, "Failed to restore synaptic parameters"
    logger.info(f"Maximum synaptic delay value: {syn_params['delay'].max()}")

    # Training data labels
    _, labels = io.get_dataset(cfg['training']['data'], return_annotations=True)

    # Prepare results database
    rdb_layers = slice(None) if opt.layers is None else slice(min(opt.layers) - 1, max(opt.layers))
    rdb = analysis.ResultsDatabase(
        records.sel(rep=range(opt.reps)), syn_params, labels, layer=rdb_layers,
        proj=('FF', 'E2E'), duration=opt.duration, offset=opt.offset
    )
    if opt.layers is not None:
        rdb_coords = rdb.get_coord_values("layer")
        syn_coords = rdb.syn_params.index.unique("layer")
        logger.info(f"Records scoped to layers: {rdb_coords}")
        logger.info(f"Synaptic parameters scoped to layers: {syn_coords}")

    # Prepare HFB database path (unconstrained variant)
    db_path = handler.get_hfb_path(trial, opt.chkpt, subdir=opt.subdir,
                                   delay_jitter=opt.delay_jitter, noise=opt.noise)
    # Modify path to indicate unconstrained
    db_path = db_path.parent / db_path.name.replace('.db', '_unconstrained.db')
    logger.info(f"Detections (unconstrained) will be saved to: '{db_path}'")

    if opt.no_action:
        return

    # === PNG detections (unconstrained) === #

    if db_path.exists():
        if opt.force:
            os.remove(db_path)
            logger.info(f"Deleted existing file: '{db_path}'")
        else:
            raise FileExistsError(f"Detections '{db_path}' already exists: "
                                  "use '-f' flag to overwrite")

    db = polydb.PNGDatabase(db_path)
    if not db.exists:
        db.create()
        logger.info(f"Created PNG database '{db.path}'")

    # Run detection
    t_start = time()
    spade_kwargs = {
        'winlen': opt.winlen,
    }
    logger.info(f"SPADE custom parameters: {spade_kwargs}")
    pipeline.detect_unconstrained_pngs(db, rdb, layers=opt.layers, num_workers=opt.num_workers,
                                       spade_kwargs=spade_kwargs)
    t_elapsed = time() - t_start
    logger.info(f"Time for detection: {t_elapsed:.1f} s")

    # === Compute HFB fraction === #
    if opt.compute_fraction:
        logger.info("\n=== Computing HFB fraction ===")
        results = compute_hfb_fraction(
            db, syn_params, layers=opt.layers,
            w_min=opt.w_min, tol=opt.tol
        )
        # Save results
        results_path = db_path.parent / db_path.name.replace('.db', '_fraction.pkl')
        io.save_pickle(results, results_path)
        logger.info(f"Saved fraction results to: '{results_path}'")


if __name__ == '__main__':
    parser = ArgumentParser(description="Detect unconstrained triplet PNGs.")
    parser.add_argument('expt_dir', type=str, help="Experiment directory (relative to pwd).")
    parser.add_argument('trial', type=int, help="Trial index.")
    parser.add_argument('--chkpt', type=int, default=None, help="Checkpoint index.")
    parser.add_argument('--subdir', type=str, default=None,
                        help="Name of results subdirectory.")
    parser.add_argument('--duration', type=float, default=200.0)
    parser.add_argument('--offset', type=float, default=50.0)
    parser.add_argument('--reps', type=int, default=10)
    parser.add_argument('--layers', type=int, nargs='*', default=None,
                        help="Detection layers.")
    parser.add_argument('--delay_jitter', type=float, default=0.0,
                        help="Apply multiplicative jitter to exc. delays with this sigma value.")
    parser.add_argument('--noise', type=int, default=0)
    parser.add_argument('--winlen', type=int, default=20,
                        help="Window size of SPADE.")
    parser.add_argument('--num_workers', type=int, default=None,
                        help="Number of parallel workers.")
    parser.add_argument('--w_min', type=float, default=0.5,
                        help="Minimum weight for HFB constraint check.")
    parser.add_argument('--tol', type=float, default=3.0,
                        help="Tolerance for delay alignment check.")
    parser.add_argument('--compute_fraction', action='store_true', default=True,
                        help="Compute HFB fraction after detection.")
    parser.add_argument('--no_compute_fraction', action='store_false', dest='compute_fraction',
                        help="Skip HFB fraction computation.")
    parser.add_argument('-f', '--force', action='store_true', default=False,
                        help="Overwrite previous results.")
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    parser.add_argument('-n', '--no_action', action='store_true', default=False)
    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    main(args)
