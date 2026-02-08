#!/usr/bin/env python3
"""Runs inference on an SNN model, recreated (and restored) from an experiment.

The spike recordings are output to the specified directory as `inference*.pkl.gz`
for analysis.
"""

import logging
import os
import tempfile
import shutil
from argparse import ArgumentParser, Namespace
from pathlib import Path
from time import time

from hsnn.utils import io, handler
from hsnn.simulation import Simulator
from hsnn.cluster import tasks
from hsnn.core import Projection
from hsnn.core.logger import get_logger
from hsnn import pipeline

logger = get_logger(__name__)


def main(opt: Namespace):
    # === Setup === #

    # Get a handle on Experiment.Trial[.checkpoint]
    logger.info(opt)
    expt = handler.ExperimentHandler(Path.cwd() / opt.expt_dir)
    trial = expt[opt.trial]
    cfg = trial.config
    if opt.chkpt is not None:
        store_path = trial.checkpoints[opt.chkpt].store_path
    else:
        store_path = None
    logger.info(f"Model store path: {store_path}")

    # Optionally inject noise
    if opt.noise > 0:
        cfg['training']['data']['transforms']['gaussiannoise'] = [opt.noise]

    # Prepare inference results
    results_path = handler.get_results_path(
        trial, opt.chkpt, subdir=opt.subdir,
        delay_jitter=opt.delay_jitter, noise=opt.noise
    )
    logger.info(f"Results will be saved to: '{results_path}'")

    # Load imageset
    if opt.data is not None:
        logger.info(f"Imageset loaded from '{opt.data}'")
        cfg['training']['data']['name'] = opt.data
    imageset, _ = io.get_dataset(cfg['training']['data'], return_annotations=True)

    if opt.no_action:
        logger.info("Exiting with no action (-n flag used)")
        return

    # === Run inference === #

    if results_path.exists():
        if opt.force:
            os.remove(results_path)
            logger.info(f"Deleted existing file: '{results_path}'")
        else:
            raise FileExistsError(f"Results '{results_path}' already exists: "
                                  "use '-f' flag to overwrite")

    # Optionally jitter synapse axonal conduction delays
    if opt.delay_jitter > 0.0:
        sim = Simulator.from_config(cfg)
        if store_path is not None:
            sim.restore(store_path)
        projections = [proj for proj in sim.network.projections
                       if proj not in {Projection.I2E, Projection.E2I}]
        pipeline.jitter_synapse_delays(
            sim.network.layers[1:], opt.delay_jitter, projections)
        syn_params = sim.network.get_syn_params(
            return_delays=True, projections=projections)
        assert syn_params is not None
        # Persist syn_params with target projections to disk
        syn_params_path = handler.get_artifact_path(
            'syn_params', trial, opt.chkpt, subdir=opt.subdir,
            delay_jitter=opt.delay_jitter, noise=opt.noise, ext='.parquet')
        syn_params_path.parent.mkdir(parents=True, exist_ok=True)
        syn_params.to_parquet(syn_params_path)
        logger.info(f"Saved jittered synaptic delays to: '{syn_params_path}'")
    else:
        syn_params_path = None

    # Optionally preprompt the network with a select image before running inference
    tmp_dir = None
    if opt.preprompt:
        tmp_dir = tempfile.mkdtemp()
        sim = Simulator.from_config(cfg)
        if store_path is not None:
            sim.restore(store_path)
        sim.network.lrate = 0.0
        sim.present([imageset[opt.preprompt]])
        store_path = Path(tmp_dir) / 'tmp_netstate.pkl'
        sim.network.store(file_path=store_path)
        logger.info(f"Saved to temporary model store: '{store_path}'")

    # Run inference (parallelised)
    t_start = time()
    inference_kwargs = {
        'store_path': store_path, 'syn_params_path': syn_params_path,
        'duration': opt.duration, 'duration_relax': opt.duration_relax,
        'debug': opt.debug
    }
    if opt.chunked:
        results = tasks.run_inference_chunked(
            cfg, imageset, workers=opt.num_workers, **inference_kwargs
        )
    else:
        results = tasks.run_inference(
            cfg, imageset, reps=opt.reps, **inference_kwargs
        )
    t_elapsed = time() - t_start
    logger.info(f"Time for inference: {t_elapsed:.1f} s")
    # Cleanup
    if tmp_dir is not None:
        shutil.rmtree(tmp_dir)
    # Save results
    io.save_pickle(results, results_path, parents=True)
    if results_path.exists():
        logger.info(f"Results saved to '{results_path}'")
    else:
        raise FileNotFoundError(f"Failed to write results: '{results_path}'")


if __name__ == '__main__':
    parser = ArgumentParser(description="Run inference on a Trial model.")
    parser.add_argument('expt_dir', type=str, help="Experiment directory (relative to pwd).")
    parser.add_argument('trial', type=int, help="Trial index.")
    parser.add_argument('--chkpt', type=int, default=None, help="Checkpoint index.")
    parser.add_argument('--subdir', type=str, default=None,
                        help="Name of subdirectory to store results inside.")
    parser.add_argument('--data', type=str, default=None,
                        help="Data directory to load imageset from (defaults to Trial config).")
    parser.add_argument('--duration', type=float, default=250.0)
    parser.add_argument('--duration_relax', type=float, default=0.0)
    parser.add_argument('--reps', type=int, default=20)
    parser.add_argument('--delay_jitter', type=float, default=0.0,
                        help="Apply multiplicative jitter to exc. delays with this sigma value.")
    parser.add_argument('--noise', type=int, default=0)
    parser.add_argument('--preprompt', type=int, default=None,
                        help="Preprompt network with this image idx before inference.")
    parser.add_argument('--num_workers', type=int, default=None,
                        help="Number of workers assigned if the images are chunked.")
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--chunked', action='store_true', default=False,
                        help="Chunk the images into subsets for faster processing (one rep).")
    parser.add_argument('-f', '--force', action='store_true', default=False,
                        help="Overwrite previous results.")
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    parser.add_argument('-n', '--no_action', action='store_true', default=False)
    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    main(args)
