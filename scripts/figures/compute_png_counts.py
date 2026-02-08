#!/usr/bin/env python3
"""Computes PNG counts per layer and saves to disk.

This script computes the number of detected PNGs per layer for each trial,
organised by architecture (SEMI, ALL) and state (pre, post).

Output structure:
{
    'SEMI': {'pre': np.ndarray, 'post': np.ndarray},  # shape: (num_trials, 4)
    'ALL': {'pre': np.ndarray, 'post': np.ndarray},
    ...
}

Examples:
# Compute counts for ALL architecture on N3P2 dataset
./scripts/figures/compute_png_counts.py ./experiments/n3p2/train_n3p2_lrate_0_04_181023 ALL

# Compute counts for all valid architectures
for arch in SEMI ALL; do
    ./scripts/figures/compute_png_counts.py ./experiments/n3p2/train_n3p2_lrate_0_04_181023 $arch -v
done

for arch in SEMI ALL; do
    ./scripts/figures/compute_png_counts.py ./experiments/n4p2/train_n4p2_lrate_0_02_181023 $arch -v
done
"""

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np

from hsnn.core.logger import get_logger
from hsnn.pipeline.png_counts import (
    get_num_polygrps,
    get_polygrps_per_layer,
    load_detections_trials,
)
from hsnn.utils import handler, io

logger = get_logger(__name__)


def main(opt: Namespace):
    logger.info(opt)

    # === Setup === #
    expt_dir = Path(opt.expt_dir).resolve().relative_to(io.EXPT_DIR)
    expt = handler.ExperimentHandler(expt_dir)
    dataset_name = Path(opt.expt_dir).parent.name

    # Get trials from metadata
    trials_dict = expt.metadata.get_trials_dict(opt.model_type)
    if opt.analysis not in trials_dict:
        raise ValueError(f"Analysis type '{opt.analysis}' not found in metadata")

    trial_names = trials_dict[opt.analysis]
    trials = [expt[trial_name] for trial_name in trial_names]
    logger.info(f"Processing {len(trials)} trials for {opt.model_type}/{opt.analysis}")

    # Determine states to process
    states: tuple[str, ...] = ("post",) if opt.analysis == "noise" else ("pre", "post")
    logger.info(f"States to process: {states}")

    # Prepare kwargs for loading detections
    detection_kwargs = {"noise": opt.noise, "subdir": opt.subdir}

    # === Load PNG databases === #
    logger.info("Loading PNG databases...")
    databases = load_detections_trials(trials, state=states, **detection_kwargs)
    logger.info(f"Loaded databases for {len(databases)} trials")

    # === Compute PNG counts per layer === #
    logger.info("Computing PNG counts per layer...")
    layers = range(1, 5)
    nrn_ids = range(4096)
    index = 1

    polygrps_trials = [
        get_polygrps_per_layer(db_dict, layers, nrn_ids, index)
        for db_dict in databases
    ]
    for idx, polygrps_dict in enumerate(polygrps_trials):
        logger.debug(f"Trial {idx}: extracted PNGs for {len(polygrps_dict)} states")

    # Combine PNG counts across trials per state
    num_polygrps: dict[str, np.ndarray] = {}
    for state in states:
        num_polygrps[state] = np.vstack(
            [get_num_polygrps(polygrps_dict)[state] for polygrps_dict in polygrps_trials]
        )
        logger.info(
            f"State '{state}': counts shape {num_polygrps[state].shape}, "
            f"total per layer: {num_polygrps[state].sum(axis=0)}"
        )

    # === Save results === #
    results_dir = opt.output_dir / dataset_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Build filename
    fname = f"png_counts_noise_{opt.noise}.pkl" if opt.noise > 0 else "png_counts.pkl"
    output_path = results_dir / fname

    # Load existing or create new
    if output_path.exists() and not opt.force:
        polygrps_counts: dict[str, dict[str, np.ndarray]] = io.load_pickle(output_path)
        logger.info(f"Loaded existing results: {list(polygrps_counts.keys())}")
    else:
        polygrps_counts = {}

    # Update with new counts
    if opt.model_type in polygrps_counts and not opt.force:
        logger.warning(
            f"Architecture '{opt.model_type}' already exists in results. "
            "Use -f to overwrite."
        )
    else:
        polygrps_counts[opt.model_type] = num_polygrps
        io.save_pickle(polygrps_counts, output_path)
        logger.info(f"Saved PNG counts for '{opt.model_type}' to '{output_path}'")

    logger.info(f"Architectures in results: {list(polygrps_counts.keys())}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Compute PNG counts per layer from detection databases."
    )
    parser.add_argument("expt_dir", type=str, help="Path to experiment directory")
    parser.add_argument(
        "model_type", type=str, choices=["SEMI", "ALL"], help="Architecture type"
    )
    parser.add_argument(
        "--analysis",
        type=str,
        default="detection",
        choices=["detection", "noise"],
        help="Analysis type from metadata",
    )
    parser.add_argument("--noise", type=int, default=0, help="Noise amplitude")
    parser.add_argument("--subdir", type=str, default=None, help="Results subdirectory")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(io.BASE_DIR / "out/figures/detection"),
        help="Output directory for results",
    )
    parser.add_argument(
        "-f", "--force", action="store_true", help="Overwrite existing results"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    main(args)
