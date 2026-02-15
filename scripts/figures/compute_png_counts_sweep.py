#!/usr/bin/env python3
"""Computes PNG counts from hyperparameter sweep experiments.

This script takes PNG detection databases from hyperparameter sweep experiments
and computes counts per layer, outputting serialised results organised by
hyperparameter value.

The swept hyperparameter is automatically detected from the experiment's hparams.

Output structure:
{
    hparam_value_1: {'pre': np.ndarray, 'post': np.ndarray},  # shape: (n_trials, n_layers)
    hparam_value_2: {'pre': np.ndarray, 'post': np.ndarray},
    ...
}

Examples:
# Compute PNG counts for learning rate sweep (post-training only, default)
./scripts/figures/compute_png_counts_sweep.py ./experiments/n3p2/train_n3p2_sweep_lrate_111125

# Compute counts for competition sweep with baseline comparison
./scripts/figures/compute_png_counts_sweep.py ./experiments/n3p2/train_n3p2_sweep_competition_121125 \
    --baseline n3p2/train_n3p2_lrate_0_04_181023 --baseline_model ALL

# Include both pre and post states
./scripts/figures/compute_png_counts_sweep.py ./experiments/n3p2/train_n3p2_sweep_delays_121125 \
    --states pre post

# Analyse only layer 4 (default)
./scripts/figures/compute_png_counts_sweep.py ./experiments/n3p2/train_n3p2_sweep_lrate_111125 \
    --layers 4
"""

import logging
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from pathlib import Path
from typing import Any

from hsnn.core.logger import get_logger
from hsnn.pipeline.information import get_nested_config
from hsnn.pipeline.png_counts import (
    combine_counts,
    get_png_counts_per_layer,
    load_detections_trials,
)
from hsnn.utils import handler, io

logger = get_logger(__name__)

# Mapping from hparam name to config key path
HPARAM_KEYPATH_MAPPING = {
    "lrate": "training/lrate",
    "I2E": "projections/I2E/namespace/eta",
    "delay_max": "namespaces/synapses/PLASTIC/delay_max",
}


def get_hparam_name(expt: handler.ExperimentHandler) -> str:
    """Extract the hyperparameter name from experiment hparams."""
    hparam_col = expt.hparams.columns[0]
    return hparam_col.split("/")[-1]


def get_hparam_value(trial: handler.TrialView, hparam_name: str) -> Any:
    """Get the hyperparameter value for a trial."""
    key_path = HPARAM_KEYPATH_MAPPING.get(hparam_name)
    if key_path is None:
        raise ValueError(
            f"Unknown hparam '{hparam_name}'. Add mapping to HPARAM_KEYPATH_MAPPING."
        )
    return get_nested_config(trial.config, key_path)


def get_trial_id(trial: handler.TrialView) -> int:
    """Extract numeric trial ID from trial name."""
    return int(trial.name.split("_")[-1])


def main(opt: Namespace):
    logger.info(opt)

    # === Setup === #
    expt_dir = Path(opt.expt_dir).resolve()
    if expt_dir.is_relative_to(io.EXPT_DIR):
        expt_dir = expt_dir.relative_to(io.EXPT_DIR)
    expt = handler.ExperimentHandler(expt_dir)

    dataset_name = expt.logdir.parent.name
    hparam_name = get_hparam_name(expt)
    logger.info(f"Dataset: {dataset_name}, Hyperparameter: {hparam_name}")

    # Collect all trials from sweep experiment
    trials_sweep: list[handler.TrialView] = list(expt)  # type: ignore
    logger.info(f"Found {len(trials_sweep)} trials in sweep experiment")

    # Optionally add baseline trials
    trials_baseline: list[handler.TrialView] = []
    if opt.baseline is not None:
        expt_baseline = handler.ExperimentHandler(opt.baseline)
        trials_dict = expt_baseline.metadata.get_trials_dict(opt.baseline_model)
        analysis_key = opt.baseline_analysis or "detection"
        if analysis_key not in trials_dict:
            raise ValueError(
                f"Analysis '{analysis_key}' not found in baseline metadata. "
                f"Available: {list(trials_dict.keys())}"
            )
        trial_names = trials_dict[analysis_key]
        trials_baseline = [expt_baseline[name] for name in trial_names]
        logger.info(f"Added {len(trials_baseline)} baseline trials from {opt.baseline}")

    # Combine and sort trials
    trials_combined = trials_sweep + trials_baseline
    trials_combined = sorted(
        trials_combined,
        key=lambda t: (get_hparam_value(t, hparam_name), get_trial_id(t)),
    )

    # Extract hparam values for each trial
    hparams = [get_hparam_value(trial, hparam_name) for trial in trials_combined]
    logger.info(f"Unique hparam values: {sorted(set(hparams))}")

    # Prepare loading kwargs
    states = tuple(opt.states)
    detection_kwargs = {"noise": opt.noise, "subdir": opt.subdir}

    # Parse layers
    layers = list(opt.layers)
    nrn_ids = range(4096)
    index = 1
    logger.info(f"Analysing layers: {layers}")

    # === Load PNG databases === #
    logger.info(f"Loading PNG databases for states: {states}")
    databases = load_detections_trials(
        trials_combined, state=states, **detection_kwargs
    )
    logger.info(f"Loaded databases for {len(databases)} trials")

    # === Compute PNG counts per layer === #
    logger.info("Computing PNG counts per layer...")
    trial_counts: list[dict[str, list]] = []
    for idx, db_dict in enumerate(databases):
        counts_dict = {
            state: get_png_counts_per_layer(db_dict[state], layers, nrn_ids, index)
            for state in states
            if state in db_dict
        }
        trial_counts.append(counts_dict)  # type: ignore
        logger.debug(f"Trial {idx}: extracted counts for {len(counts_dict)} states")

    # Group by hparam value
    counts_by_hparam: dict[Any, list[dict]] = defaultdict(list)
    for idx, hparam in enumerate(hparams):
        counts_by_hparam[hparam].append(trial_counts[idx])

    # Stack within each hparam group
    counts_stacked = {
        hparam: combine_counts(counts_list, states=states)
        for hparam, counts_list in counts_by_hparam.items()
    }

    # Log summary
    for hparam, counts_dict in counts_stacked.items():
        for state, arr in counts_dict.items():
            total_per_layer = arr.sum(axis=0)
            logger.info(
                f"hparam={hparam}, state={state}: shape {arr.shape}, "
                f"total per layer: {total_per_layer}"
            )

    # === Save results === #
    results_dir = opt.output_dir / dataset_name / hparam_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Build filename
    layers_str = "_".join(map(str, layers))
    fname_parts = ["png_counts", "sweep", f"layers_{layers_str}"]
    fname = io.formatted_name("_".join(fname_parts), "pkl", noise=opt.noise)
    output_path = results_dir / fname

    # Load existing or create new
    if output_path.exists() and not opt.force:
        existing = io.load_pickle(output_path)
        logger.info(f"Loaded existing results with hparams: {list(existing.keys())}")
        existing.update(counts_stacked)
        counts_stacked = existing

    io.save_pickle(counts_stacked, output_path)
    logger.info(f"Saved results to: '{output_path}'")
    logger.info(f"Hyperparameter values in results: {sorted(counts_stacked.keys())}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Compute PNG counts from hyperparameter sweep experiments."
    )
    parser.add_argument("expt_dir", type=str, help="Path to sweep experiment directory")
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Path to baseline experiment for comparison (relative to EXPT_DIR)",
    )
    parser.add_argument(
        "--baseline_model",
        type=str,
        default="ALL",
        choices=["FF", "SEMI", "ALL"],
        help="Architecture type for baseline trials",
    )
    parser.add_argument(
        "--baseline_analysis",
        type=str,
        default="detection",
        help="Analysis type key in baseline metadata",
    )
    parser.add_argument(
        "--states",
        type=str,
        nargs="+",
        default=["post"],
        choices=["pre", "post"],
        help="Training states to compute (default: post only)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[4],
        help="Layers to analyse (default: 4 only)",
    )
    parser.add_argument("--noise", type=int, default=0, help="Noise amplitude")
    parser.add_argument("--subdir", type=str, default=None, help="Results subdirectory")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(io.BASE_DIR / "out/figures/supplementary/robustness"),
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
