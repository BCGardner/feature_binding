#!/usr/bin/env python3
"""Computes ranked information measures from hyperparameter sweep experiments.

This script takes inference results from hyperparameter sweep experiments and
computes stimulus-specific information measures, outputting serialised results
organised by hyperparameter value.

The swept hyperparameter is automatically detected from the experiment's hparams.

Output structure:
{
    hparam_value_1: {'pre': np.ndarray, 'post': np.ndarray},  # shape: (n_trials, n_neurons)
    hparam_value_2: {'pre': np.ndarray, 'post': np.ndarray},
    ...
}

Examples:
# Compute max information measures for learning rate sweep
./scripts/figures/compute_information_sweep.py ./experiments/n3p2/train_n3p2_sweep_lrate_111125 \
    --target 1

# Compute side-specific measures for competition sweep
./scripts/figures/compute_information_sweep.py ./experiments/n3p2/train_n3p2_sweep_competition_121125 \
    --side left --target 1

# Include baseline experiment for comparison
./scripts/figures/compute_information_sweep.py ./experiments/n3p2/train_n3p2_sweep_delays_121125 \
    --baseline n3p2/train_n3p2_lrate_0_04_181023 --baseline_model ALL

# Post-state only (default), or include pre-state
./scripts/figures/compute_information_sweep.py ./experiments/n3p2/train_n3p2_sweep_lrate_111125 \
    --states post

./scripts/figures/compute_information_sweep.py ./experiments/n3p2/train_n3p2_sweep_lrate_111125 \
    --states pre post
"""

import logging
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from pathlib import Path
from typing import Any

from hsnn.core.logger import get_logger
from hsnn.pipeline.information import (
    combine_measures,
    compute_trial_measures,
    get_nested_config,
    load_results_trials,
    rank_measures,
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
    """Extract the hyperparameter name from experiment hparams.

    Args:
        expt: Experiment handler.

    Returns:
        Hyperparameter name (e.g., 'lrate', 'I2E', 'delay_max').
    """
    hparam_col = expt.hparams.columns[0]
    # Extract the last part of the path
    return hparam_col.split("/")[-1]


def get_hparam_value(trial: handler.TrialView, hparam_name: str) -> Any:
    """Get the hyperparameter value for a trial.

    Args:
        trial: Trial view.
        hparam_name: Name of the hyperparameter.

    Returns:
        Hyperparameter value.
    """
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
    trials_baseline = []
    if opt.baseline is not None:
        expt_baseline = handler.ExperimentHandler(opt.baseline)
        trials_dict = expt_baseline.metadata.get_trials_dict(opt.baseline_model)
        analysis_key = opt.baseline_analysis or "inference"
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

    def sort_key(trial: handler.TrialView):
        return (get_hparam_value(trial, hparam_name), get_trial_id(trial))

    trials_combined = sorted(trials_combined, key=sort_key)

    # Extract hparam values for each trial
    hparams = [get_hparam_value(trial, hparam_name) for trial in trials_combined]
    logger.info(f"Unique hparam values: {sorted(set(hparams))}")

    # Load dataset labels (all trials share the same dataset)
    cfg = trials_combined[0].config
    _, labels = io.get_dataset(cfg["training"]["data"], return_annotations=True)

    # Prepare loading kwargs
    states = tuple(opt.states)
    results_kwargs = {"noise": opt.noise, "subdir": opt.subdir}
    offset = 0.0 if opt.analysis == "onsets" else opt.offset

    # === Load data === #
    logger.info(f"Loading results for states: {states}")
    results = load_results_trials(trials_combined, state=states, **results_kwargs)

    # Determine duration from first result
    first_state = states[0]
    duration = results[0][first_state].item(0).duration - offset
    logger.info(f"Duration: {duration}, Offset: {offset}")

    # === Compute measures === #
    logger.info("Computing information measures...")
    specific_measures = compute_trial_measures(
        results=results,
        labels=labels,
        target=opt.target,
        duration=duration,
        offset=offset,
        layer=opt.layer,
        states=states,
    )

    # Rank measures
    ranked_measures = rank_measures(specific_measures, side=opt.side)

    # Group by hparam value
    measures_by_hparam: dict[Any, list[dict]] = defaultdict(list)
    for idx, hparam in enumerate(hparams):
        measures_by_hparam[hparam].append(ranked_measures[idx])

    # Stack within each hparam group
    measures_stacked = {
        hparam: combine_measures(measures_list, states=states)
        for hparam, measures_list in measures_by_hparam.items()
    }

    # Log summary
    for hparam, measures_dict in measures_stacked.items():
        for state, arr in measures_dict.items():
            logger.info(f"hparam={hparam}, state={state}: shape {arr.shape}")

    # === Save results === #
    results_dir = opt.output_dir / dataset_name / hparam_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Build filename
    fname_parts = ["information", "sweep"]
    fname_parts += ["max"] if opt.side is None else [opt.side]
    fname_parts += ["convex"] if opt.target == 1 else ["concave"]
    fname = io.formatted_name("_".join(fname_parts), "pkl", noise=opt.noise)
    output_path = results_dir / fname

    # Load existing or create new
    if output_path.exists() and not opt.force:
        existing = io.load_pickle(output_path)
        logger.info(f"Loaded existing results with hparams: {list(existing.keys())}")
        # Merge with new results
        existing.update(measures_stacked)
        measures_stacked = existing

    io.save_pickle(measures_stacked, output_path)
    logger.info(f"Saved results to: '{output_path}'")
    logger.info(f"Hyperparameter values in results: {sorted(measures_stacked.keys())}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Compute information measures from hyperparameter sweep experiments."
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
        default="inference",
        help="Analysis type key in baseline metadata",
    )
    parser.add_argument(
        "--analysis",
        type=str,
        default="inference",
        choices=["inference", "detection", "noise", "onsets"],
        help="Analysis type (affects offset handling)",
    )
    parser.add_argument(
        "--states",
        type=str,
        nargs="+",
        default=["post"],
        choices=["pre", "post"],
        help="Training states to compute (default: post only)",
    )
    parser.add_argument("--layer", type=int, default=4, help="Layer to analyse")
    parser.add_argument("--target", type=int, default=1, help="Target class label")
    parser.add_argument(
        "--offset", type=float, default=50.0, help="Temporal offset (ms)"
    )
    parser.add_argument(
        "--side",
        type=str,
        default=None,
        help="Specific side, or max across all sides if None",
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
