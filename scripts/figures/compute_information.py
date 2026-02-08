#!/usr/bin/env python3
"""Computes ranked information measures from Trial inference recordings.

This script takes inference results and computes stimulus-specific information
measures for neurons, outputting serialised results organised by architecture
and training state for downstream analysis/visualisation.

This computes the maximum information conveyed by each neuron regarding the target feature
(e.g. convex/concave), irrespective of where the target is located, and conditioned on
there being a higher firing activity for that target.

Examples:
./scripts/figures/compute_information.py ./experiments/n3p2/train_n3p2_lrate_0_04_181023 SEMI \
    --analysis inference --layer 4 --target 1

./scripts/figures/compute_information.py ./experiments/n3p2/train_n3p2_lrate_0_04_181023 ALL \
    --analysis noise --noise 10 --subdir noise

for arch in FF SEMI ALL; do
    ./scripts/figures/compute_information.py ./experiments/n4p2/train_n4p2_lrate_0_02_181023 $arch
done

for side in top left right; do
    ./scripts/figures/compute_information.py ./experiments/n3p2/train_n3p2_lrate_0_04_181023 ALL \
        --side $side  --target 1 --output_dir ./out/figures/supplementary/information -v
done

for side in top left bottom right; do
    ./scripts/figures/compute_information.py ./experiments/n4p2/train_n4p2_lrate_0_02_181023 ALL \
        --side $side  --target 1 --output_dir ./out/figures/supplementary/information -v
done
"""

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

from hsnn.core.logger import get_logger
from hsnn.pipeline.information import (
    combine_measures,
    compute_trial_measures,
    load_results_trials,
    rank_measures,
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

    # Prepare kwargs for loading results
    states = ("pre", "post")
    results_kwargs = {"noise": opt.noise, "subdir": opt.subdir}
    offset = 0.0 if opt.analysis == "onsets" else opt.offset

    # === Load data === #
    results = load_results_trials(trials, state=states, **results_kwargs)
    duration = results[0]["post"].item(0).duration - offset
    logger.info(f"Duration: {duration}, Offset: {offset}")

    # Load dataset labels
    cfg = trials[0].config
    _, labels = io.get_dataset(cfg["training"]["data"], return_annotations=True)

    # === Compute measures === #
    specific_measures = compute_trial_measures(
        results=results,
        labels=labels,
        target=opt.target,
        duration=duration,
        offset=offset,
        layer=opt.layer,
        states=states,
    )

    # Rank and combine measures
    ranked_measures = rank_measures(specific_measures, side=opt.side)
    measures_ranked_stacked = combine_measures(ranked_measures, states=states)

    # === Save results === #
    results_dir = opt.output_dir / dataset_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Build filename
    fname_parts = ["information"]
    fname_parts += ["max"] if opt.side is None else [opt.side]
    fname_parts += ["convex"] if opt.target == 1 else ["concave"]
    fname = io.formatted_name("_".join(fname_parts), "pkl", noise=opt.noise)
    output_path = results_dir / fname

    if output_path.exists() and not opt.force:
        measures_dict = io.load_pickle(output_path)
        logger.info(f"Loaded existing results: {list(measures_dict.keys())}")
    else:
        measures_dict = {}

    measures_dict[opt.model_type] = measures_ranked_stacked
    io.save_pickle(measures_dict, output_path)
    logger.info(f"Saved results to: '{output_path}'")
    logger.info(f"Architectures in results: {list(measures_dict.keys())}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Compute information measures from inference results."
    )
    parser.add_argument("expt_dir", type=str, help="Path to experiment directory")
    parser.add_argument(
        "model_type", type=str, choices=["FF", "SEMI", "ALL"], help="Architecture type"
    )
    parser.add_argument(
        "--analysis",
        type=str,
        default="inference",
        choices=["inference", "detection", "noise", "onsets"],
        help="Analysis type from metadata",
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
        help="Specific side, take max measure across all sides if None",
    )
    parser.add_argument("--noise", type=int, default=0, help="Noise amplitude")
    parser.add_argument("--subdir", type=str, default=None, help="Results subdirectory")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(io.BASE_DIR / "out/figures/information"),
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
