#!/usr/bin/env python3
"""Computes ranked F1 scores from PNG detections.

This script takes PNG detection databases and computes F1 scores for each PNG's
ability to classify specific boundary elements (convex/concave), outputting
serialised results organised by architecture (for a post-trained state).

This computes either the maximum F1 score across all sides, or side-specific
F1 scores, sorted in descending order.

Examples:
./scripts/figures/compute_png_metrics.py ./experiments/n3p2/train_n3p2_lrate_0_04_181023 SEMI \
    --analysis detection --target 1

./scripts/figures/compute_png_metrics.py ./experiments/n3p2/train_n3p2_lrate_0_04_181023 ALL \
    --analysis detection --side left --target 1

for arch in FF SEMI ALL; do
    ./scripts/figures/compute_png_metrics.py ./experiments/n4p2/train_n4p2_lrate_0_02_181023 $arch
done
"""

import logging
from argparse import ArgumentParser, Namespace
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

import hsnn.analysis.png.db as polydb
from hsnn.analysis import png
from hsnn.core.logger import get_logger
from hsnn.utils import handler, io

logger = get_logger(__name__)


def load_databases_trials(
    trials: list[handler.TrialView],
    state: str = "post",
    **kwargs,
) -> list[polydb.PNGDatabase]:
    """Load PNG databases for multiple trials in parallel."""
    loader_fn = partial(handler.load_detections, state=state, **kwargs)
    with ThreadPoolExecutor() as executor:
        db_dicts = list(executor.map(loader_fn, trials))
    return [db_dict[state] for db_dict in db_dicts]


def get_polygrps_from_db(
    database: polydb.PNGDatabase,
    layer: int = 4,
    nrn_ids: Iterable = range(4096),
    index: int = 1,
) -> list[png.PNG]:
    """Get all detected PNGs from database."""
    polygrps = database.get_pngs(layer, nrn_ids, index)
    assert isinstance(polygrps, list)
    return polygrps


def get_metrics_side(
    occ_array: xr.DataArray,
    labels: pd.DataFrame,
    target: int = 1,
    threshold: float = 0.0,
    precision_min: float | None = 0.5,
) -> dict[str, pd.DataFrame]:
    """Get thresholded PNG performance metrics per side.

    Returns:
        Tabulated metrics containing `precision`, `recall`, `F1-score` per side.
    """
    metrics_side = {}
    for side in labels.drop("image_id", axis=1).columns:
        metrics_side[side] = png.stats.get_thresholded_metrics(
            occ_array, labels, side, target, threshold, precision_min
        )
    return metrics_side


def compute_occurrences_single(args: tuple) -> tuple[int, xr.DataArray | None]:
    """Compute occurrences array for a single trial's PNGs.

    Args:
        args: (idx, polygrps, num_reps, num_imgs, index, duration, offset)

    Returns:
        Index of the trial, Computed occurrences array, or None if no PNGs.
    """
    idx, polygrps, num_reps, num_imgs, index, duration, offset = args
    if len(polygrps) == 0:
        return idx, None
    occ_array = png.stats.get_occurrences_array(
        polygrps, num_reps, num_imgs, index=index, duration=duration, offset=offset
    )
    return idx, occ_array


def get_sorted_scores(
    metrics_dict: dict[str, pd.DataFrame], side: str | None = None
) -> np.ndarray:
    """Get sorted F1 scores, either max across sides or for a specific side.

    Args:
        metrics_dict: Metrics per side from get_metrics_side.
        side: Specific side to use, or None for max across all sides.

    Returns:
        Sorted F1 scores in descending order.
    """
    if side is not None:
        if side not in metrics_dict:
            raise ValueError(
                f"Side '{side}' not found. Available: {list(metrics_dict.keys())}"
            )
        scores = metrics_dict[side]["score"].values
        return np.sort(scores[~np.isnan(scores)])[::-1]

    # Compute max score across all sides for each PNG
    # First, align all scores by PNG index
    all_scores_df = pd.DataFrame(
        {s: metrics_dict[s]["score"] for s in metrics_dict.keys()}
    )
    max_scores = all_scores_df.max(axis=1).values
    return np.sort(max_scores[~np.isnan(max_scores)])[::-1]


def combine_scores(trial_scores: list[np.ndarray]) -> np.ndarray:
    """Stack scores from multiple trials, padding to max length."""
    max_len = max(len(s) for s in trial_scores)
    padded = []
    for scores in trial_scores:
        if len(scores) < max_len:
            padded.append(
                np.pad(scores, (0, max_len - len(scores)), constant_values=np.nan)
            )
        else:
            padded.append(scores)
    return np.vstack(padded)


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

    # Prepare kwargs for loading detections
    detection_kwargs = {"noise": opt.noise, "subdir": opt.subdir}
    offset = 0.0 if opt.analysis == "onsets" else opt.offset

    # === Load data === #
    # Load PNG databases
    databases = load_databases_trials(trials, state="post", **detection_kwargs)

    # Load dataset labels and imageset info
    cfg = trials[0].config
    imageset, labels = io.get_dataset(cfg["training"]["data"], return_annotations=True)
    num_imgs = len(imageset)

    # Get duration from first trial's results
    results = handler.load_results(trials[0], state="post", **detection_kwargs)
    duration = results["post"].item(0).duration - offset
    if opt.num_reps > len(results["post"]["rep"]):
        raise ValueError(
            f"num_reps {opt.num_reps} > maximum available {len(results['post']['rep'])}"
        )
    num_reps = opt.num_reps
    logger.info(f"Duration: {duration}, Offset: {offset}, Num reps: {num_reps}")

    # === Extract PNGs from databases (sequential - databases may not be picklable) === #
    nrn_ids = range(4096)
    polygrps_list = []
    for db in databases:
        polygrps_list.append(get_polygrps_from_db(db, opt.layer, nrn_ids, opt.index))
    logger.info(f"Extracted PNGs from {len(polygrps_list)} databases")

    # === Compute occurrences arrays in parallel (CPU-bound) === #
    occ_work_items = [
        (i, polygrps, num_reps, num_imgs, opt.index, duration, offset)
        for i, polygrps in enumerate(polygrps_list)
    ]

    occ_arrays: list[xr.DataArray | None] = [None] * len(polygrps_list)
    with ProcessPoolExecutor() as executor:
        for idx, occ_array in executor.map(compute_occurrences_single, occ_work_items):
            occ_arrays[idx] = occ_array
    logger.info(
        f"Computed {sum(1 for o in occ_arrays if o is not None)} occurrences arrays"
    )

    # === Compute metrics and extract sorted scores === #
    trial_scores = []
    for i, occ_array in enumerate(occ_arrays):
        if occ_array is None:
            logger.warning(f"Trial {i}: No PNGs found")
            trial_scores.append(np.array([]))
            continue
        metrics = get_metrics_side(
            occ_array, labels, opt.target, opt.threshold, opt.precision_min
        )
        scores = get_sorted_scores(metrics, opt.side)
        trial_scores.append(scores)
        logger.info(f"Trial {i}: {len(scores)} PNGs with valid scores")

    # Stack scores across trials
    scores_stacked = combine_scores(trial_scores)
    logger.info(f"Stacked scores shape: {scores_stacked.shape}")

    # === Save results === #
    results_dir = opt.output_dir / dataset_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Build filename
    fname_parts = ["png_f1"]
    fname_parts += ["max"] if opt.side is None else [opt.side]
    if opt.target == 1:
        fname_parts += ["convex"]
    else:
        fname_parts += ["concave"]
    fname = io.formatted_name("_".join(fname_parts), "pkl", noise=opt.noise)
    output_path = results_dir / fname

    # Load existing or create new
    if output_path.exists() and not opt.force:
        scores_dict = io.load_pickle(output_path)
        logger.info(f"Loaded existing results: {list(scores_dict.keys())}")
    else:
        scores_dict = {}

    scores_dict[opt.model_type] = scores_stacked
    io.save_pickle(scores_dict, output_path)
    logger.info(f"Saved results to: '{output_path}'")
    logger.info(f"Architectures in results: {list(scores_dict.keys())}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Compute F1 scores from PNG detections.")
    parser.add_argument("expt_dir", type=str, help="Path to experiment directory")
    parser.add_argument(
        "model_type", type=str, choices=["FF", "SEMI", "ALL"], help="Architecture type"
    )
    parser.add_argument(
        "--analysis",
        type=str,
        default="detection",
        choices=["detection", "onsets"],
        help="Analysis type from metadata",
    )
    parser.add_argument("--layer", type=int, default=4, help="Layer to analyse")
    parser.add_argument("--index", type=int, default=1, help="PNG position index")
    parser.add_argument(
        "--target",
        type=int,
        default=1,
        choices=[0, 1],
        help="Target class label (0=concave, 1=convex)",
    )
    parser.add_argument(
        "--offset", type=float, default=50.0, help="Temporal offset (ms)"
    )
    parser.add_argument(
        "--num_reps",
        type=int,
        default=10,
        help="Number of repetitions used for detection",
    )
    parser.add_argument(
        "--side",
        type=str,
        default=None,
        help="Specific side (e.g., 'left', 'right'), or None for max across all sides",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Occurrence threshold for positive classification",
    )
    parser.add_argument(
        "--precision_min",
        type=float,
        default=0.5,
        help="Minimum precision to include PNG (None to disable)",
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
