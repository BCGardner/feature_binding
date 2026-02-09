#!/usr/bin/env python3
"""Computes network responses to progressive occlusion of a left-convex feature.

This script runs inference with a "sliding curtain" occluder that progressively
masks the afferent drive to L0 Poisson neurons, simulating partial occlusion
of a diagnostic boundary feature.

The occlusion is applied after Gabor encoding and global normalisation to ensure:
1. Firing rates in unoccluded regions remain stable (no artificial gain modulation)
2. Performance degradation is attributable solely to loss of feature information

Examples:
# Run with default occlusion levels
python scripts/analysis/inference_occlusion.py \
    ./experiments/n4p2/train_n4p2_lrate_0_02_181023 15

# Run for several Trials
for trial in 3 7 15; do
    python scripts/analysis/inference_occlusion.py \
        ./experiments/n4p2/train_n4p2_lrate_0_02_181023 $trial
done

# Run with custom occlusion levels and more reps
python scripts/analysis/inference_occlusion.py \
    ./experiments/n4p2/train_n4p2_lrate_0_02_181023 15 \
    --bump_start 10 --bump_width 21 \
    --occlusion_levels 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
"""

import logging
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from time import time
from typing import Sequence

import numpy as np
import xarray as xr

from hsnn.cluster import tasks
from hsnn.core.logger import get_logger
from hsnn.utils import handler, io

logger = get_logger(__name__)


def _prepare_occludable_config(cfg: dict, mask_width: int) -> dict:
    """Prepare config with occludable encoder and mask width.

    Args:
        cfg: Base configuration dictionary.
        mask_width: Width of occlusion curtain in pixels.

    Returns:
        Modified config with occludable encoder settings.
    """
    from copy import deepcopy

    cfg = deepcopy(cfg)

    # Switch encoder to occludable variant
    cfg["network"]["encoder"] = "occludablegabor"

    # Add mask_width to encoder config for the OccludableGaborEncoder
    cfg["encoder"]["mask_width"] = mask_width

    return cfg


def run_occlusion_sweep(
    base_cfg: dict,
    imageset: Sequence[np.ndarray],
    store_path: Path | None,
    bump_start: int,
    bump_width: int,
    occlusion_levels: Sequence[float],
    duration: float,
    reps: int,
    num_workers: int | None = None,
) -> xr.DataArray:
    """Run inference across multiple occlusion levels using tasks.run_inference.

    Args:
        base_cfg: Base configuration dictionary.
        imageset: Images with occlusion applied.
        store_path: Path to trained model checkpoint.
        bump_start: Pixel column where the convex feature starts.
        bump_width: Width of the feature in pixels.
        occlusion_levels: Fraction of the feature to occlude [0.0, 1.0].
        duration: Presentation duration in ms.
        reps: Number of repetitions per occlusion level.
        num_workers: Number of workers for parallel inference.

    Returns:
        xr.DataArray with dimensions (occlusion, rep, img, layer, nrn_cls).
    """
    results_list = []
    mask_widths = []

    for level in occlusion_levels:
        # Calculate mask width: covers from x=0 to bump_start + (level * bump_width)
        mask_width = int(bump_start + (bump_width * level)) if level > 0 else 0
        mask_widths.append(mask_width)

        logger.info(
            f"Occlusion level {level * 100:.0f}% "
            f"(mask width: {mask_width}px, feature coverage: {level * 100:.0f}%)"
        )

        # Prepare config with occludable encoder
        cfg = _prepare_occludable_config(base_cfg, mask_width)

        # Run inference using existing infrastructure
        records = tasks.run_inference(
            cfg,
            imageset,
            store_path=store_path,
            duration=duration,
            duration_relax=0.0,
            reps=reps,
        )

        results_list.append(records)

    # Combine across occlusion levels
    results = xr.concat(results_list, dim="occlusion")
    results = results.assign_coords(occlusion=list(occlusion_levels))

    # Add metadata as attributes
    results.attrs.update(
        {
            "bump_start": bump_start,
            "bump_width": bump_width,
            "mask_widths": mask_widths,
            "duration": duration,
            "reps": reps,
        }
    )

    return results


def main(opt: Namespace):
    logger.info(opt)

    # Handle "None" string for save_layer and save_nrn_cls to disable filtering
    if opt.save_layer == ["None"] or opt.save_layer == [None]:
        opt.save_layer = None
    if opt.save_nrn_cls == ["None"] or opt.save_nrn_cls == [None]:
        opt.save_nrn_cls = None

    # === Setup === #
    expt = handler.ExperimentHandler(Path.cwd() / opt.expt_dir)
    trial = expt[opt.trial]
    cfg = trial.config

    # Get artifact store and checkpoint path
    store = handler.ArtifactStore(trial, opt.chkpt)
    store_path = store.checkpoint.store_path if store.checkpoint is not None else None
    logger.info(f"Model store path: {store_path}")

    # Load imageset
    imageset, _ = io.get_dataset(cfg["training"]["data"], return_annotations=True)
    logger.info(f"Loaded {len(imageset)} images")

    # Prepare results path using handler.get_results_path for consistency
    results_path = handler.get_results_path(trial, opt.chkpt, subdir=opt.subdir)
    logger.info(f"Results will be saved to: '{results_path}'")

    if opt.no_action:
        logger.info("Exiting with no action (-n flag used)")
        return

    # === Run inference === #

    if results_path.exists():
        if opt.force:
            os.remove(results_path)
            logger.info(f"Deleted existing file: '{results_path}'")
        else:
            raise FileExistsError(
                f"Results '{results_path}' already exist. Use -f to overwrite."
            )

    # === Run experiment === #
    t_start = time()
    results = run_occlusion_sweep(
        base_cfg=cfg,
        imageset=imageset,
        store_path=store_path,
        bump_start=opt.bump_start,
        bump_width=opt.bump_width,
        occlusion_levels=opt.occlusion_levels,
        duration=opt.duration,
        reps=opt.reps,
        num_workers=opt.num_workers,
    )
    t_elapsed = time() - t_start
    logger.info(f"Experiment completed in {t_elapsed:.1f}s")

    # === Filter and save results === #
    # Optionally filter by layer and neuron class to reduce file size
    if opt.save_layer is not None or opt.save_nrn_cls is not None:
        sel_kwargs = {}
        if opt.save_layer is not None:
            sel_kwargs["layer"] = opt.save_layer
        if opt.save_nrn_cls is not None:
            sel_kwargs["nrn_cls"] = opt.save_nrn_cls
        results = results.sel(**sel_kwargs)
        logger.info(f"Filtered results: {sel_kwargs}")

    io.save_pickle(results, results_path, parents=True)
    if results_path.exists():
        logger.info(f"Results saved to: '{results_path}'")
    else:
        raise FileNotFoundError(f"Failed to write results: '{results_path}'")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Run occlusion sensitivity analysis on a trained model."
    )
    parser.add_argument(
        "expt_dir", type=str, help="Experiment directory (relative to pwd)"
    )
    parser.add_argument("trial", type=int, help="Trial index.")
    parser.add_argument("--chkpt", type=int, default=-1, help="Checkpoint index")
    parser.add_argument(
        "--subdir",
        type=str,
        default="occlusion",
        help="Subdirectory for storing results (default: 'occlusion')",
    )
    parser.add_argument(
        "--bump_start",
        type=int,
        default=10,
        help="Pixel column where the convex feature starts",
    )
    parser.add_argument(
        "--bump_width",
        type=int,
        default=21,
        help="Width of the convex feature in pixels",
    )
    parser.add_argument(
        "--occlusion_levels",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="Fraction of feature to occlude (0.0 to 1.0)",
    )
    parser.add_argument(
        "--duration", type=float, default=250.0, help="Presentation duration in ms"
    )
    parser.add_argument(
        "--reps", type=int, default=20, help="Number of repetitions per occlusion level"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of workers for parallel inference",
    )
    parser.add_argument(
        "--save_layer",
        type=int,
        nargs="+",
        default=[4],
        choices=[1, 2, 3, 4],
        help="Layer(s) to save in results (default: [4]). Use --save_layer None to save all.",
    )
    parser.add_argument(
        "--save_nrn_cls",
        type=str,
        nargs="+",
        default=["EXC"],
        choices=["EXC", "INH"],
        help="Neuron class(es) to save in results (default: ['EXC']). Use --save_nrn_cls None to save all.",
    )
    parser.add_argument(
        "-f", "--force", action="store_true", help="Overwrite existing results"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "-n", "--no_action", action="store_true", help="Dry run without execution"
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    main(args)
