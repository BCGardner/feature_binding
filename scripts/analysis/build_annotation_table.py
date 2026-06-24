#!/usr/bin/env python3
"""Builds and persists the canonical HFB annotation tables for a combination.

Re-analyses existing significance-tested detections and post-trained recordings
for a single representative trial, and writes three parquet tables under the trial
checkpoint's ``reuse`` sub-directory: ``hfb_annotations``, ``png_label_metrics`` and
``neuron_information`` (see ``hsnn.pipeline.reuse``). Downstream notebooks/tasks read
these back via ``reuse.load_tables(reuse.open_store(...))`` rather than rebuilding.

Re-running is idempotent: a trial whose three tables already exist is skipped (no
re-analysis) unless ``--force`` is given or one or more tables are missing.

Examples:
# Representative trial analysis (N4P2, FF+LAT+FB)
./scripts/analysis/build_annotation_table.py ./experiments/n4p2/train_n4p2_lrate_0_02_181023 ALL -v

# Every detection trial for a combination (needed for the cross-trial figure notebooks)
./scripts/analysis/build_annotation_table.py ./experiments/n4p2/train_n4p2_lrate_0_02_181023 ALL \
    --all-detection -v
"""

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

from hsnn.core.logger import get_logger
from hsnn.pipeline import reuse
from hsnn.utils import handler, io

logger = get_logger(__name__)


def _tables_present(store: handler.ArtifactStore, subdir: str) -> bool:
    """True if every reuse table parquet already exists under ``subdir``."""
    try:
        present = set(store.list_artifacts(subdir))
    except FileNotFoundError:
        return False
    return all(f"{name}.parquet" in present for name in reuse.TABLE_NAMES)


def _build_and_persist(experiment: str, opt: Namespace, trial: str | int | None) -> None:
    """Builds and persists the reuse tables for a single trial of a combination.

    Skips the (time-consuming) rebuild when every table already exists for the resolved
    trial/checkpoint, unless ``--force`` is given or one or more tables are missing.
    """
    store = reuse.open_store(experiment, opt.model_type, trial=trial, checkpoint=opt.chkpt)
    if not opt.force and _tables_present(store, opt.subdir):
        logger.info(
            f"Tables already present for trial '{store.trial.name}' under "
            f"'{store.logdir / opt.subdir}'; skipping (use -f to rebuild)."
        )
        return

    tables, inputs = reuse.build_tables(
        experiment, opt.model_type, trial=trial,
        checkpoint=opt.chkpt, duration=opt.duration, offset=opt.offset,
        num_reps=opt.num_reps,
    )
    logger.info(f"Trial: {inputs.trial.name}")
    for name, df in tables.items():
        logger.info(f"{name}: {df.shape}")

    reuse.persist_tables(inputs.store, tables, subdir=opt.subdir, overwrite=True)
    logger.info(f"Persisted tables to '{inputs.store.logdir / opt.subdir}'")


def main(opt: Namespace) -> None:
    logger.info(opt)
    experiment = str(Path(opt.expt_dir).resolve().relative_to(io.EXPT_DIR))

    if opt.all_detection:
        expt = handler.ExperimentHandler(experiment) # type: ignore
        trials = expt.metadata.get_trials_dict(opt.model_type)["detection"]
        logger.info(f"Building all {len(trials)} detection trials: {trials}")
        for trial in trials:
            _build_and_persist(experiment, opt, trial)
    else:
        _build_and_persist(experiment, opt, opt.trial)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    parser.add_argument("expt_dir", type=str, help="Path to experiment directory")
    parser.add_argument(
        "model_type", type=str, choices=["SEMI", "ALL"], help="Architecture type"
    )
    parser.add_argument(
        "--trial", type=int, default=None,
        help="Trial index, e.g. 0, 1, 3 (default: representative trial)",
    )
    parser.add_argument(
        "--all-detection", action="store_true",
        help="Build every detection trial for the combination (resolved from metadata); "
             "mutually exclusive with --trial",
    )
    parser.add_argument(
        "--chkpt", type=int, default=-1, help="Checkpoint index (default: -1, post)"
    )
    parser.add_argument(
        "--duration", type=float, default=reuse.DEFAULT_DURATION,
        help="Observation window (ms)",
    )
    parser.add_argument(
        "--offset", type=float, default=reuse.DEFAULT_OFFSET,
        help="Observation offset (ms)",
    )
    parser.add_argument(
        "--num_reps", type=int, default=None,
        help="Detection repetitions for occurrence metrics (default: infer from PNGs)",
    )
    parser.add_argument(
        "--subdir", type=str, default=reuse.REUSE_SUBDIR,
        help="Checkpoint sub-directory for the persisted tables",
    )
    parser.add_argument(
        "-f", "--force", action="store_true", help="Overwrite existing tables"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.all_detection and args.trial is not None:
        parser.error("--all-detection and --trial are mutually exclusive")

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    main(args)
