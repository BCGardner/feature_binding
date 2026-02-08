#!/usr/bin/env python3
"""Run Snakemake over one or more trials for an experiment[, checkpoint].

Example:
    ./scripts/run_main_workflow.py path/to/expt_dir 0 1 --chkpt -1 -v
"""

import argparse
import logging
import subprocess

from hsnn.core.logger import get_logger

logger = get_logger(__name__)


def main(opt: argparse.Namespace):
    logger.info(opt)
    base_cmd = ["snakemake --cores all"]
    if opt.configfile is not None:
        base_cmd += [f"--configfile {opt.configfile}"]
    if opt.rule is not None:
        base_cmd += [opt.rule]
    base_cmd += [
        "--config",
        f"expt_dir={opt.expt_dir}",
        f"delay_jitter={opt.delay_jitter}",
        f"noise={opt.noise}",
        f"verbose={opt.verbose}",
        f"no_action={opt.no_action}"
    ]
    if opt.layers is not None:
        base_cmd += [f"layers={opt.layers}"]
    if opt.chkpt is not None:
        base_cmd += [f"chkpt={opt.chkpt}"]
    if opt.subdir is not None:
        base_cmd += [f"subdir={opt.subdir}"]

    for val in opt.trials:
        cmd = ' '.join(base_cmd + [f"trial={val}"])
        logger.info(f"Shell command: '{cmd}'")
        subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Snakemake over a sequence of Trials")
    parser.add_argument('expt_dir', type=str, help="Experiment directory (relative to pwd)")
    parser.add_argument("trials", type=int, nargs='+', help="Trial indices")
    parser.add_argument('--layers', type=int, nargs='*', default=None,
                        help="Detection layers (PNGs).")
    parser.add_argument('--configfile', type=str, default=None,
                        help="Path to the Snakemake config file.")
    parser.add_argument('--chkpt', type=int, default=None, help="Checkpoint index")
    parser.add_argument('--subdir', type=str, default=None,
                        help="Name of subdirectory to store results inside.")
    parser.add_argument('--rule', type=str, default=None, help="Snakemake rule")
    parser.add_argument('--delay_jitter', type=float, default=0.0,
                        help="Apply multiplicative jitter to exc. delays with this sigma value.")
    parser.add_argument('--noise', type=int, default=0)
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    parser.add_argument('-n', '--no_action', action='store_true', default=False)
    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    main(args)
