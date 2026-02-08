#!/usr/bin/env python3
from pathlib import Path

from omegaconf import OmegaConf

from hsnn.cluster.tuner import ExperimentRunner
from hsnn.utils import io


def main() -> None:
    model_cfg = io.get_local_config()
    expt_runner = ExperimentRunner(model_cfg, Path.cwd())
    results = expt_runner.run()
    OmegaConf.save(model_cfg, Path(results.experiment_path) / 'config.yaml')


if __name__ == '__main__':
    main()
