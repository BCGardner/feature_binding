from pathlib import Path

from omegaconf import DictConfig, OmegaConf

SAMPLES_DIR = Path(__file__).parents[1] / 'samples'
config = DictConfig(OmegaConf.load(SAMPLES_DIR / 'data_ops.yaml'))


def get_data(name: str) -> list:
    arguments = OmegaConf.to_container(config[name])
    assert isinstance(arguments, dict)
    return list(zip(arguments['test_input'], arguments['expected']))
