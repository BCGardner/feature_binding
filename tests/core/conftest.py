import pytest
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from hsnn.core.config import paramset, ModelParams


SAMPLES_DIR = Path(__file__).parents[1] / 'samples'


@pytest.fixture(scope='module')
def config_path() -> str:
    return str((SAMPLES_DIR / 'config.yaml').resolve())


@pytest.fixture(scope='module')
def model_cfg(config_path) -> DictConfig:
    dst = OmegaConf.load(config_path)
    if isinstance(dst, DictConfig):
        return dst
    return DictConfig({})


@pytest.fixture(scope='module')
def model_params(model_cfg) -> ModelParams:
    return paramset.ModelParams(model_cfg)
