import gzip
import pickle
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from .data import ImageSet
from ..transforms import transform_registry, Compose

BASE_DIR = (Path(__file__).parents[3]).resolve()
DATA_DIR = BASE_DIR / 'data'
EXPT_DIR = BASE_DIR / 'experiments'


def as_dict(cfg: str | Path | Mapping) -> dict:
    if isinstance(cfg, (str, Path)):
        cfg_ = OmegaConf.to_container(OmegaConf.load(cfg))
    elif isinstance(cfg, DictConfig):
        cfg_ = OmegaConf.to_container(cfg)
    else:
        cfg_ = dict(cfg)
    assert isinstance(cfg_, dict)
    return cfg_


def as_dict_config(cfg: str | Path | Mapping) -> DictConfig:
    if isinstance(cfg, (str, Path)):
        cfg_ = DictConfig(OmegaConf.load(cfg))
    else:
        cfg_ = DictConfig(cfg)
    return cfg_


def get_local_config(name: str = 'config.yaml') -> dict:
    cfg_path = Path.cwd() / name
    return as_dict(cfg_path)


def get_dataset(data_cfg: Mapping, pattern: str = '*.png',
                return_annotations: bool = False) -> ImageSet | Tuple[ImageSet, pd.DataFrame]:
    dataset_dir: Path = DATA_DIR / data_cfg['name']
    tsf_ops = data_cfg.get('transforms', {})
    if len(tsf_ops):
        transform = Compose(*[transform_registry[k](*v) for k, v in tsf_ops.items()])
    else:
        transform = None
    imageset = ImageSet(dataset_dir, transform=transform, pattern=pattern)
    if return_annotations:
        annotations = pd.read_csv(dataset_dir / 'annotations.csv')
        return imageset, annotations
    else:
        return imageset


def save_pickle(data: Any, file_path: str | Path, parents: bool = False):
    """Saves data as a pickled object. If the file extension is `.gz`, it uses gzip compression.
    """
    file_path = Path(file_path)
    if parents:
        file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open('wb') as fp:
        if file_path.suffix == '.gz':
            with gzip.GzipFile(fileobj=fp, mode='wb') as gz_file:
                pickle.dump(data, gz_file)
        else:
            pickle.dump(data, fp)


def load_pickle(file_path: str | Path) -> Any:
    """Loads data from a pickled object. If the file extension is `.gz`, it uses gzip decompression.
    """
    file_path = Path(file_path)
    with file_path.open('rb') as fp:
        if file_path.suffix == '.gz':
            with gzip.GzipFile(fileobj=fp, mode='rb') as gz_file:
                return pickle.load(gz_file)
        else:
            return pickle.load(fp)


def formatted_name(base_name: str, ext: Optional[str] = None, **kwargs) -> str:
    """Generates a formatted name based on the base name and additional keyword arguments.

    - If the keyword argument is a boolean, it will be included in the name only if it's True.
    - If the keyword argument is a number, it will be included in the name only if it's greater than 0.
    - Otherwise, the keyword argument will be included in the name as is.

    Args:
        base_name (str): Base name for the file.
        ext (Optional[str], optional): File extension. Defaults to None.

    Returns:
        str: Formatted name based on the base name and additional keyword arguments.
    """
    name = base_name
    for k, v in kwargs.items():
        if isinstance(v, bool):
            if v:
                name += f'_{k}'
        elif isinstance(v, float):
            if v > 0.0:
                v_str = str(v).replace('.', '_')
                name += f'_{k}_{v_str}'
        elif isinstance(v, int):
            if v > 0:
                name += f'_{k}_{v}'
        else:
            name += f'_{k}_{v}'
    if ext is not None:
        name += ext if ext.startswith('.') else f'.{ext}'
    return name
