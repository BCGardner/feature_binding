from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from omegaconf import ListConfig
from ray.tune.search import BasicVariantGenerator
from ray.tune.search.bayesopt import BayesOptSearch

from hsnn.utils import io
from .tuning import traverse_dict


class SearchAlg(ABC):
    def __init__(self, config: Mapping) -> None:
        self._config = io.as_dict(config)
        tuning_cfg: dict = config['tuning']
        self._hparams: dict = tuning_cfg.get('hyper_params', {})

    @abstractmethod
    def create(self, **kwargs) -> Any:
        ...


class DefaultSearcher(SearchAlg):
    def create(self, **kwargs) -> Any:
        return BasicVariantGenerator(**kwargs)


class BayesOpt(SearchAlg):
    def create(self, **kwargs) -> Any:
        kwargs_ = self._parse_kwargs(**kwargs)
        return BayesOptSearch(**kwargs_)

    def _parse_kwargs(self, **kwargs) -> dict[str, Any]:
        def _get_cfg(path: str | Path) -> dict:
            point = io.as_dict(path)
            return self._update_hparams(point)

        dst = {}
        for k, v in kwargs.items():
            if k == 'points_to_evaluate':
                if isinstance(v, (list, ListConfig)):
                    paths = [Path(path).expanduser() for path in v]
                else:
                    path = Path(v).expanduser()
                    paths = sorted(path.glob('*.yaml')) if path.is_dir() else [path]
                dst[k] = [_get_cfg(p) for p in paths]
            else:
                dst[k] = v
        return dst

    def _update_hparams(self, point: dict) -> dict:
        config = deepcopy(self._config)
        for hparam in self._hparams.keys():
            keys = hparam.split('.')
            ptr = traverse_dict(config, keys[:-1])
            ptr[keys[-1]] = traverse_dict(point, keys)
        return config
