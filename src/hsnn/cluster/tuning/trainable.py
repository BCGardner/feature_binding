from typing import Any, Dict, Optional, Sequence

import numpy as np

from .base import BaseTrainable, set_nested
from .scalars import scalar_registry, BaseScalar, StateMixin
from .criterion import criterion_registry, BaseCriterion
from ._utils import parse_scale_factor

__all__ = ["TrainSNN"]


class TrainSNN(BaseTrainable):
    def setup(self, config: Dict, data: Sequence[np.ndarray], labels: np.ndarray):  # type: ignore[override]
        self._update_config(config)
        super().setup(config, data, labels)
        self._setup_criterion(config['tuning'])
        self._setup_scalars(config['tuning'])

    def step(self):
        report: dict[str, Any] = {key: None for key in ['loss'] + list(self.scalars.keys())}
        if self.iteration == 0 and self._should_stop():
            report['loss'] = self.stop_threshold
            return report
        for _ in range(self.num_epochs_iter):
            if self.shuffle:
                indices = np.random.choice(len(self.data), len(self.data), replace=False)
                data_ = [self.data[idx] for idx in indices]
            else:
                data_ = self.data
            self.sim.present(data_)
        if self._should_stop():
            report['loss'] = self.stop_threshold
            return report
        report['loss'] = self.criterion.loss(self.sim, self.data, self.labels)
        metrics = {key: scalar(self.sim, self.data, self.labels)
                   for key, scalar in self.scalars.items()}
        report.update(**metrics)
        return report

    def _update_config(self, config: Dict) -> None:
        """Optional: updates config parameters from "tuning.scale_factors".
        """
        tune_cfg: dict = config['tuning']
        scale_factors: dict = tune_cfg.get('scale_factors', {})
        for dst_path, expr in scale_factors.items():
            val = parse_scale_factor(expr, config)
            set_nested(val, dst_path, config)

    def _setup_criterion(self, tune_cfg: dict):
        """Setup criterion measure for search algorithm and optionally stop condition.
        """
        crit_cfg: dict = tune_cfg.get('criterion', {'name': 'fixed', 'args': []})
        self.criterion: BaseCriterion = criterion_registry[crit_cfg['name']](*crit_cfg['args'])
        self.stop_threshold: Optional[float] = None
        self.stop_criterion: Optional[BaseCriterion] = None
        if 'stop' in tune_cfg:
            stop_cfg: dict = tune_cfg['stop']
            crit_cfg = stop_cfg.get('criterion', None)  # type: ignore[assignment]
            self.stop_threshold = stop_cfg['threshold']
            if crit_cfg is not None:
                self.stop_criterion = criterion_registry[crit_cfg['name']](*crit_cfg['args'])

    def _setup_scalars(self, tune_cfg: dict):
        """Setup scalars reported in addition to criterion loss per training iteration.
        """
        self.scalars: dict[str, BaseScalar] = {}
        scalar_configs: list[dict[str, Any]] = tune_cfg.get('scalars', [])
        for cfg in scalar_configs:
            scalar = scalar_registry[cfg['name']](*cfg['args'])
            if isinstance(scalar, StateMixin):
                scalar.build(self.sim)
            self.scalars[cfg['name']] = scalar

    def _should_stop(self) -> bool:
        """If stop condition and associated stop criterion present, then check if training should be halted.
        """
        if self.stop_criterion is not None and self.stop_threshold is not None:
            return self.stop_criterion.loss(self.sim, self.data, self.labels) >= self.stop_threshold
        else:
            return False


# class TuneSNN(BaseTrainable):
#     def setup(self, config: Dict, data: Sequence[np.ndarray], labels: np.ndarray):  # type: ignore[override]
#         super().setup(config, data, labels)
#         tune_cfg: dict = config['tuning']
#         subset: Iterable = tune_cfg.get('subset', range(len(data)))
#         self.data = [data[idx] for idx in subset]
#         self.labels = np.asarray([labels[idx] for idx in subset])
