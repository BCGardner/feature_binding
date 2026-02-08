from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence, Type

import numpy as np
import numpy.typing as npt
from scipy.stats import entropy

import hsnn.simulation.functional as F
from hsnn import analysis
from hsnn.analysis import activity, measures
from hsnn.analysis._types import RatesArray, RatesDatabase
from hsnn.simulation import Simulator
from ._utils import get_rates_db

__all__ = ["criterion_registry"]

criterion_registry: dict[str, Type[BaseCriterion]] = {}


class BaseCriterion(ABC):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = cls.__name__.lower()
        criterion_registry[name.removesuffix('criterion')] = cls

    def __init__(self, *args) -> None:
        return None

    @abstractmethod
    def loss(self, simulator: Simulator, data: Sequence[np.ndarray],
             labels: Optional[np.ndarray] = None) -> float:
        ...


class FixedCriterion(BaseCriterion):
    def loss(self, simulator: Simulator, data: Sequence[np.ndarray],
             labels: Optional[np.ndarray] = None) -> float:
        return 0.0


class RatesRangeCriterion(BaseCriterion):
    def __init__(self, max_loss: float = 1.0, lb: float = 0, ub: float = 300,
                 duration: float = 150, duration_relax: float = 0, offset: float = 50,
                 reps: int = 1, single_sample: bool = True, conditional_fired: bool = True) -> None:
        self.max_loss = max_loss
        self.lb = lb
        self.ub = ub
        self.duration = duration
        self.duration_relax = duration_relax
        self.offset = offset
        self.reps = reps
        self.single_sample = single_sample
        self.conditional_fired = conditional_fired

    def loss(self, simulator: Simulator, data: Sequence[np.ndarray],
             labels: Optional[np.ndarray] = None) -> float:
        samples = [data[0]] if self.single_sample else data
        rates_db = get_rates_db(simulator.network, samples, self.duration,
                                self.duration_relax, self.offset, self.reps,
                                self.conditional_fired)
        if self._inbounds(rates_db):
            return 0.0
        else:
            return self.max_loss

    def _inbounds(self, rates_db: RatesDatabase) -> bool:
        for _, group in rates_db['rate'].groupby('layer'):
            rs = group.values
            if not len(rs[rs > self.lb]) or len(rs[rs > self.ub]):
                return False
        return True


class RatesCriterion(BaseCriterion):
    def __init__(self, target: npt.ArrayLike, duration: float = 150,
                 duration_relax: float = 0, offset: float = 50, reps: int = 1,
                 conditional_fired: bool = True) -> None:
        self.target = np.asarray(target, dtype=np.float_)
        self.duration = duration
        self.duration_relax = duration_relax
        self.offset = offset
        self.reps = reps
        self.conditional_fired = conditional_fired

    def loss(self, simulator: Simulator, data: Sequence[np.ndarray],
             labels: Optional[np.ndarray] = None) -> float:
        rates_db = get_rates_db(simulator.network, data, self.duration,
                                self.duration_relax, self.offset, self.reps,
                                self.conditional_fired)
        rates = activity.average_layer_rates(rates_db, self.conditional_fired)
        return float(np.linalg.norm(rates - self.target))


class MutualMeasureCriterion(BaseCriterion):
    def __init__(self, layer: int = 4, selectivity: str = 'whole',
                 top_n: Optional[int] = None, reps: int = 10) -> None:
        _choices = {'whole', 'contour'}
        if selectivity not in _choices:
            raise ValueError(f"selectivity must be one of {_choices}")
        self.layer = layer
        self.selectivity = selectivity
        self.top_n = top_n
        self.reps = reps

    def loss(self, simulator: Simulator, data: Sequence[np.ndarray],
             labels: Optional[np.ndarray] = None) -> float:
        rates_array = self._infer_rates(simulator, data)
        if self.selectivity == 'whole':
            features = np.array([rates_array['img'].values])
        elif labels is not None and self.selectivity == 'contour':
            features = labels.T
        else:
            raise ValueError(f"invalid labels: {labels}")

        metrics = []
        for targets in features:
            unique_targets, weights = np.unique(targets, return_counts=True)
            specific_measures = measures.get_specific_measures(rates_array, targets)
            mutual_measures = np.average(specific_measures, axis=1, weights=weights)
            if self.top_n is not None:
                mutual_measures = np.sort(mutual_measures)[::-1][:self.top_n]
            metric = np.log2(len(unique_targets)) - np.mean(mutual_measures)
            metrics.append(metric)
        return float(np.mean(metrics))

    def _infer_rates(self, simulator: Simulator, data: Sequence[np.ndarray]) -> RatesArray:
        records = simulator.infer_reps(data, reps=self.reps)
        return analysis.infer_rates(records.sel(layer=self.layer, nrn_cls='EXC'))


class SpecificMeasureCriterion(BaseCriterion):
    """Measures single-cell specific information conveyed about each stimulus
    category, I(s, R), and takes the average across all categories w.r.t. the top N most informative
    neurons. This is substracted from the maximum information gain possible, the stimulus entropy H(S).
    """
    def __init__(self, layer: int = 4, topN: int = 100, duration: float = 250,
                 duration_relax: float = 0, offset: float = 50, bin_width: Optional[int] = None,
                 target: int = 1, reps: int = 20) -> None:
        self.layer = layer
        self.topN = topN
        self.duration = duration
        self.duration_relax = duration_relax
        self.offset = offset
        self.bin_width = bin_width
        self.target = target
        self.reps = reps

    def loss(self, simulator: Simulator, data: Sequence[np.ndarray],
             labels: Optional[np.ndarray] = None) -> float:
        assert labels is not None
        features = labels.T
        rates_array = self._infer_rates(simulator, data)

        losses = []
        for targets in features:
            specific_measures = measures.get_specific_measures(rates_array, targets, self.bin_width)
            ranked_measures = specific_measures.apply(lambda x: x.sort_values(ascending=False).values)[:self.topN]
            mean_measures = np.mean(ranked_measures) if self.target is None else np.mean(ranked_measures[self.target])
            losses.append(self._max_info(targets) - mean_measures)
        return np.mean(losses).item()

    def _infer_rates(self, simulator: Simulator, data: Sequence[np.ndarray]) -> RatesArray:
        records = F.infer_batch_reps(simulator.network, data, self.duration,
                                     self.duration_relax, reps=self.reps)
        return analysis.infer_rates(records.sel(layer=self.layer, nrn_cls='EXC'),
                                    self.duration - self.offset, self.offset)

    def _max_info(self, targets: np.ndarray) -> float:
        _, weights = np.unique(targets, return_counts=True)
        pk = weights / np.sum(weights)
        return entropy(pk, base=2).item()
