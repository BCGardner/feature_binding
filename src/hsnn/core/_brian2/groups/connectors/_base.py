from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Type

import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree
from brian2 import NeuronGroup, Synapses

from ....logger import get_logger

__all__ = ["BaseConnector", "connector_registry"]

logger = get_logger(__name__)

connector_registry: Dict[str, Type[BaseConnector]] = {}


class BaseConnector(ABC):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = cls.__name__.lower()
        connector_registry[name.removesuffix('connector')] = cls

    def __init__(self) -> None:
        self._kdtrees: Dict[str, KDTree] = {}

    @abstractmethod
    def sample(self, points: npt.NDArray[np.float_], point_ref: npt.NDArray[np.float_],
               **kwargs) -> npt.NDArray[np.int_]:
        """Returns indices of afferent points which are sampled according to a distribution.

        Args:
            points (npt.NDArray[np.float_]): Source spatial coordinates, of shape (X, Y).
            point_ref (npt.NDArray[np.float_]): Target spatial coordinates, of shape (X, Y).

        Returns:
            npt.NDArray[np.int_]: Indices of selected source points.
        """
        ...

    @abstractmethod
    def connect(self, synapses: Synapses, **kwargs) -> None:
        ...

    def get_kdtree(self, group: NeuronGroup) -> KDTree:
        name: str = group.name
        if name not in self._kdtrees:
            self.add_kdtree(group)
        return self._kdtrees[name]

    def add_kdtree(self, group: NeuronGroup) -> None:
        name: str = group.name
        if name in self._kdtrees:
            logger.warning(f"overwriting KDTree stored for group '{name}'...")
        self._kdtrees[name] = self._create_kdtree(group.x[:], group.y[:])

    def _create_kdtree(self, xs: npt.NDArray[np.float_], ys: npt.NDArray[np.float_]) -> KDTree:
        points = np.vstack((xs, ys), dtype=np.float_).T
        return KDTree(points) # type: ignore

    def _get_indices(self, source: NeuronGroup, target: NeuronGroup, max_distance: float,
                     sort_indices: bool = True, self_conns: bool = False, **sample_kwargs) -> Tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
        kdtree_source = self.get_kdtree(source)
        kdtree_target = self.get_kdtree(target)
        indices = kdtree_target.query_ball_tree(kdtree_source, max_distance)
        skip_equal = True if source.name == target.name and not self_conns else False
        pre_ids: List[int] = []
        post_ids: List[int] = []
        for j, idxs in enumerate(indices):
            idxs_ = np.array(idxs)
            idxs_ = idxs_[idxs_ != j] if skip_equal else idxs_
            args = self.sample(kdtree_source.data[idxs_], kdtree_target.data[j], **sample_kwargs)
            connections = idxs_[args]
            pre_ids.extend(connections)
            post_ids.extend([j] * len(connections))
        pre_ids_ = np.array(pre_ids, dtype=np.int_)
        post_ids_ = np.array(post_ids, dtype=np.int_)
        if sort_indices:
            idxs = np.lexsort((post_ids, pre_ids))
            return pre_ids_[idxs], post_ids_[idxs]
        else:
            return pre_ids_, post_ids_
