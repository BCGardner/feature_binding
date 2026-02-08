from __future__ import annotations

from copy import copy
from typing import Any, Hashable, Iterable, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from pandas.core.groupby.generic import DataFrameGroupBy

from .base import attach_labels, get_midx, get_proj_layer_mapping
from .ratesdb import create_rates_db
from . import activity
from hsnn.core.types import SpikePatterns
from hsnn.core.logger import get_logger
from hsnn import ops

__all__ = ["ResultsDatabase"]

_DIMS = {'rep', 'img', 'layer', 'nrn_cls'}

pidx = pd.IndexSlice
logger = get_logger(__name__)


class ResultsDatabase:
    """Database of network spike recordings to refine and extract (layer, nrn) IDs for further analysis.
    """
    def __init__(self, records: xr.DataArray, syn_params: pd.DataFrame,
                 labels: pd.DataFrame, layer: slice = slice(None),
                 proj: tuple[str, ...] = ('FF',), duration: Optional[float] = None,
                 offset: float = 0.0, separation: float = 50.0) -> None:
        """Initialises the database from existing spike recordings (e.g. obtained from inference runs).

        Args:
            records (xr.DataArray): Spike recordings, with dims ('rep', 'img', 'layer', 'nrn_cls').
            syn_params (pd.DataFrame): Synaptic parameters, with index (layer, proj, pre, post); columns (w, delay).
            labels (pd.DataFrame): Image annotations, with columns ('image_id', feature1, feature2, ...).
            layer (slice, optional): Range of layers selected. Defaults to slice(None).
            proj (tuple[str], optional): Collection of synaptic projections selected. Defaults to ('FF',).
            duration (Optional[float], optional): Only considers spike times in [offset, offset + duration). Defaults to None.
            offset (float, optional): Disregards spike times before offset. Defaults to 0.0.
            separation (float, optional): Padding to apply between concatenated spike trains. Defaults to 50.0.
        """
        assert set(records.dims) == _DIMS, f"records missing dims: {_DIMS.difference(records.dims)}"
        self._records = attach_labels(
            records.sel(layer=layer, nrn_cls='EXC').transpose('layer', 'img', 'rep'), labels
        )
        self._syn_params = syn_params.loc[pidx[layer, proj], :].sort_index(inplace=False)
        self._labels = labels
        self._duration = records.item(0).duration if duration is None else duration
        self._offset = offset
        self._separation = separation
        self._data: pd.DataFrame
        self.reset()

    def _clone(self, data: Optional[pd.DataFrame] = None) -> ResultsDatabase:
        rdb = copy(self)
        rdb._data = self._data.copy() if data is None else data.copy()
        return rdb

    def __repr__(self) -> str:
        return repr(self._data)

    @property
    def data(self) -> pd.DataFrame:
        return self._data.copy()

    @property
    def labels(self) -> pd.DataFrame:
        return self._labels.copy()

    @property
    def index(self) -> pd.Index:
        return self._data.index.unique()

    @property
    def dims(self) -> tuple[Hashable, ...]:
        return self._records.dims

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def offset(self) -> float:
        return self._offset

    @property
    def separation(self) -> float:
        return self._separation

    @property
    def num_reps(self) -> int:
        return len(self._records['rep'])

    @property
    def syn_params(self) -> pd.DataFrame:
        return self._syn_params

    def reset(self) -> None:
        dfs = []
        for _, subset in self._records.groupby('layer'):
            dfs.append(self._get_activities(subset))
        self._data = pd.concat(dfs)

    def get_coord_values(self, coord: Hashable) -> npt.NDArray:
        return self._records[coord].values

    def get_max_duration(self) -> float:
        return (len(self._data['img'].unique()) * len(self._records.coords['rep'])) \
            * (self._duration + self._separation)

    def get_patterns(self) -> SpikePatterns:
        patterns: SpikePatterns = {}
        imgs = self._data['img'].unique()
        for layer in self._data.index.unique('layer'):
            nrns = self._get_nrn_values(layer)
            records_ = self._records.sel(layer=layer, img=imgs)
            patterns[np.int_(layer)] = \
                ops.concatenate_spike_trains(records_, nrns, self._duration,
                                             self._offset, self._separation)
        return patterns

    def get_selectivities(self, by: str = 'rate', drop_zero: bool = True) -> pd.DataFrame:
        return activity._get_selectivities(self._data, self._records.attrs['labels'], by, drop_zero)

    def drop_under(self, key: str, lb: float) -> ResultsDatabase:
        return self._clone(self._groupby_idx().filter(lambda group: group[key].max() >= lb))

    def drop_zero(self, key: str) -> ResultsDatabase:
        return self._clone(self._groupby_idx().filter(lambda group: group[key].max() > 0))

    def filter_equal(self, key: str, value: Any) -> ResultsDatabase:
        return self._clone(self._groupby_idx().filter(lambda group: (group[key] == value).any()))

    def filter_connected(self, post: int, layer: int, w_min: float = 0.5) -> ResultsDatabase:
        # try:
        #     self._data.xs((layer, post), level=('layer', 'nrn'))
        # except KeyError:
        #     logger.error(f"aborted operation: missing layer or post nrn")
        #     return self
        syn_pre = self._syn_params.xs((layer, post), level=('layer', 'post'))
        index_pre = syn_pre[syn_pre['w'] >= w_min].index
        layers = index_pre.get_level_values('proj').map(get_proj_layer_mapping(layer))
        index_pre = get_midx(layers, index_pre.get_level_values('pre'))
        index_post = get_midx(layer, post)
        midx = index_pre.append(index_post).sort_values()
        return self.filter_indices(midx)

    def filter_targets(self, **targets) -> ResultsDatabase:
        """Filters to entries matching the given column-value target conditions.

        Args:
            **targets: Key-value pairs corresponding to column names and target values.

        Returns:
            ResultsDatabase: Returns a copy of the filtered database.
        """
        mask = np.ones(len(self._data), dtype=np.bool_)
        for label, value in targets.items():
            mask = mask & (self._data[label] == value)
        return self._clone(self._data[mask])

    def filter_images(self, values: Iterable) -> ResultsDatabase:
        """Filters to entries corresponding to the given image ID `values`.

        Args:
            values (Iterable): Image IDs to filter by.

        Returns:
            ResultsDatabase: Returns a copy of the filtered database.
        """
        mask = self._data['img'].isin(values)
        return self._clone(self._data[mask])

    def filter_indices(self, midx: pd.MultiIndex) -> ResultsDatabase:
        """Filters to entries corresponding to the given `(layer, nrn)` indices.

        Args:
            midx (pd.MultiIndex): Index containing tuples of `(layer, nrn)` to filter by.

        Returns:
            ResultsDatabase: Returns a copy of the filtered database.
        """
        mask = self._data.index.isin(midx)
        return self._clone(self._data[mask])

    def _get_activities(self, subset: xr.DataArray) -> pd.DataFrame:
        rates_db = create_rates_db(
            subset, labels=self._labels, duration=self._duration,
            offset=self._offset
        )
        return activity.get_dataframe(rates_db)

    def _groupby_idx(self) -> DataFrameGroupBy:
        return self._data.groupby(['layer', 'nrn'])

    def _subset_dim(self, dim: Any, indexer: Optional[Any] = None) -> np.ndarray:
        xarr = self._records[dim]
        return xarr.values if indexer is None else xarr.sel(**{dim: indexer}).values

    def _get_nrn_values(self, layer: int) -> pd.Index:
        return self.index.get_level_values('nrn')[self.index.get_level_values('layer') == layer]
