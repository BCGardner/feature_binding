from typing import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from ._types import RatesDatabase

__all__ = [
    "get_dataframe",
    "tabulate_selectivities"
]


def get_dataframe(rates_db: RatesDatabase) -> pd.DataFrame:
    labels = rates_db.rate.labels
    df = rates_db.to_dataframe().reset_index()
    df = df[['layer', 'nrn', 'img', 'rate', 'stdev', 'occurs'] + labels]
    return df.set_index(['layer', 'nrn']).sort_index()


def tabulate_selectivities(rates_db: RatesDatabase, by: str = 'rate',
                           drop_zero: bool = True) -> pd.DataFrame:
    data = get_dataframe(rates_db)
    return _get_selectivities(data, rates_db.rate.labels, by, drop_zero)


def rank_selectivities(selectivities: pd.DataFrame, layer: int, target: int) -> pd.DataFrame:
    valid_targets = {0, 1}
    if target not in valid_targets:
        raise ValueError(f"invalid target, valid choices: {valid_targets}")
    selectivities_ = selectivities.loc[layer]
    ranked_nrns = {}
    for col in selectivities_.columns.unique(0):
        vals = (selectivities_[col][0] - selectivities_[col][1]) * (-1)**target
        ranked_nrns[col] = vals.sort_values(ascending=False).index.unique('nrn')
    return pd.DataFrame(ranked_nrns)


def binned_population_activity(spike_times: npt.NDArray[np.float64], duration: float,
                               num_nrns: int, bin_size: int = 3):
    """Estimate the average firing activity per neuron across a population.
    """
    bins = np.arange(0, duration + bin_size, bin_size)
    population_activity, _ = np.histogram(spike_times, bins=bins)
    return population_activity / (bin_size * num_nrns) * 1E3


def average_layer_rates(rates_db: xr.Dataset, conditional_fired: bool = True) -> npt.NDArray[np.float64]:
    rates_xr = rates_db['rate']
    assert set(rates_xr.dims) == set(('img', 'nrn', 'layer'))
    if not conditional_fired:
        return np.asarray(rates_xr.mean(('img', 'nrn')), dtype=float)
    rates_xr = rates_xr.transpose('nrn', ...)
    ret = []
    for _, rates_xr in rates_xr.groupby('layer'):
        rates_nrn = [np.mean(rs[rs > 0]) for rs in rates_xr.values if np.any(rs > 0)]
        if len(rates_nrn):
            ret.append(np.mean(rates_nrn))
        else:
            ret.append(0)
    return np.asarray(ret, dtype=float)


def _get_selectivities(data: pd.DataFrame, labels: Sequence[str], by: str = 'rate',
                       drop_zero: bool = True) -> pd.DataFrame:
    dfs = []
    for label in labels:
        df_ = data.groupby(['layer', 'nrn', label])[by].mean()
        dfs.append(df_)
    df = pd.DataFrame(pd.concat(dfs, axis=1, keys=labels).unstack())
    if drop_zero:
        return df[df.any(axis=1)]
    else:
        return df
