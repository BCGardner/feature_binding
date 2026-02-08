from typing import Optional

import pandas as pd
import xarray as xr

from ._types import RatesDatabase
from .base import _copy_attrs, attach_labels, infer_rates, infer_occurrences, get_averages_stdev

__all__ = ["create_rates_db"]


def create_rates_db(records: xr.DataArray, labels: Optional[pd.DataFrame] = None,
                    duration: Optional[float] = None, offset: float = 0,
                    conditional_fired: bool = False) -> RatesDatabase:
    """Creates a RatesDatabase (xr.Dataset) summarising network activity.

    Args:
        records (xr.DataArray): Spike recordings, including dim 'rep'. All dims must contain same number of neurons.
        labels (Optional[pd.DataFrame], optional): Data annotations. Defaults to None.
        duration (Optional[float], optional): Only considers spike times in [offset, offset + duration). Defaults to None.
        offset (float, optional): Disregards spike times before offset. Defaults to 0.
        conditional_fired (bool, optional): Averaging over non-empty spike trains. Defaults to False.

    Returns:
        RatesDatabase: Dataset containing DataArrays: ('rates', 'stdevs', 'occurs').
    """
    if 'nrn_cls' in records.dims:
        raise AttributeError(f"select one dim for 'nrn_cls': {records['nrn_cls'].values}")
    rates = infer_rates(records, duration, offset)
    if labels is not None:
        rates = attach_labels(rates, labels)
    occurrences = infer_occurrences(rates)
    rates_av, stdevs = get_averages_stdev(rates, conditional_fired)
    description: str = rates.description
    if conditional_fired:
        description += ' (conditional on having fired)'
    attrs = _copy_attrs(rates, description=description, reps=len(rates.rep))
    rates_av_ = xr.DataArray(
        rates_av,
        name='rates',
        dims=occurrences.dims,  # Has same dims as rates dim minus 'rep'
        coords=occurrences.coords,
        attrs=attrs
    )
    stdevs_ = xr.DataArray(
        stdevs,
        name='stdevs',
        dims=occurrences.dims,
        coords=occurrences.coords,
        attrs=attrs
    )
    return xr.Dataset(
        {'rate': rates_av_, 'stdev': stdevs_, 'occurs': occurrences},
        attrs=attrs
    )
