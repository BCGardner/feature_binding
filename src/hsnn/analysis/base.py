from typing import Any, Iterable, Optional, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from hsnn import ops
from ._types import RatesArray, OccurrencesArray

__all__ = ["attach_labels", "infer_rates", "infer_occurrences"]


def _copy_attrs(src: xr.DataArray | xr.Dataset, **kwargs) -> dict:
    attrs = src.attrs.copy()
    attrs.update(**kwargs)
    return attrs


def attach_labels(src: xr.DataArray, labels: pd.DataFrame) -> xr.DataArray:
    """Attaches dataset annotation labels.

    Args:
        src (xr.DataArray): DataArray to attach image labels.
        labels (pd.DataFrame): Sequence of label arrays.

    Returns:
        xr.DataArray: Copy of the original DataArray with updated label attributes.
    """
    labels = labels.drop(columns='image_id')
    return src.assign_coords(
        coords={col: ('img', labels[col].values) for col in labels.columns},
    ).assign_attrs(labels=labels.columns.to_list())


def infer_rates(spike_records: xr.DataArray, duration: Optional[float] = None,
                offset: float = 0, description: str = 'Firing rates') -> RatesArray:
    """Infers firing rates from a collection of SpikeRecords.

    Args:
        spike_records (xr.DataArray): Assuming shared number of neurons across dims.
        duration (Optional[float], optional): Observation period relative to offset. Defaults to None.
        offset (float, optional): Applies an offset to the observation period (to account for spike onset latency). Defaults to 0.
        description (str, optional): Defaults to 'Firing rates'.

    Returns:
        RatesArray: Firing rates in hertz w.r.t. individual neuron observations.
    """
    duration_: float = spike_records.item(0).duration if duration is None else duration
    num_nrns: int = spike_records.item(0).num_nrns
    data = np.full(spike_records.shape + (num_nrns,), np.nan)
    for coord, record in np.ndenumerate(spike_records):
        rates = ops.mask_rates(record.spike_events, duration_, offset, record.num_nrns)
        data[coord] = rates[1]

    attrs = _copy_attrs(spike_records, unit='hertz', description=description,
                        duration=duration_, offset=offset)
    return xr.DataArray(
        data,
        dims=spike_records.dims + ('nrn',),
        coords=spike_records.coords,
        name='rates',
        attrs=attrs
    ).assign_coords({'nrn': np.arange(num_nrns)})


def infer_occurrences(rates: RatesArray, description: str = 'Response rates') -> OccurrencesArray:
    """Gets the fraction of runs where a neuron responded with at least one spike.

    Args:
        rates (RatesArray): Firing rates, including dim 'rep'.
        description (str, optional): Defaults to 'Response rates'.

    Returns:
        OccurrencesArray: Fraction of repetitions with a neuronal response.
    """
    rates_ = rates.transpose(..., 'rep')
    data = (rates_.values > 0).mean(-1)
    attrs = _copy_attrs(rates, unit=None, description=description,
                        reps=len(rates.rep))
    return xr.DataArray(
        data,
        name='occurrences',
        dims=rates_.dims[:-1],
        coords=rates_.drop_vars('rep').coords,
        attrs=attrs
    )


def get_midx(layer: Any, nrn: Any) -> pd.MultiIndex:
    if isinstance(nrn, Iterable):
        nrn = list(nrn)
    if isinstance(layer, Iterable):
        layer = list(layer)
    arrays = np.broadcast_arrays(np.atleast_1d(layer), np.atleast_1d(nrn))
    return pd.MultiIndex.from_arrays(arrays, names=['layer', 'nrn']).sort_values()


def get_proj_layer_mapping(layer: int, projs: Optional[Sequence[str]] = None) -> dict[str, int]:
    _mapping = {
        'FF': layer - 1,
        'E2E': layer
    }
    if projs is not None:
        return {k: _mapping[k] for k in projs}
    else:
        return _mapping.copy()


def get_averages_stdev(
    rates: RatesArray,
    conditional_nonzero: bool = False
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Get averages and standard deviations of firing rates DataArray.

    Args:
        rates (RatesArray): Firing rates, including dim `rep`.
        conditional_nonzero (bool, optional): Whether to exclude nonzero rates. Defaults to False.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: Averages and stdevs of rates.
    """
    values = rates.transpose(..., 'rep').values
    shape_ = list(np.shape(values))
    del shape_[-1]
    if conditional_nonzero:
        indices = np.nonzero(values.mean(-1))
        averages = np.zeros(shape_)
        stdevs = np.zeros(shape_)
        for coord in zip(*indices):
            rs = values[coord][values[coord] > 0]
            averages[coord] = np.mean(rs)
            stdevs[coord] = np.std(rs)
    else:
        averages = np.mean(values, -1)
        stdevs = np.std(values, -1)
    return averages, stdevs
