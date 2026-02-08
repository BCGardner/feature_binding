import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from ._types import DeltaTuple

pidx = pd.IndexSlice


def get_deltas(group: pd.DataFrame, layer: int, surrogates: xr.DataArray,
                deltas: DeltaTuple, threshold: float = 0.0) -> DeltaTuple:
    """Get delta values for neurons in previous layer of current one, l := l - 1.
    """
    pre_ids = []
    deltas_ = []
    for pre, grp in group.groupby('pre'):
        mask = np.isin(deltas[0], grp.index.get_level_values('post').values)
        vals = deltas[1][mask]
        delta = np.sum(np.asarray(grp['w']) * vals).item()
        if layer > int(surrogates['layer'].min()):
            delta *= float(surrogates.sel(layer=layer-1, nrn=pre))
        if delta > threshold:
            pre_ids.append(pre)
            deltas_.append(delta)
    return np.array(pre_ids, dtype=int), np.array(deltas_, dtype=float)


def get_weighted_average(vals: np.ndarray, weights: np.ndarray) -> float:
    assert np.isclose(np.sum(weights), 1.0)
    return np.sum(vals * weights).item()


def get_weighted_stdev(vals: np.ndarray, weights: np.ndarray) -> float:
    vals_av = get_weighted_average(vals, weights)
    return float(np.sqrt(np.sum(weights * (vals - vals_av)**2) / np.sum(weights)))


def series_to_array(series: pd.Series, num_post: int, num_pre: int) -> npt.NDArray[np.float_]:
    """Constructs a sparse weight matrix with dims (`post`, `pre`) from synaptic weights in a layer.

    Args:
        series (pd.Series): Synaptic weights for a layer, indexed with `pre` and `post`.
        num_post (int): Num. post nrns.
        num_pre (int): Num. pre nrns.

    Returns:
        npt.NDArray[np.float_]: Sparse weight matrix with dims (`post`, `pre`).
    """
    indices = [series.index.get_level_values('post'), series.index.get_level_values('pre')]
    dims = (num_post, num_pre)
    args = np.ravel_multi_index(indices, dims)
    ws = np.zeros(np.prod(dims))
    ws[args] = np.asarray(series)
    return ws.reshape(dims)


def process_syn_params(syn_params: pd.DataFrame, L: int) -> pd.DataFrame:
    """Extract relevant `FF` synaptic parameters up to last layer `L`.
    """
    syn_params = syn_params.loc[pidx[:L]]
    if set(syn_params.index.unique('layer')) != set(range(1, L + 1)):
        raise ValueError(f"synapse parameters missing layer(s)")
    if 'proj' in syn_params.index.names:
        syn_params = syn_params.xs('FF', level='proj')  # type: ignore[assignment]
    return syn_params
