from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from ._types import DeltaTuple
from . import _utils_traceback as _utils_tb

__all__ = [
    "get_exact_sensitivities",
    "get_sensitivities",
    "reduce_sensitivities",
    "get_weight_matrices",
    "get_centroid",
    "get_centroid_stdevs"
]


def get_exact_sensitivities(post_ids: npt.ArrayLike, gradients: xr.DataArray,
                            syn_params: pd.DataFrame, num_inputs: int,
                            weights: list[npt.NDArray[np.float_]] | None = None) -> npt.NDArray[np.float_]:
    """Get exact sensitivities of final layer neurons to input layer activations.

    Args:
        post_ids (npt.ArrayLike): Get sensitivities w.r.t. these final layer neurons.
        gradients (xr.DataArray): Estimated activation gradients, dims: ('layer', 'nrn').
        syn_params (pd.DataFrame): Synapse parameters containing weights, including index `FF`.
        num_inputs (int): Number of input layer neurons.
        weights (list[npt.NDArray[np.float_]] | None, optional): Predetermined weights for optimisation. Defaults to None.

    Raises:
        ValueError: Invalid gradients dims, expected: (`layer`, `nrn`).

    Returns:
        npt.NDArray[np.float_]: Sensitivities to input neurons (originally indexed `(Z, Y, X)`).
    """
    _dims = {'layer', 'nrn'}
    if set(gradients.dims) != _dims:
        raise ValueError(f"gradients must have dims: {_dims}")

    post_ids = np.asarray(post_ids)
    L = int(gradients['layer'][-1])
    num_nrns = len(gradients['nrn'])
    syn_params_ = _utils_tb.process_syn_params(syn_params, L)['w']
    if weights is None:
        weights = get_weight_matrices(syn_params_, num_nrns, num_inputs)
    else:
        weights = weights

    deltas = np.zeros(num_nrns)
    deltas[post_ids] = gradients.sel(layer=L, nrn=post_ids).values
    for i, w in enumerate(weights[1:][::-1]):
        l = L - i
        deltas = deltas.dot(w) * gradients.sel(layer=l-1).values
    return deltas.dot(weights[0])


def get_sensitivities(post_ids: npt.ArrayLike, gradients: xr.DataArray,
                      syn_params: pd.DataFrame, threshold: float = 0.0) -> DeltaTuple:
    """Get sensitivities of final layer neurons to input layer activations.

    Args:
        post_ids (npt.ArrayLike): Get sensitivities w.r.t. these final layer neurons.
        gradients (xr.DataArray): Estimated activation gradients, dims: ('layer', 'nrn') (e.g. active / silent).
        syn_params (pd.DataFrame): Synapse parameters containing or corresponding to ('FF').

    Raises:
        ValueError: Invalid gradients dims, expected: ('layer', 'nrn').

    Returns:
        DeltaTuple: Sensitivities, as 2-tuple containing (L0 ids, sensitivities).
    """
    _dims = {'layer', 'nrn'}
    if set(gradients.dims) != _dims:
        raise ValueError(f"gradients must have dims: {_dims}")
    post_ids = np.asarray(post_ids)
    L = int(gradients['layer'][-1])

    syn_params = _utils_tb.process_syn_params(syn_params, L)
    deltas = (post_ids, np.array(gradients.sel(layer=L, nrn=post_ids), dtype=float))
    syn_params_ = syn_params.sort_index(level=['layer', 'pre', 'post'], ascending=[False, True, True])
    for l, grp in syn_params_.groupby('layer', sort=False):
        l = cast(int, l)
        idx = grp.index.get_level_values('post')
        grp = grp[idx.isin(deltas[0])]
        deltas = _utils_tb.get_deltas(grp, l, gradients, deltas, threshold=threshold)
    return deltas


def reduce_sensitivities(sensitivities: npt.NDArray[np.float_] | DeltaTuple, input_shape: tuple) -> pd.Series:
    """Average out channel sensitivities at each X, Y position.

    Args:
        sensitivities (npt.NDArray[np.float_] | DeltaTuple): Computed sensitivity parameters.
        input_shape (tuple): Shape of the input layer.

    Returns:
        pd.Series: Reduced sensitivities, indexed as X, Y with associated sensitivity values.
    """
    if isinstance(sensitivities, tuple):
        coords = np.unravel_index(sensitivities[0], input_shape)
        data = sensitivities[1]
    elif isinstance(sensitivities, np.ndarray):
        indices = np.flatnonzero(sensitivities)
        coords = np.unravel_index(indices, input_shape)
        data = sensitivities[indices]
    midx = pd.MultiIndex.from_arrays(coords, names=['channel', 'Y', 'X'])
    return pd.Series(data=data, index=midx).groupby(['X', 'Y']).sum()


def get_weight_matrices(syn_params: pd.Series, num_nrns: int,
                        num_inputs: int) -> list[npt.NDArray[np.float_]]:
    """Get weight matrix for each layer in `syn_params`.

    Args:
        syn_params (pd.Series): Synaptic weights series, indexed: `layer`, `pre`, `post`.
        num_nrns (int): Number of excitatory neurons in layers after L0.
        num_inputs (int): Total number of input layer neurons.

    Returns:
        list[npt.NDArray[np.float_]]: Weight matrices, indexed by (post) layer.
    """
    assert set(syn_params.index.names) == {'layer', 'pre', 'post'}
    weights = []
    for layer, grp in syn_params.groupby('layer'):
        num_pre = num_inputs if layer == 1 else num_nrns
        weights.append(_utils_tb.series_to_array(grp, num_nrns, num_pre))
    return weights


def get_centroid(sensitivities: pd.Series) -> tuple[float, float] | None:
    """Get weighted mean values for X and Y.

    Args:
        sensitivities (pd.Series): Computed traceback sensitivities (reduced).

    Returns:
        tuple[float, float] | None: Weighted mean values for X and Y.
    """
    sensitivities = sensitivities.dropna()
    if len(sensitivities) == 0:
        return None

    xs = sensitivities.index.get_level_values('X').values
    ys = sensitivities.index.get_level_values('Y').values
    weights = sensitivities.values / np.sum(sensitivities)
    return _utils_tb.get_weighted_average(xs, weights), _utils_tb.get_weighted_average(ys, weights)


def get_centroid_stdevs(sensitivities: pd.Series) -> tuple[float, float] | None:
    """Get weighted standard deviations for X and Y.

    Args:
        sensitivities (pd.Series): Computed traceback sensitivities (reduced).

    Returns:
        tuple[float, float] | None: Weighted standard deviations for X and Y.
    """
    sensitivities = sensitivities.dropna()
    if len(sensitivities) == 0:
        return None

    xs = sensitivities.index.get_level_values('X').values
    ys = sensitivities.index.get_level_values('Y').values
    weights = sensitivities.values / np.sum(sensitivities)
    return _utils_tb.get_weighted_stdev(xs, weights), _utils_tb.get_weighted_stdev(ys, weights)
