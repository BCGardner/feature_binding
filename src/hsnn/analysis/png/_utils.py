import os
import sys
from itertools import combinations
from typing import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

from .base import PNG
from hsnn import ops


class PrintManager:
    def __init__(self, verbose: bool = False) -> None:
        self._verbose = verbose

    def __enter__(self):
        if not self._verbose:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._verbose:
            sys.stdout.close()
            sys.stdout = self._original_stdout


def drop_repeating(pngs: Sequence[PNG]) -> Sequence[PNG]:
    kept: list[PNG] = []
    for png in pngs:
        pairs = list(zip(png.layers, png.nrns))
        if len(pairs) == len(set(pairs)):
            kept.append(png)
    return kept


def _merge_stategy(element: dict[str, np.ndarray], lags: npt.NDArray[np.float64],
                   times: npt.NDArray[np.float64], strategy: str = 'mode') -> None:
    if strategy == 'mode':
        if len(times) > len(element['times']):
            element['lags'] = lags
    elif strategy == 'mean':
        arrays = np.stack([element['lags'], lags], axis=0)
        weights = [len(element['times']), len(times)]
        element['lags'] = np.average(arrays, axis=0, weights=weights)
    else:
        raise NotImplementedError(f"merge strategy '{strategy}' unsupported")


def _concat_times(t1: npt.NDArray[np.float64], t2: npt.NDArray[np.float64],
                  min_sep: float) -> npt.NDArray[np.float64]:
    args = np.flatnonzero(np.abs(t2[:, np.newaxis] - t1).min(axis=1) >= min_sep)
    return np.sort(np.concatenate([t1, t2[args]]))


def merge_pngs(pngs: Sequence[PNG], tol: float = 1, strategy: str = 'mode',
               overlaps: bool = False, decimals: int = 0) -> Sequence[PNG]:
    """Merge PNGs that are equal under PNG.__hash__/__eq__ and whose lag
    patterns are within a tolerance. Two PNGs that are not hash-equal
    are never considered for merging.
    """
    # Group PNGs by their hash/equality identity
    by_id: dict[PNG, list[PNG]] = {}
    for png in pngs:
        by_id.setdefault(png, []).append(png)

    merged: list[PNG] = []

    # Apply lag-based merging for each group
    for group_pngs in by_id.values():
        mapping: list[dict[str, np.ndarray]] = []

        for png in group_pngs:
            # Canonical per-PNG ordering for lags/layers/nrns
            indices = np.lexsort((png.nrns, png.layers))
            lags = png.lags[indices]
            layers = png.layers[indices]
            nrns = png.nrns[indices]
            times = png.times

            for elem in mapping:
                if np.abs(elem['lags'] - lags).max() <= tol:
                    _merge_stategy(elem, lags, times, strategy)
                    min_sep = 0.0 if overlaps else elem['lags'].max()
                    elem['times'] = _concat_times(elem['times'], times, min_sep)
                    break
            else:
                mapping.append(dict(lags=lags, times=times,
                                    layers=layers, nrns=nrns))

        # Reconstruct PNGs for this identity group
        for elem in mapping:
            indices = np.argsort(elem['lags'])
            layers = elem['layers'][indices]
            nrns = elem['nrns'][indices]
            lags = np.round(elem['lags'][indices], decimals=decimals)
            merged.append(PNG(layers, nrns, lags, elem['times']))

    merged = sorted(merged, key=lambda x: len(x.times), reverse=True)
    return merged


def iscausal(diffs: npt.ArrayLike, delays: npt.ArrayLike, tol: float) -> bool:
    """Determines if relative timing differences `diffs` are greater than
    reference timing differences `delays` and within tolerance `tol`.

    Args:
        diffs (npt.ArrayLike): Relative timing differences.
        delays (npt.ArrayLike): Reference timing differences (e.g. conduction delays).
        tol (float): Tolerance window to assume a match.

    Returns:
        bool: True if `diffs` are greater than `delays`, within `tol`.
    """
    rise_times = np.asarray(diffs) - np.asarray(delays)
    return np.all((rise_times >= 0) & (rise_times <= tol)).item()


def _get_diffs_matrix(png: PNG) -> pd.DataFrame:
    """Gets a matrix of spike-timing differences between all neuron pairs.

    Args:
        png (PNG): Containing `layers`, `nrns`, and `lags`.

    Returns:
        pd.DataFrame: Timing differences, where rows/columns are indexed by `(layer, nrn)`.
    """
    index = pd.MultiIndex.from_arrays([png.layers, png.nrns], names=['layer', 'nrn'])
    data = png.lags[:, np.newaxis] - png.lags
    return pd.DataFrame(data=data, index=index, columns=index)


def isstructural(png: PNG, syn_params: pd.DataFrame, projs: Sequence[str] = ('FF', 'E2E'),
                 tol: float = 3.0) -> bool:
    """Checks if a given PNG structurally aligns with the underlying network
    structure according to its synaptic parameters `syn_params`.

    Args:
        png (PNG): A polychronous neuronal group.
        syn_params (pd.DataFrame): Containing conduction delays `delay`.
        projs (Sequence[str], optional): Synaptic projections considered. Defaults to ('FF', 'E2E').
        tol (float, optional): Tolerance level for assessing timing alignments. Defaults to 3.0.

    Returns:
        bool: A boolean indicating if the PNG is structural.
    """
    diffs_matrix = _get_diffs_matrix(png)
    for (layer, post), row in diffs_matrix.iterrows():  # type: ignore
        row_ = row.drop((layer, post))  # Ignore self-connections
        causal_diffs = row_[row_ >= 0]  # Afferent timing diffs indexed by (layer, nrn)
        # Get presynaptic layers, nrns, and the associated projection types
        layers_pre = causal_diffs.index.get_level_values('layer').values
        nrns_pre = causal_diffs.index.get_level_values('nrn').values
        projs_pre_post = [ops.layers_to_proj(layer, layer_pre) for layer_pre in layers_pre]
        for nrn_pre, proj, t_diff in zip(nrns_pre, projs_pre_post, causal_diffs):
            if proj in projs:
                try:
                    delay = syn_params.xs((layer, proj, nrn_pre, post))['delay']
                    if not iscausal(t_diff, delay, tol):
                        return False
                except KeyError:
                    # PNG is not structurally aligned if network lacks connection
                    return False
    return True


def isconstrained(png: PNG, syn_params: pd.DataFrame, w_min: float = 0.5,
                  tol: float = 3.0) -> bool:
    """Determines if a given PNG satisfies synaptic constraints, including both weight
    `w` and `delay`.

    Args:
        png (PNG): A polychronous neuronal group.
        syn_params (pd.DataFrame): Containing `w` and `delay`.
        w_min (float, optional): Minimum weight of connections. Defaults to 0.5.
        tol (float, optional): Tolerance level for assessing timing alignments. Defaults to 3.0.

    Returns:
        bool: A boolean indicating if the PNG matches the synaptic constraints.
    """
    idx_pairs = list(combinations(range(len(png.layers)), 2))
    # triad_delays: dict[str, float] = {}
    for i, j in idx_pairs:
        layer_pre = png.layers[i]
        layer_post = png.layers[j]
        nrn_pre = png.nrns[i]
        nrn_post = png.nrns[j]
        t_diff = float(png.lags[j] - png.lags[i])
        # Reject unsupported layer gaps
        if abs(layer_post - layer_pre) > 1:
            return False
        proj = ops.layers_to_proj(layer_post, layer_pre)
        try:
            data = syn_params.xs((layer_post, proj, nrn_pre, nrn_post))
        except KeyError:
            return False
        if float(data['w']) < w_min:  # type: ignore
            return False
        delay_val = float(data['delay'])  # type: ignore
        if not iscausal(t_diff, delay_val, tol):
            return False
    #     # Accumulate for triad composite check
    #     if len(png.layers) == 3:
    #         if proj == 'FF':
    #             if layer_pre == layer_post - 1:
    #                 # distinguish low->high vs low->bind by postsyn neuron equality later
    #                 if j == 1:  # heuristic: second firing neuron assumed high-level
    #                     triad_delays["LH"] = delay_val
    #                 else:
    #                     triad_delays["LB"] = delay_val
    #         elif proj == 'E2E':
    #             triad_delays["HB"] = delay_val
    # if len(png.layers) == 3 and len(triad_delays) == 3:
    #     if not _triad_consistent(triad_delays, tol):
    #         return False
    return True


def layers_to_proj(layer_post: int, layer_pre: int) -> str:
    """Gets the projection name connecting the given layers.

    Args:
        layer_post (int): Postsynaptic layer.
        layer_pre (int): Presynaptic layer

    Raises:
        ValueError: Raised if no projection exists between these layers.

    Returns:
        str: Projection name.
    """
    if layer_post == layer_pre:
        return 'E2E'
    elif layer_post == layer_pre + 1:
        return 'FF'
    elif layer_post == layer_pre - 1:
        return 'FB'
    else:
        raise ValueError(f"No projection between layers {layer_pre} and {layer_post}")


def triad_consistent(delays_map: dict[str, float], tol: float) -> bool:
    """Check composite delay relation for a 3-neuron HFB:
    d_low_bind ~ d_low_high + d_high_bind subject to tolerance, tol.
    Keys: 'LH', 'HB', 'LB'
    """
    if {"LH", "HB", "LB"} - set(delays_map):
        return False
    lhs = delays_map["LB"]  # direct route
    rhs = delays_map["LH"] + delays_map["HB"]  # indirect route
    return (rhs - tol) <= lhs <= (rhs + 2 * tol)


def isconnected(png: PNG, syn_params: pd.DataFrame) -> bool:
    """Check if all neuron pairs in a PNG have synaptic connections.

    Only verifies connection existence, not weight or delay constraints.
    For HFB triplets with structure [L-1, L, L], checks:
    - Low -> High (feedforward)
    - Low -> Bind (feedforward)
    - High -> Bind (lateral E2E)

    Args:
        png: A polychronous neuronal group.
        syn_params: Network synaptic parameters with index [layer, proj, pre, post].

    Returns:
        True if all required connections exist in the network.
    """
    idx_pairs = list(combinations(range(len(png.layers)), 2))
    for i, j in idx_pairs:
        layer_pre = png.layers[i]
        layer_post = png.layers[j]
        nrn_pre = png.nrns[i]
        nrn_post = png.nrns[j]

        # Reject unsupported layer gaps
        if abs(layer_post - layer_pre) > 1:
            return False

        proj = ops.layers_to_proj(layer_post, layer_pre)
        # Just check existence - don't filter by weight or delay
        if (layer_post, proj, nrn_pre, nrn_post) not in syn_params.index:
            return False
    return True
