# type: ignore
import pandas as pd

from ..base import get_proj_layer_mapping, get_midx
from . import _utils

pidx = pd.IndexSlice


def get_structural_indices(nrn: int, layer: int, syn_params: pd.DataFrame,
                           w_min: float = 0.5, tol: float = 3.0) -> pd.MultiIndex:
    """Gets network (layer, neuron) indices filtered to those structurally linked to a select neuron.

    This corresponds to a three-neuron HFB motif, where the select neuron is the high-level feature neuron.

    Args:
        nrn (int): Select neuron index (i.e. a selective neuron constituting part of an HFB).
        layer (int): Select neuron's layer index.
        syn_params (pd.DataFrame): Network synaptic parameters, containing `weight` and `delay`.
        w_min (float, optional): Minimum afferent and efferent weights of `nrn`. Defaults to 0.5.
        tol (float, optional): Tolerance of spike propagation delay of included synaptic pathways. Defaults to 3.0.
    """
    # Step 0: Filter syn_params to entries with w >= w_min (w_min = 0.5)
    syn_params_ = syn_params[syn_params['w'] >= w_min]

    try:
        # Step 1: Get all (weights filtered) presynaptic IDs for select L4 neuron
        select_presyn = syn_params_.xs(key=(layer, 'FF'), level=('layer', 'proj')).loc[pidx[:, nrn], :]
        select_preIDs = select_presyn.index.get_level_values('pre')

        # Step 2: Prepare candidate set of preIDs, postIDs of the select L4 neuron
        candidate_preIDs = set()
        candidate_postIDs = set()

        # Step 3: Iterate over each postsyn target of select L4 neuron, accumulate pool of pre/post candidate IDs
        select_postsyn = syn_params_.xs(key=(layer, 'E2E'), level=('layer', 'proj')).loc[pidx[nrn, :], :].droplevel('pre')
        select_postIDs = select_postsyn.index.get_level_values('post')  # binding neuron IDs
    except KeyError:
        return get_midx([], [])

    for postID in select_postIDs:
        select2target_delay = select_postsyn.loc[postID]['delay']  # d_HB (scalar)
        target_presyn = syn_params_.xs(key=(layer, 'FF'), level=('layer', 'proj')).loc[pd.IndexSlice[:, postID], :]
        common_preIDs = target_presyn.index.get_level_values('pre').intersection(select_preIDs)

        d_lh = select_presyn.loc[common_preIDs]['delay'].droplevel('post')  # d_LH (vector: {L}->H)
        d_sum = select2target_delay + d_lh  # fastest indirect route: d_LH + d_HB (vector: {L}->H->B)

        lbs = (d_sum - tol).clip(lower=0)  # allow direct route to be up to tol faster
        ubs = d_sum + 2 * tol  # allow tol at H + tol at B

        pre2target_delays = target_presyn.loc[common_preIDs]['delay'].droplevel('post')  # d_LB (vector: {L}->B)
        preIDs = pre2target_delays.index[(pre2target_delays >= lbs) & (pre2target_delays <= ubs)]

        if len(preIDs):
            candidate_postIDs.update([postID])
            candidate_preIDs.update(preIDs)
    if len(candidate_postIDs):
        candidate_postIDs.update([nrn])
    indices_pre = get_midx(layer - 1, candidate_preIDs)
    indices_post = get_midx(layer, candidate_postIDs)
    return indices_pre.append(indices_post).sort_values()


def get_presyn_indices(post: int, layer: int, syn_params: pd.DataFrame,
                       w_min: float = 0.5, proj: tuple = ('FF', 'E2E')) -> pd.MultiIndex:
    """Retrieves a MultiIndex of presynaptic neuron indices for a given postsynaptic neuron,
    filtered by synaptic weight and projection type, including the postID itself.

    Args:
        post (int): Postsynaptic neuron index.
        layer (int): Layer of the postsynaptic neuron.
        syn_params (pd.DataFrame): Network synaptic parameters, including `w`.
        w_min (float, optional): Minimum included afferent weight value. Defaults to `0.5`.
        proj (tuple, optional): Projection types to filter by. Defaults to `('FF', 'E2E')`.

    Returns:
        pd.MultiIndex: Index consisting of tuples `(layer, nrn)` immediately associated
        with the given postID.
    """
    syn_pre = syn_params.xs((layer, post), level=('layer', 'post')).loc[pidx[proj, :], :]
    index_pre = syn_pre[syn_pre['w'] >= w_min].index
    layers = index_pre.get_level_values('proj').map(get_proj_layer_mapping(layer))
    index_pre = get_midx(layers, index_pre.get_level_values('pre'))
    index_post = get_midx(layer, post)
    return index_pre.append(index_post).sort_values()


def get_triad_indexes(
    nrn: int,
    layer: int,
    syn_params: pd.DataFrame,
    w_min: float = 0.5,
    tol: float = 3.0,
) -> list[pd.MultiIndex]:
    try:
        syn_low_high = syn_params.loc[pidx[layer, "FF", :, nrn]]
        syn_low_high = syn_low_high.loc[syn_low_high["w"] >= w_min]
        pre_high: pd.Index = syn_low_high.index.unique("pre")

        syn_high_bind = syn_params.loc[pidx[layer, "E2E", nrn, :]]
        syn_high_bind = syn_high_bind[syn_high_bind["w"] >= w_min]
        post_high: pd.Index = syn_high_bind.index.unique("post")

        syn_low_bind = syn_params.loc[pidx[layer, "FF", pre_high, post_high]].droplevel(("layer", "proj"))
        syn_low_bind = syn_low_bind[syn_low_bind["w"] >= w_min]

        candidate_low: pd.Index = syn_low_bind.index.unique("pre").sort_values()
        candidate_bind: pd.Index = syn_low_bind.index.unique("post").sort_values()

        delays_low_high: pd.Series = syn_low_high.loc[candidate_low, "delay"]
        delays_high_bind: pd.Series = syn_high_bind.loc[candidate_bind, "delay"]
        delays_low_bind: pd.Series = syn_low_bind["delay"]
    except KeyError:
        return []

    midxs = []
    for low_nrn in candidate_low:
        delays_LB = delays_low_bind.loc[low_nrn]
        for bind_nrn in delays_LB.index.unique("post"):
            delays_map = {
                "LH": delays_low_high.loc[low_nrn],
                "HB": delays_high_bind.loc[bind_nrn],
                "LB": delays_low_bind.loc[(low_nrn, bind_nrn)],
            }
            if _utils.triad_consistent(delays_map, tol=tol):
                midxs.append(get_midx(
                    [layer - 1, layer, layer], [low_nrn, nrn, bind_nrn])
                )
    return midxs


def get_triad_indexes_unconstrained(
    nrn: int,
    layer: int,
    syn_params: pd.DataFrame,
) -> list[pd.MultiIndex]:
    """Get all possible triad indexes with [L-1, L, L] layer structure without
    synaptic weight or delay constraints.

    Returns all neuron combinations where:
    - First neuron is in layer L-1 (low-level)
    - Second neuron is `nrn` in layer L (high-level)
    - Third neuron is in layer L (binding)

    Only requires that the synaptic connections exist (any weight).

    Args:
        nrn: Second-firing focal neuron (high-level).
        layer: Layer of focal neuron.
        syn_params: Network synaptic parameters.

    Returns:
        List of MultiIndex, each containing (layer, nrn) tuples for a triad.
    """
    try:
        # Get all presynaptic neurons in L-1 that project to nrn (any weight)
        syn_low_high = syn_params.loc[pidx[layer, "FF", :, nrn]]
        pre_high: pd.Index = syn_low_high.index.unique("pre")

        # Get all postsynaptic neurons in L that nrn projects to (any weight)
        syn_high_bind = syn_params.loc[pidx[layer, "E2E", nrn, :]]
        post_high: pd.Index = syn_high_bind.index.unique("post")

        # Get neurons in L-1 that project to both nrn AND binding neurons
        syn_low_bind = syn_params.loc[pidx[layer, "FF", pre_high, post_high]].droplevel(("layer", "proj"))

        candidate_low: pd.Index = syn_low_bind.index.unique("pre").sort_values()
        candidate_bind: pd.Index = syn_low_bind.index.unique("post").sort_values()
    except KeyError:
        return []

    # Generate all valid triads (no delay/weight filtering)
    midxs = []
    for low_nrn in candidate_low:
        try:
            bind_nrns = syn_low_bind.loc[low_nrn].index.unique("post")
        except KeyError:
            continue
        for bind_nrn in bind_nrns:
            if bind_nrn in candidate_bind:
                midxs.append(get_midx(
                    [layer - 1, layer, layer], [low_nrn, nrn, bind_nrn])
                )
    return midxs
