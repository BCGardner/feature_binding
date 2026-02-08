import pandas as pd
import numpy as np

from hsnn.analysis.png.filters import get_structural_indices
from hsnn.analysis.base import get_midx


def _make_syn_params(rows):
    """
    rows: list of dicts with keys: layer_post, proj, pre, post, w, delay
    """
    df = pd.DataFrame(rows)
    df.set_index(['layer_post', 'proj', 'pre', 'post'], inplace=True)
    df.index.rename(['layer', 'proj', 'pre', 'post'], inplace=True)
    return df[['w', 'delay']]


def test_returns_empty_when_select_neuron_has_no_connections():
    syn_params = _make_syn_params([
        # everything is on layer 5, not layer 4
        dict(layer_post=5, proj='FF', pre=10, post=20, w=0.6, delay=2.0),
    ])
    idx = get_structural_indices(nrn=20, layer=4, syn_params=syn_params,
                                 w_min=0.5, tol=3.0)
    assert len(idx) == 0


def test_returns_empty_when_weights_below_threshold():
    # select neuron has pre/post, but all weights < w_min
    syn_params = _make_syn_params([
        dict(layer_post=4, proj='FF', pre=10, post=20, w=0.49, delay=2.0),
        dict(layer_post=4, proj='E2E', pre=20, post=30, w=0.49, delay=3.0),
    ])
    idx = get_structural_indices(nrn=20, layer=4, syn_params=syn_params,
                                 w_min=0.5, tol=3.0)
    assert len(idx) == 0


def test_simple_valid_triad_includes_pre_high_bind():
    """
    Construct a single valid L-H-B triad:

        layer-1: pre = 10
        layer  : high = 20 (select), bind = 30

    with delays satisfying the tolerance, and check that
    get_structural_indices returns:

        (layer-1, pre), (layer, high), (layer, bind)
    """
    layer = 4
    pre, high, bind = 10, 20, 30

    syn_params = _make_syn_params([
        # L -> H
        dict(layer_post=layer, proj='FF', pre=pre, post=high, w=0.6, delay=2.0),
        # H -> B
        dict(layer_post=layer, proj='E2E', pre=high, post=bind, w=0.7, delay=3.0),
        # L -> B; composite delay: 2 + 3 = 5, so choose LB around 5 within tol
        dict(layer_post=layer, proj='FF', pre=pre, post=bind, w=0.8, delay=5.0),
    ])

    idx = get_structural_indices(nrn=high, layer=layer, syn_params=syn_params,
                                 w_min=0.5, tol=1.0)

    # expected set of (layer, nrn)
    expected = get_midx([layer - 1, layer, layer], [pre, high, bind])

    assert set(idx) == set(expected)
    # ensure sorted (pre, high, bind) order as implemented
    assert list(idx) == sorted(expected)


def test_excludes_targets_with_no_common_presyn():
    """
    If a candidate binding neuron B does not share any presynaptic
    neurons above w_min with the select neuron H, it should not be included.
    """
    layer = 4
    pre_shared, pre_other = 10, 11
    high, bind_ok, bind_bad = 20, 30, 31

    syn_params = _make_syn_params([
        # L(shared) -> H
        dict(layer_post=layer, proj='FF', pre=pre_shared, post=high, w=0.6, delay=2.0),
        # L(shared) -> bind_ok
        dict(layer_post=layer, proj='FF', pre=pre_shared, post=bind_ok, w=0.6, delay=5.0),
        # H -> bind_ok
        dict(layer_post=layer, proj='E2E', pre=high, post=bind_ok, w=0.7, delay=3.0),

        # L(other) -> bind_bad, but no L(other) -> H => no common pre
        dict(layer_post=layer, proj='FF', pre=pre_other, post=bind_bad, w=0.8, delay=5.0),
        dict(layer_post=layer, proj='E2E', pre=high, post=bind_bad, w=0.9, delay=3.0),
    ])

    idx = get_structural_indices(nrn=high, layer=layer, syn_params=syn_params,
                                 w_min=0.5, tol=1.0)

    # only pre_shared and bind_ok (plus high) should be included
    expected = get_midx([layer - 1, layer, layer], [pre_shared, high, bind_ok])
    assert set(idx) == set(expected)
    assert (layer, bind_bad) not in idx


def test_delay_window_respected_by_tol():
    """
    For a given triad, only include (pre, bind) pairs where the direct
    delay L->B lies in [LH+HB - tol, LH+HB + 2*tol].
    """
    layer = 4
    pre, high = 10, 20
    bind_ok, bind_too_fast, bind_too_slow = 30, 31, 32

    LH = 3.0
    HB_ok = 4.0
    sum_ok = LH + HB_ok  # 7.0
    tol = 1.0
    lbs = sum_ok - tol      # 6.0
    ubs = sum_ok + 2 * tol  # 9.0

    syn_params = _make_syn_params([
        # L -> H
        dict(layer_post=layer, proj='FF', pre=pre, post=high, w=0.6, delay=LH),

        # H -> each candidate bind
        dict(layer_post=layer, proj='E2E', pre=high, post=bind_ok, w=0.7, delay=HB_ok),
        dict(layer_post=layer, proj='E2E', pre=high, post=bind_too_fast, w=0.7, delay=HB_ok),
        dict(layer_post=layer, proj='E2E', pre=high, post=bind_too_slow, w=0.7, delay=HB_ok),

        # L -> B direct delays:
        dict(layer_post=layer, proj='FF', pre=pre, post=bind_ok, w=0.8, delay=7.0),   # inside [6, 9] => keep
        dict(layer_post=layer, proj='FF', pre=pre, post=bind_too_fast, w=0.8, delay=4.0),  # < lbs => drop
        dict(layer_post=layer, proj='FF', pre=pre, post=bind_too_slow, w=0.8, delay=10.0), # > ubs => drop
    ])

    idx = get_structural_indices(nrn=high, layer=layer, syn_params=syn_params,
                                 w_min=0.5, tol=tol)

    expected = get_midx([layer - 1, layer, layer], [pre, high, bind_ok])
    assert set(idx) == set(expected)
    assert (layer, bind_too_fast) not in idx
    assert (layer, bind_too_slow) not in idx


def test_candidate_postids_includes_select_neuron_when_any_binding_found():
    """
    When at least one candidate binding neuron is found, the select neuron
    itself should be included in the 'post' set.
    """
    layer = 4
    pre, high, bind = 10, 20, 30

    syn_params = _make_syn_params([
        dict(layer_post=layer, proj='FF', pre=pre, post=high, w=0.6, delay=2.0),
        dict(layer_post=layer, proj='E2E', pre=high, post=bind, w=0.7, delay=3.0),
        dict(layer_post=layer, proj='FF', pre=pre, post=bind, w=0.8, delay=5.0),
    ])

    idx = get_structural_indices(nrn=high, layer=layer, syn_params=syn_params,
                                 w_min=0.5, tol=1.0)

    # both high and bind should be present at layer
    posts = [nrn for (layer_idx, nrn) in idx if layer_idx == layer]
    assert high in posts
    assert bind in posts


def test_multiple_presyn_candidates_for_same_binding():
    """
    Two different presynaptic neurons (L1, L2) both satisfy the composite
    delay condition for the same binding neuron, they should both be included.
    """
    layer = 4
    pre1, pre2 = 10, 11
    high, bind = 20, 30

    LH1 = 2.0
    LH2 = 3.0
    HB = 4.0
    tol = 1.0

    syn_params = _make_syn_params([
        # L1 -> H, L2 -> H
        dict(layer_post=layer, proj='FF', pre=pre1, post=high, w=0.6, delay=LH1),
        dict(layer_post=layer, proj='FF', pre=pre2, post=high, w=0.6, delay=LH2),

        # H -> B
        dict(layer_post=layer, proj='E2E', pre=high, post=bind, w=0.7, delay=HB),

        # L1 -> B: around LH1 + HB
        dict(layer_post=layer, proj='FF', pre=pre1, post=bind, w=0.8, delay=LH1 + HB),
        # L2 -> B: around LH2 + HB
        dict(layer_post=layer, proj='FF', pre=pre2, post=bind, w=0.8, delay=LH2 + HB),
    ])

    idx = get_structural_indices(nrn=high, layer=layer, syn_params=syn_params,
                                 w_min=0.5, tol=tol)

    expected = get_midx(
        [layer - 1, layer - 1, layer, layer],
        [pre1, pre2, high, bind],
    )
    assert set(idx) == set(expected)


def test_multiple_binding_neurons_for_same_high_neuron():
    """
    The select neuron H can form valid triads with multiple binding neurons
    (B1, B2). All such binding neurons should be included.
    """
    layer = 4
    pre = 10
    high = 20
    bind1, bind2 = 30, 31

    LH = 2.0
    HB1 = 3.0
    HB2 = 4.0
    tol = 1.0

    syn_params = _make_syn_params([
        # L -> H
        dict(layer_post=layer, proj='FF', pre=pre, post=high, w=0.6, delay=LH),

        # H -> B1, B2
        dict(layer_post=layer, proj='E2E', pre=high, post=bind1, w=0.7, delay=HB1),
        dict(layer_post=layer, proj='E2E', pre=high, post=bind2, w=0.7, delay=HB2),

        # L -> B1 approx LH + HB1
        dict(layer_post=layer, proj='FF', pre=pre, post=bind1, w=0.8, delay=LH + HB1),
        # L -> B2 approx LH + HB2
        dict(layer_post=layer, proj='FF', pre=pre, post=bind2, w=0.8, delay=LH + HB2),
    ])

    idx = get_structural_indices(nrn=high, layer=layer, syn_params=syn_params,
                                 w_min=0.5, tol=tol)

    expected = get_midx(
        [layer - 1, layer, layer, layer],
        [pre, high, bind1, bind2],
    )
    assert set(idx) == set(expected)


def test_no_binding_when_all_delays_violate_window():
    """
    Even if H and candidate binding neurons share presyns and weights >= w_min,
    if all L->B delays are outside the [lbs, ubs] window, there should be
    no binding neurons selected.
    """
    layer = 4
    pre = 10
    high = 20
    bind1, bind2 = 30, 31

    LH = 3.0
    HB = 4.0
    tol = 1.0
    sum_ = LH + HB  # 7.0
    lbs = sum_ - tol      # 6.0
    ubs = sum_ + 2 * tol  # 9.0

    syn_params = _make_syn_params([
        # L -> H
        dict(layer_post=layer, proj='FF', pre=pre, post=high, w=0.6, delay=LH),

        # H -> both bindings
        dict(layer_post=layer, proj='E2E', pre=high, post=bind1, w=0.7, delay=HB),
        dict(layer_post=layer, proj='E2E', pre=high, post=bind2, w=0.7, delay=HB),

        # L -> B1, B2: delays all outside [lbs, ubs]
        dict(layer_post=layer, proj='FF', pre=pre, post=bind1, w=0.8, delay=lbs - 1.0),
        dict(layer_post=layer, proj='FF', pre=pre, post=bind2, w=0.8, delay=ubs + 1.0),
    ])

    idx = get_structural_indices(nrn=high, layer=layer, syn_params=syn_params,
                                 w_min=0.5, tol=tol)

    # No valid binding neurons => only pre/high candidate sets are empty
    assert len(idx) == 0


def test_indices_depend_on_select_neuron():
    """
    Two different high-level neurons H1 and H2 on the same layer share the same
    presynaptic neuron and binding neuron. Calling get_structural_indices with
    nrn=H1 vs nrn=H2 should center the candidates on that select neuron, but
    both should include the same presyn/bind neurons.
    """
    layer = 4
    pre = 10
    high1, high2 = 20, 21
    bind = 30

    LH1 = 2.0
    LH2 = 2.5
    HB1 = 3.0
    HB2 = 3.5
    tol = 1.0

    syn_params = _make_syn_params([
        # L -> H1, H2
        dict(layer_post=layer, proj='FF', pre=pre, post=high1, w=0.6, delay=LH1),
        dict(layer_post=layer, proj='FF', pre=pre, post=high2, w=0.6, delay=LH2),

        # H1/H2 -> B
        dict(layer_post=layer, proj='E2E', pre=high1, post=bind, w=0.7, delay=HB1),
        dict(layer_post=layer, proj='E2E', pre=high2, post=bind, w=0.7, delay=HB2),

        # L -> B delays chosen to be within both composite windows
        dict(layer_post=layer, proj='FF', pre=pre, post=bind, w=0.8, delay=LH1 + HB1),
    ])

    idx1 = get_structural_indices(nrn=high1, layer=layer, syn_params=syn_params,
                                  w_min=0.5, tol=tol)
    idx2 = get_structural_indices(nrn=high2, layer=layer, syn_params=syn_params,
                                  w_min=0.5, tol=tol)

    expected1 = get_midx([layer - 1, layer, layer], [pre, high1, bind])
    expected2 = get_midx([layer - 1, layer, layer], [pre, high2, bind])

    assert set(idx1) == set(expected1)
    assert set(idx2) == set(expected2)


def test_direct_route_on_exact_bounds_is_included():
    """
    L->B delay equal exactly to lower or upper bound of [lbs, ubs] should be
    treated as valid (inclusive bounds).
    """
    layer = 4
    pre = 10
    high = 20
    bind_low, bind_high = 30, 31

    LH = 3.0
    HB = 4.0
    tol = 1.0
    sum_ = LH + HB  # 7.0
    lbs = sum_ - tol      # 6.0
    ubs = sum_ + 2 * tol  # 9.0

    syn_params = _make_syn_params([
        # L -> H
        dict(layer_post=layer, proj='FF', pre=pre, post=high, w=0.6, delay=LH),

        # H -> both bindings
        dict(layer_post=layer, proj='E2E', pre=high, post=bind_low, w=0.7, delay=HB),
        dict(layer_post=layer, proj='E2E', pre=high, post=bind_high, w=0.7, delay=HB),

        # L -> B on exact bounds
        dict(layer_post=layer, proj='FF', pre=pre, post=bind_low, w=0.8, delay=lbs),
        dict(layer_post=layer, proj='FF', pre=pre, post=bind_high, w=0.8, delay=ubs),
    ])

    idx = get_structural_indices(nrn=high, layer=layer, syn_params=syn_params,
                                 w_min=0.5, tol=tol)

    expected = get_midx(
        [layer - 1, layer, layer, layer],
        [pre, high, bind_low, bind_high],
    )
    assert set(idx) == set(expected)
