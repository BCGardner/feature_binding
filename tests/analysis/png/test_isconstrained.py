import numpy as np
import pandas as pd

from hsnn.analysis.png.base import PNG
from hsnn.analysis.png._utils import isconstrained

TOL = 1.0
W_MIN = 0.5

def _make_syn_params(rows):
    """
    rows: list of dicts with keys: layer_post, proj, pre, post, w, delay
    """
    df = pd.DataFrame(rows)
    df.set_index(['layer_post', 'proj', 'pre', 'post'], inplace=True)
    df.index.rename(['layer', 'proj', 'pre', 'post'], inplace=True)
    return df[['w', 'delay']]


def _make_png(layers, nrns, lags):
    return PNG(np.array(layers), np.array(nrns), np.array(lags, dtype=float), times=np.array([]))


def test_valid_triad():
    # low (L), high (H), bind (B)
    # lags: low=0, high=2, bind=5
    # delays match exactly; composite: LB = LH + HB = 2 + 3 = 5
    png = _make_png([3, 4, 4], [10, 20, 30], [0.0, 2.0, 5.0])
    syn_params = _make_syn_params([
        dict(layer_post=4, proj='FF', pre=10,
             post=20, w=0.6, delay=2.0),  # L->H
        dict(layer_post=4, proj='FF', pre=10,
             post=30, w=0.7, delay=5.0),  # L->B
        dict(layer_post=4, proj='E2E', pre=20,
             post=30, w=0.8, delay=3.0),  # H->B
    ])
    assert isconstrained(png, syn_params, w_min=W_MIN, tol=TOL)


def test_weight_below_threshold():
    png = _make_png([3, 4, 4], [11, 21, 31], [0.0, 2.0, 5.0])
    syn_params = _make_syn_params([
        dict(layer_post=4, proj='FF', pre=11, post=21,
             w=0.49, delay=2.0),  # below w_min
        dict(layer_post=4, proj='FF', pre=11, post=31, w=0.7, delay=5.0),
        dict(layer_post=4, proj='E2E', pre=21, post=31, w=0.8, delay=3.0),
    ])
    assert not isconstrained(png, syn_params, w_min=W_MIN, tol=TOL)


def test_presyn_arrival_after_post_spike():
    # diff < delay (2 < 3) -> early post relative to arrival -> reject
    png = _make_png([3, 4], [12, 22], [0.0, 2.0])
    syn_params = _make_syn_params([
        dict(layer_post=4, proj='FF', pre=12, post=22, w=0.6, delay=3.0),
    ])
    assert not isconstrained(png, syn_params, w_min=W_MIN, tol=TOL)


def test_rise_time_exceeds_tol():
    # diff-delay = 3 > tol=1
    png = _make_png([3, 4], [13, 23], [0.0, 5.0])
    syn_params = _make_syn_params([
        dict(layer_post=4, proj='FF', pre=13, post=23, w=0.6, delay=2.0),
    ])
    assert not isconstrained(png, syn_params, w_min=W_MIN, tol=TOL)


def test_missing_synapse():
    # Missing H->B (E2E) entry
    png = _make_png([3, 4, 4], [14, 24, 34], [0.0, 2.0, 5.0])
    syn_params = _make_syn_params([
        dict(layer_post=4, proj='FF', pre=14, post=24, w=0.6, delay=2.0),
        dict(layer_post=4, proj='FF', pre=14, post=34, w=0.7, delay=5.0),
        # missing E2E 24->34
    ])
    assert not isconstrained(png, syn_params, w_min=W_MIN, tol=TOL)


def test_composite_delay_violation():
    # Pairwise causal, but LB delay too small relative to LH+HB
    # LH=3, HB=3 => sum=6; LB=3; bounds (rhs - tol)=5, (rhs + 2*tol)=8 -> violates triad check
    png = _make_png([3, 4, 4], [15, 25, 35], [0.0, 3.0, 6.0])
    syn_params = _make_syn_params([
        dict(layer_post=4, proj='FF', pre=15, post=25, w=0.6, delay=3.0),
        dict(layer_post=4, proj='FF', pre=15,
             post=35, w=0.7, delay=3.0),  # too short
        dict(layer_post=4, proj='E2E', pre=25, post=35, w=0.8, delay=3.0),
    ])
    assert not isconstrained(png, syn_params, w_min=W_MIN, tol=TOL)


def test_layer_gap_gt_one():
    # Layers 3 -> 5 (gap=2) invalid
    png = _make_png([3, 5], [16, 26], [0.0, 4.0])
    syn_params = _make_syn_params([
        dict(layer_post=5, proj='FF', pre=16, post=26, w=0.6, delay=4.0),
    ])
    assert not isconstrained(png, syn_params, w_min=W_MIN, tol=TOL)


def test_duplicate_neuron_ids():
    # Repeating neuron IDs still evaluated; here valid causal link
    png = _make_png([3, 4, 4], [17, 27, 27], [0.0, 2.0, 5.0])
    syn_params = _make_syn_params([
        dict(layer_post=4, proj='FF', pre=17, post=27, w=0.6, delay=2.0),
        dict(layer_post=4, proj='E2E', pre=27, post=27,
             w=0.9, delay=3.0),  # self intra-layer
    ])
    # Requires only existing pairs; LB not present, triad_delays incomplete -> still True
    assert not isconstrained(png, syn_params, w_min=W_MIN, tol=TOL)
