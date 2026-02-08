import numpy as np
import pandas as pd
import pytest

from hsnn.analysis.png.base import PNG
from hsnn.analysis.png import refinery
from hsnn.analysis.png._utils import isconstrained
from hsnn.analysis.png.detection import _unique_pngs

# Helpers -----------------------------------------------------------------


def make_png(layers, nrns, lags, times):
    return PNG(np.array(layers), np.array(nrns), np.array(lags, dtype=float), np.array(times, dtype=float))


def make_syn_params(rows):
    """
    rows: list of dict(layer, proj, pre, post, w, delay)
    """
    df = pd.DataFrame(rows)
    df.set_index(['layer', 'proj', 'pre', 'post'], inplace=True)
    return df[['w', 'delay']]


# Common simple syn params: L3->L4 (FF) connections for neurons 1->10, 2->11
SYN_SIMPLE = make_syn_params([
    dict(layer=4, proj='FF', pre=1, post=10, w=0.7, delay=5.0),
    dict(layer=4, proj='FF', pre=2, post=11,
         w=0.4, delay=5.0),  # below threshold
])

# Filtering by hash --------------------------------------------------------


def test_unique_pngs_filters_by_hash_and_keeps_one():
    """
    If two PNGs are equal under PNG.__hash__/__eq__, _unique_pngs should
    return only one of them.
    """
    # Same layers/nrns/lags -> same hash
    png1 = PNG(
        layers=np.array([3, 4, 4]),
        nrns=np.array([10, 20, 30]),
        lags=np.array([0.0, 1.0, 2.0]),
        times=np.array([100.0]),
    )
    png2 = PNG(
        layers=np.array([3, 4, 4]),
        nrns=np.array([10, 20, 30]),
        lags=np.array([0.0, 1.0, 2.0]),
        times=np.array([200.0, 300.0]),
    )

    uniques = _unique_pngs([png1, png2], issorted=False)

    # Only one PNG should remain because hashes are identical
    assert len(uniques) == 1
    # And it should be hash-equal to the originals
    assert uniques[0] == png1
    assert uniques[0] == png2


def test_unique_pngs_prefers_highest_num_occurrences():
    """
    When multiple hash-equal PNGs are present, _unique_pngs should retain
    the one with the largest number of occurrences (len(times)).
    This relies on sorted_pngs being called before set(), so that the
    max-occ PNG is the one that survives in the set.
    """
    # Same structural identity, different occurrence counts
    png_low_occ = PNG(
        layers=np.array([3, 4, 4]),
        nrns=np.array([10, 20, 30]),
        lags=np.array([0.0, 1.0, 2.0]),
        times=np.array([100.0]),  # 1 occurrence
    )
    png_mid_occ = PNG(
        layers=np.array([3, 4, 4]),
        nrns=np.array([10, 20, 30]),
        lags=np.array([0.0, 1.1, 2.1]),
        times=np.array([200.0, 300.0]),  # 2 occurrences
    )
    png_high_occ = PNG(
        layers=np.array([3, 4, 4]),
        nrns=np.array([10, 20, 30]),
        lags=np.array([0.0, 1.2, 2.2]),
        times=np.array([400.0, 500.0, 600.0]),  # 3 occurrences
    )

    # Case 1: low-occ first in the input
    uniques_1 = _unique_pngs([png_low_occ, png_mid_occ, png_high_occ], issorted=False)
    assert len(uniques_1) == 1
    kept_1 = uniques_1[0]
    assert kept_1.num_occ == 3
    assert np.array_equal(kept_1.times, png_high_occ.times)
    assert kept_1 == png_low_occ
    assert kept_1 == png_high_occ

    # Case 2: high-occ first in the input
    uniques_2 = _unique_pngs([png_high_occ, png_mid_occ, png_low_occ], issorted=False)
    assert len(uniques_2) == 1
    kept_2 = uniques_2[0]
    assert kept_2.num_occ == 3

    # Case 3: mid-occ first in the input and not sorted
    uniques_3 = _unique_pngs([png_mid_occ, png_low_occ, png_high_occ], issorted=True)
    assert len(uniques_3) == 1
    kept_3 = uniques_3[0]
    assert kept_3.num_occ == 2

# Constrain and Merge ------------------------------------------------------


def test_merge_constrained_hfb_swapped_roles():
    """Two distinct HFB PNGs with swapped H/B roles are structurally constrained and
    should not be merged, even for tol=3.0 on lags and merge.
    """

    # Synaptic parameters:
    # L = (3,10), H = (4,20), B = (4,30)
    # PNG A: 10 -> 20 -> 30
    # PNG B: 10 -> 30 -> 20
    syn_params = make_syn_params([
        # L -> H, L -> B, H -> B
        dict(layer=4, proj='FF',  pre=10, post=20, w=0.8, delay=1.0),  # L->H_A
        dict(layer=4, proj='FF',  pre=10, post=30, w=0.8, delay=2.0),  # L->B_A
        dict(layer=4, proj='E2E', pre=20, post=30, w=0.8, delay=1.0),  # H->B_A
        # For swapped roles (B as high, H as bind) we also need:
        dict(layer=4, proj='E2E', pre=30, post=20, w=0.8, delay=1.0),  # H->B_B
    ])

    # PNG A: 10 (L3) -> 20 (H) -> 30 (B)
    png_a = make_png(
        layers=[3, 4, 4],
        nrns=[10, 20, 30],
        lags=[0, 1, 2],
        times=[100, 200],
    )

    # PNG B: 10 (L3) -> 30 (H) -> 20 (B)  (swapped H/B roles)
    png_b = make_png(
        layers=[3, 4, 4],
        nrns=[10, 30, 20],
        lags=[0, 2, 3],
        times=[300],
    )

    # Sanity: both should be constrained individually
    assert isconstrained(png_a, syn_params, w_min=0.5, tol=3.0)
    assert isconstrained(png_b, syn_params, w_min=0.5, tol=3.0)

    pipeline = refinery.Compose([
        refinery.Constrained(syn_params, w_min=0.5, tol=3.0),
        refinery.Merge(tol=3.0, strategy="mean"),
    ])

    result = pipeline([png_a, png_b])

    # This two PNGs should not get merged due to different firing orders
    assert len(result) == 2

# DropRepeating ------------------------------------------------------------


def test_droprepeating():
    png_ok = make_png([3, 4], [1, 10], [0.0, 5.0], [100.0])
    png_repeat = make_png([3, 4, 4], [1, 10, 10], [0.0, 5.0, 8.0], [120.0])
    out = refinery.DropRepeating()([png_ok, png_repeat])
    assert png_ok in out
    assert png_repeat not in out

def test_droprepeating_layers():
    png_ok = make_png([3, 4, 4], [1, 1, 10], [0.0, 5.0, 8.0], [120.0])
    out = refinery.DropRepeating()([png_ok])
    assert png_ok in out

# FilterLayers -------------------------------------------------------------


def test_filterlayers():
    png_match = make_png([3, 4, 4], [1, 10, 11], [0.0, 5.0, 8.0], [10.0])
    png_other = make_png([2, 3, 4], [5, 6, 7], [0.0, 4.0, 9.0], [5.0])
    out = refinery.FilterLayers([3, 4, 4])([png_match, png_other])
    assert out == [png_match]

# FilterIndex --------------------------------------------------------------


def test_filterindex_any_position():
    png = make_png([3, 4, 4], [1, 10, 11], [0.0, 5.0, 8.0], [10.0])
    out = refinery.FilterIndex((4, 10))([png])
    assert out == [png]


def test_filterindex_with_position_pass():
    png = make_png([3, 4, 4], [1, 10, 11], [0.0, 5.0, 8.0], [10.0])
    out = refinery.FilterIndex((4, 10), position=1)([png])
    assert out == [png]


def test_filterindex_with_position_fail():
    png = make_png([3, 4, 4], [1, 10, 11], [0.0, 5.0, 8.0], [10.0])
    out = refinery.FilterIndex((3, 1), position=1)([png])
    assert out == []

# Constrained --------------------------------------------------------------


def test_constrained_pass_two_neuron_ff():
    png = make_png([3, 4], [1, 10], [0.0, 5.0], [50.0, 75.0])
    ref = refinery.Constrained(SYN_SIMPLE, w_min=0.5, tol=1.0)
    out = ref([png])
    assert out == [png]
    assert isconstrained(png, SYN_SIMPLE, w_min=0.5, tol=1.0)


def test_constrained_fail_weight():
    png = make_png([3, 4], [2, 11], [0.0, 5.0], [60.0])
    ref = refinery.Constrained(SYN_SIMPLE, w_min=0.5, tol=1.0)
    out = ref([png])
    assert out == []


def test_constrained_fail_rise_time():
    # delay=5, diff=3 (lags 0,3) => diff < delay (rise_time negative)
    syn_params = make_syn_params([
        dict(layer=4, proj='FF', pre=1, post=10, w=0.8, delay=5.0)
    ])
    png = make_png([3, 4], [1, 10], [0.0, 3.0], [20.0])
    ref = refinery.Constrained(syn_params, w_min=0.5, tol=2.0)
    out = ref([png])
    assert out == []

# Merge --------------------------------------------------------------------


def test_merge_mode_combines_times():
    png1 = make_png([3, 4], [1, 10], [0.0, 5.0], [10.0, 30.0])
    png2 = make_png([3, 4], [1, 10], [0.0, 5.0], [15.0])
    merged = refinery.Merge(tol=0.5, strategy='mode')([png1, png2])
    assert len(merged) == 1
    assert len(merged[0].times) == 3  # times concatenated (order irrelevant)


def test_merge_mean_weighted_average():
    png1 = make_png([3, 4], [1, 10], [0.0, 5.0], [10.0, 12.0])  # weight=2
    png2 = make_png([3, 4], [1, 10], [0.0, 6.0], [14.0])        # weight=1
    merged = refinery.Merge(tol=2.0, strategy='mean')([png1, png2])
    assert len(merged) == 1
    expected_lag2 = (5.0*2 + 6.0*1)/3
    assert np.isclose(merged[0].lags[1], np.round(expected_lag2))
    # Times merged respecting min_sep=max(5.333...,6)=6 => second PNG time dropped
    assert merged[0].times.tolist() == [10.0, 12.0]


def test_merge_no_merge_outside_tol():
    png1 = make_png([3, 4], [1, 10], [0.0, 5.0], [10.0])
    png2 = make_png([3, 4], [1, 10], [0.0, 9.0], [12.0])
    merged = refinery.Merge(tol=1.0)([png1, png2])
    assert len(merged) == 2

# Compose ------------------------------------------------------------------


def test_compose_pipeline():
    png_valid = make_png([3, 4], [1, 10], [0.0, 5.0], [10.0])
    png_repeat = make_png([3, 4, 4], [1, 10, 10], [0.0, 5.0, 8.0], [15.0])
    ref = refinery.Compose([
        refinery.DropRepeating(),
        refinery.FilterLayers([3, 4])
    ])
    out = ref([png_valid, png_repeat])
    assert out == [png_valid]

# Edge: empty input --------------------------------------------------------


def test_all_refines_empty_input():
    refines = [
        refinery.DropRepeating(),
        refinery.FilterLayers([3, 4]),
        refinery.FilterIndex((3, 1)),
        refinery.Merge(),
        refinery.PassThrough()
    ]
    for ref in refines:
        assert ref([]) == []

# PassThrough --------------------------------------------------------------


def test_passthrough_identity():
    pngs = [make_png([3, 4], [1, 10], [0.0, 5.0], [1.0])]
    out = refinery.PassThrough()(pngs)
    assert out == pngs
