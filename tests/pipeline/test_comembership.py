# type: ignore
"""Verification tests for the role co-membership matrix (`comembership.py`).

Two hand-checkable in-memory fixtures lock in the cell construction (within-layer H-B
switching and the structural mask; cross-layer switch-up/switch-down and the count
identity), and a gated suite asserts the invariants + a count snapshot on the one
representative-trial annotation table.
"""

import pandas as pd
import pytest

from hsnn.pipeline import reuse
from hsnn.pipeline.reuse import comembership
from hsnn.utils import io

ROLES = list("LHB")

_EXPERIMENT = "n4p2/train_n4p2_lrate_0_02_181023"
_MODEL_TYPE = "ALL"
_CHECKPOINT = -1
_REP_TRIAL = "TrainSNN_1fdbf_00015"  # representative trial, checkpoint -1 (post)


def _annotation(pngs) -> pd.DataFrame:
    """Builds a minimal ``hfb_annotations`` frame from ``(L, H, B)`` (layer, id) tuples.

    Each PNG is ``((l_layer, l_id), (h_layer, h_id), (b_layer, b_id))``; the anchor
    ``layer`` is the high-level neuron's layer.
    """
    rows = []
    for low, high, bind in pngs:
        rows.append({
            "layer": high[0],
            "l_layer": low[0], "l_id": low[1],
            "h_layer": high[0], "h_id": high[1],
            "b_layer": bind[0], "b_id": bind[1],
        })
    return pd.DataFrame(rows)


def _frame(data, dtype=None) -> pd.DataFrame:
    """A 3x3 ``L/H/B``-indexed frame from a row-major nested list."""
    return pd.DataFrame(data, index=ROLES, columns=ROLES, dtype=dtype)


# 5 PNGs over anchor layers 2 and 3 (top_layer = 3):
#
#   p1: anchor 3  L(2,20) H(3,30) B(3,31)
#   p2: anchor 3  L(2,21) H(3,32) B(3,30)   # (3,30) is H in p1 and B in p2 -> H-B switch
#   p3: anchor 2  L(1,10) H(2,20) B(2,22)
#   p4: anchor 2  L(1,11) H(2,23) B(2,24)   # (2,23),(2,24),(1,11) single-role -> diagonals
#   p5: anchor 2  L(1,10) H(2,25) B(2,22)
#
# GLOBAL_H = {(3,30),(3,32),(2,20),(2,23),(2,25)}
# GLOBAL_B = {(3,31),(3,30),(2,22),(2,24)}
# GLOBAL_L = {(2,20),(2,21),(1,10),(1,11)}        (no l_layer == 0 present)
#
# Layer-3 row populations: L={(2,20),(2,21)}, H={(3,30),(3,32)}, B={(3,31),(3,30)}.
#  (3,30) is H in p1 and B in p2; the only other L3 high-level neuron (3,32) is never a
#  binding neuron, so cell(H,B)@3 = 1/2 (and symmetrically (B,H)@3 = 1/2). Both L3
#  high/binding neurons are in the top layer and are never low-level, so the (H,L)/(B,L)
#  low-level column is structurally masked with count 0.

@pytest.fixture
def within_layer_ann():
    return _annotation([
        ((2, 20), (3, 30), (3, 31)),  # p1
        ((2, 21), (3, 32), (3, 30)),  # p2
        ((1, 10), (2, 20), (2, 22)),  # p3
        ((1, 11), (2, 23), (2, 24)),  # p4
        ((1, 10), (2, 25), (2, 22)),  # p5
    ])


def test_within_layer_l3_matrix(within_layer_ann):
    res = comembership.role_comembership(within_layer_ann, 3, top_layer=3)

    expected_counts = _frame([[2, 1, 0],
                              [0, 2, 1],
                              [0, 1, 2]])
    expected_props = _frame([[1.0, 0.5, 0.0],
                            [0.0, 1.0, 0.5],
                            [0.0, 0.5, 1.0]])
    expected_mask = _frame([[False, False, False],
                           [True,  False, False],
                           [True,  False, False]])

    pd.testing.assert_frame_equal(res.counts, expected_counts, check_dtype=False)
    pd.testing.assert_frame_equal(res.proportions, expected_props, check_exact=False)
    pd.testing.assert_frame_equal(res.mask, expected_mask)
    assert res.row_counts == {"L": 2, "H": 2, "B": 2}


def test_within_layer_l2_unmasked_and_diagonals(within_layer_ann):
    # Layer 2 is NOT the top layer (top_layer = 3), so its low-level column is unmasked.
    res = comembership.role_comembership(within_layer_ann, 2, top_layer=3)

    for r in ROLES:
        assert res.proportions.loc[r, r] == 1.0  # diagonal-only neurons -> clean 1.0

    # (H,L): (2,20) is also L of the layer-3 circuit p1 -> count 1 of 3 high-level neurons.
    assert res.counts.loc["H", "L"] == 1
    assert res.proportions.loc["H", "L"] == pytest.approx(1 / 3)
    assert not bool(res.mask.loc["H", "L"])
    # Layer-1 low-level neurons are never high-level/binding here.
    assert res.proportions.loc["L", "H"] == 0.0
    assert res.proportions.loc["L", "B"] == 0.0
    # The whole layer-2 mask is False (nothing structurally impossible below the top layer).
    assert not res.mask.to_numpy().any()


# 9 PNGs over anchor layers 2, 3, 4 (top_layer = 4); the middle (layer-3) matrix carries
# distinct switch-up and switch-down counts:
#
#   q1: anchor 2  L(1,10) H(2,200) B(2,210)
#   q2: anchor 2  L(1,11) H(2,201) B(2,211)
#   q8: anchor 2  L(1,12) H(2,202) B(2,200)   # (2,200) also a B@2 -> (L,B)@3 = 2
#   q3: anchor 3  L(2,200) H(3,300) B(3,310)  # L(2,200) is H@2 (q1) -> (L,H)@3
#   q4: anchor 3  L(2,210) H(3,301) B(3,311)  # L(2,210) is B@2 (q1) -> (L,B)@3
#   q9: anchor 3  L(2,211) H(3,302) B(3,312)  # extra L3 H/B never low-level (size-3 rows)
#   q5: anchor 4  L(3,300) H(4,400) B(4,410)  # L(3,300) is H@3 (q3) -> (H,L)@3
#   q6: anchor 4  L(3,310) H(4,401) B(4,411)  # L(3,310) is B@3 (q3) -> (B,L)@3
#   q7: anchor 4  L(3,301) H(4,402) B(4,412)  # L(3,301) is H@3 (q4) -> 2nd (H,L)@3
#
# Layer-3 row populations: L={(2,200),(2,210),(2,211)}, H={(3,300),(3,301),(3,302)},
#                          B={(3,310),(3,311),(3,312)} (all size 3).
#   switch-up (H,L)@3 = 2/3: H-row ∩ GLOBAL_L = {(3,300),(3,301)}.
#   switch-down (L,H)@3 = 1/3: L-row ∩ GLOBAL_H = {(2,200)}.
#   (B,L)@3 = 1/3 ({(3,310)}); (L,B)@3 = 3/3 (all of (2,200),(2,210),(2,211) are binding
#   neurons of layer-2 circuits: (2,200) via q8, (2,210) via q1, (2,211) via q2).
#   The switch-up/switch-down counts (2 vs 1) and the B-family (1 vs 3) are distinct.

@pytest.fixture
def cross_layer_ann():
    return _annotation([
        ((1, 10), (2, 200), (2, 210)),  # q1
        ((1, 11), (2, 201), (2, 211)),  # q2
        ((1, 12), (2, 202), (2, 200)),  # q8
        ((2, 200), (3, 300), (3, 310)),  # q3
        ((2, 210), (3, 301), (3, 311)),  # q4
        ((2, 211), (3, 302), (3, 312)),  # q9
        ((3, 300), (4, 400), (4, 410)),  # q5
        ((3, 310), (4, 401), (4, 411)),  # q6
        ((3, 301), (4, 402), (4, 412)),  # q7
    ])


def test_cross_layer_middle_matrix_distinct_switches(cross_layer_ann):
    res = comembership.role_comembership(cross_layer_ann, 3, top_layer=4)

    # switch-up (H,L) and switch-down (L,H) take distinct hand-computed values.
    assert res.counts.loc["H", "L"] == 2
    assert res.proportions.loc["H", "L"] == pytest.approx(2 / 3)
    assert res.counts.loc["L", "H"] == 1
    assert res.proportions.loc["L", "H"] == pytest.approx(1 / 3)

    # the B/(L,B) family is also distinct.
    assert res.counts.loc["B", "L"] == 1
    assert res.proportions.loc["B", "L"] == pytest.approx(1 / 3)
    assert res.counts.loc["L", "B"] == 3
    assert res.proportions.loc["L", "B"] == pytest.approx(1.0)


def test_cross_layer_count_identity(cross_layer_ann):
    res = {l: comembership.role_comembership(cross_layer_ann, l, top_layer=4)
           for l in (2, 3, 4)}
    for l in (2, 3):
        assert res[l].counts.loc["H", "L"] == res[l + 1].counts.loc["L", "H"]
        assert res[l].counts.loc["B", "L"] == res[l + 1].counts.loc["L", "B"]
    # explicit hand values: (H,L)@2==(L,H)@3==1; (B,L)@2==(L,B)@3==3;
    #                       (H,L)@3==(L,H)@4==2; (B,L)@3==(L,B)@4==1.
    assert res[2].counts.loc["H", "L"] == 1
    assert res[2].counts.loc["B", "L"] == 3
    assert res[3].counts.loc["H", "L"] == 2
    assert res[3].counts.loc["B", "L"] == 1


def test_cross_layer_top_mask(cross_layer_ann):
    res = comembership.role_comembership(cross_layer_ann, 4, top_layer=4)
    assert bool(res.mask.loc["H", "L"]) and bool(res.mask.loc["B", "L"])
    assert res.counts.loc["H", "L"] == 0 and res.counts.loc["B", "L"] == 0
    # everything else False.
    others = res.mask.copy()
    others.loc[["H", "B"], "L"] = False
    assert not others.to_numpy().any()


@pytest.mark.parametrize("fixture, anchor_layers, top_layer", [
    ("within_layer_ann", (2, 3), 3),
    ("cross_layer_ann", (2, 3, 4), 4),
])
def test_role_reuse_counts_reconciles_with_row_population(
        request, fixture, anchor_layers, top_layer):
    # The reuse-count distinct value must equal the role's row-population size at the same
    # anchor layer (the reconciliation the persisted table used to back), and the reuse
    # factor must be N_l / distinct.
    ann = request.getfixturevalue(fixture)
    rrc = comembership.role_reuse_counts(ann, layers=anchor_layers).set_index(["layer", "role"])
    for layer in anchor_layers:
        res = comembership.role_comembership(ann, layer, top_layer=top_layer)
        n_pngs = int((ann["layer"] == layer).sum())
        for role in ROLES:
            row = rrc.loc[(layer, role)]
            assert int(row["distinct_count"]) == res.row_counts[role]
            assert int(row["n_pngs"]) == n_pngs
            if row["distinct_count"]:
                assert row["reuse_factor"] == pytest.approx(n_pngs / row["distinct_count"])


def test_role_reuse_counts_explicit(within_layer_ann):
    # Hand-checked values for the within-layer fixture (anchor layers 2 and 3).
    rrc = comembership.role_reuse_counts(within_layer_ann, layers=(2, 3)).set_index(["layer", "role"])
    # Layer 2: 3 PNGs; distinct L={10,11}=2, H={20,23,25}=3, B={22,24}=2.
    assert rrc.loc[(2, "L"), "n_pngs"] == 3
    assert rrc.loc[(2, "L"), "distinct_count"] == 2
    assert rrc.loc[(2, "L"), "reuse_factor"] == pytest.approx(1.5)
    assert rrc.loc[(2, "H"), "distinct_count"] == 3
    assert rrc.loc[(2, "B"), "distinct_count"] == 2
    # Layer 3: 2 PNGs; each role has 2 distinct neurons -> reuse factor 1.0.
    for role in ROLES:
        assert rrc.loc[(3, role), "n_pngs"] == 2
        assert rrc.loc[(3, role), "distinct_count"] == 2
        assert rrc.loc[(3, role), "reuse_factor"] == pytest.approx(1.0)


_ANCHOR_LAYERS = (2, 3, 4)
_TOP_LAYER = 4

# Snapshot: integer comembership_count per (anchor_layer, role_from,
# role_to), pinned from role_switching.csv for the
# representative trial TrainSNN_1fdbf_00015, checkpoint -1.
_EXPECTED_COUNTS = {
    (2, "L", "L"): 532,  (2, "L", "H"): 344,  (2, "L", "B"): 336,
    (2, "H", "L"): 1150, (2, "H", "H"): 1369, (2, "H", "B"): 976,
    (2, "B", "L"): 1106, (2, "B", "H"): 976,  (2, "B", "B"): 1321,
    (3, "L", "L"): 1529, (3, "L", "H"): 1150, (3, "L", "B"): 1106,
    (3, "H", "L"): 1386, (3, "H", "H"): 1657, (3, "H", "B"): 1103,
    (3, "B", "L"): 1256, (3, "B", "H"): 1103, (3, "B", "B"): 1501,
    (4, "L", "L"): 1975, (4, "L", "H"): 1386, (4, "L", "B"): 1256,
    (4, "H", "L"): 0,    (4, "H", "H"): 1725, (4, "H", "B"): 1114,
    (4, "B", "L"): 0,    (4, "B", "H"): 1114, (4, "B", "B"): 1523,
}


@pytest.fixture(scope="module")
def real_data():
    if not (io.EXPT_DIR / _EXPERIMENT).exists():
        pytest.skip("N4P2 experiment data not available.")
    store = reuse.open_store(_EXPERIMENT, _MODEL_TYPE, checkpoint=_CHECKPOINT)
    try:
        ann = reuse.load_tables(store)["hfb_annotations"]
    except FileNotFoundError as exc:
        pytest.skip(f"Persisted reuse tables not available: {exc}")
    # Reconciliation counts are recomputed from the annotation table (the same trivial
    # aggregation the figure notebooks use), so no separate persisted artifact is needed.
    rrc = comembership.role_reuse_counts(ann, layers=_ANCHOR_LAYERS)
    return ann, rrc


@pytest.fixture(scope="module")
def real_results(real_data):
    ann, _ = real_data
    return {l: comembership.role_comembership(ann, l, top_layer=_TOP_LAYER)
            for l in _ANCHOR_LAYERS}


def test_real_integrality(real_results):
    for layer, res in real_results.items():
        for r1 in ROLES:
            for r2 in ROLES:
                product = res.proportions.loc[r1, r2] * res.row_counts[r1]
                assert product == pytest.approx(res.counts.loc[r1, r2], abs=1e-6), \
                    f"L{layer} ({r1},{r2}): proportion*row_pop != count"


def test_real_partial_symmetry(real_results):
    # Within-layer H/B co-membership is symmetric; do NOT assert full transpose symmetry.
    for layer, res in real_results.items():
        assert res.counts.loc["H", "B"] == res.counts.loc["B", "H"], f"L{layer} H/B"


def test_real_cross_layer_count_identity(real_results):
    for l in (2, 3):
        assert real_results[l].counts.loc["H", "L"] == \
            real_results[l + 1].counts.loc["L", "H"], f"(H,L)@{l} != (L,H)@{l+1}"
        assert real_results[l].counts.loc["B", "L"] == \
            real_results[l + 1].counts.loc["L", "B"], f"(B,L)@{l} != (L,B)@{l+1}"


def test_real_diagonal(real_results):
    for layer, res in real_results.items():
        for r in ROLES:
            assert res.proportions.loc[r, r] == 1.0, f"L{layer} diagonal {r}"
            assert res.counts.loc[r, r] == res.row_counts[r], f"L{layer} diagonal {r}"


def test_real_structural_mask(real_results):
    for layer, res in real_results.items():
        if layer == _TOP_LAYER:
            for r in ("H", "B"):
                assert bool(res.mask.loc[r, "L"])
                assert res.counts.loc[r, "L"] == 0
                assert res.proportions.loc[r, "L"] == 0.0
            others = res.mask.copy()
            others.loc[["H", "B"], "L"] = False
            assert not others.to_numpy().any()
        else:
            assert not res.mask.to_numpy().any(), f"L{layer} should have no masked cells"


def test_real_reconciliation(real_data, real_results):
    _, rrc = real_data
    for layer, res in real_results.items():
        for role in ROLES:
            expected = int(rrc.query("layer == @layer and role == @role")
                           ["distinct_count"].iloc[0])
            assert res.row_counts[role] == expected, \
                f"L{layer} {role}: row_pop {res.row_counts[role]} != distinct {expected}"


def test_real_cross_task_reuse_factor_identity(real_data, real_results):
    # cond(H,B)/cond(B,H) == rf_H/rf_B (Task 1 reuse factors), within tolerance.
    _, rrc = real_data

    def rf(layer, role):
        return float(rrc.query("layer == @layer and role == @role")
                     ["reuse_factor"].iloc[0])

    for layer, res in real_results.items():
        lhs = res.proportions.loc["H", "B"] / res.proportions.loc["B", "H"]
        rhs = rf(layer, "H") / rf(layer, "B")
        assert lhs == pytest.approx(rhs, rel=1e-6), f"L{layer} reuse-factor identity"


def test_real_count_snapshot(real_results):
    for (layer, r1, r2), expected in _EXPECTED_COUNTS.items():
        got = int(real_results[layer].counts.loc[r1, r2])
        assert got == expected, f"snapshot L{layer} ({r1},{r2}): {got} != {expected}"
