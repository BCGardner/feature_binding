# type: ignore
"""Verification tests for low-level feature completeness / coverage (`completeness.py`).

A tiny, fully hand-worked static fixture (five layer-1 neurons forming the informative pool, and
seven anchor-layer-2 PNGs) makes every recruitment count, coverage figure and any-vs-matched
agreement checkable by eye.

Fixture layout (anchor layer 2; the L neuron lives in layer 1):

    neuron_information (layer-1 pool source):
        id  layer  info_bits  pref_side  pref_conf   feature_label   informative@2/3  @0.5
        10    1       1.0       left      convex      left-convex          yes          yes
        11    1       0.8       left      convex      left-convex          yes          yes
        12    1       0.5       right     concave     right-concave        NO           yes
        13    1       0.9       right     concave     right-concave        yes          yes
        14    1       0.1       left      convex      left-convex          NO           NO

    hfb_annotations (anchor layer 2; l_id is the layer-1 L neuron):
        #  l_id  png_pref_label   f1     note
        1   10   left-convex     0.95    high F1, matches neuron 10's own label
        2   11   right-concave   0.75    mismatch (neuron 11 prefers left-convex)
        3   13   right-concave   0.60    low F1, matches neuron 13's own label
        4   10   left-convex     0.40    low F1, reuse of neuron 10 (collapses)
        5   99   left-convex     0.99    l_id 99 is not in the pool -> never recruited
        6   12   right-concave   0.80    neuron 12 is informative only at the 0.5 threshold
        7   11   (unlabelled)    0.95    no png_pref_label -> excluded from every tier
"""

import numpy as np
import pandas as pd
import pytest

from hsnn.pipeline import reuse
from hsnn.pipeline.reuse import completeness
from hsnn.utils import io

_LABELS = ["left-convex", "right-concave"]


@pytest.fixture
def neuron_information() -> pd.DataFrame:
    """Five layer-1 neurons; ids 10/11/13 are informative at 2/3 bit, 12 only at 0.5 bit."""
    rows = [
        (10, 1, 1.0, "left", "convex"),
        (11, 1, 0.8, "left", "convex"),
        (12, 1, 0.5, "right", "concave"),
        (13, 1, 0.9, "right", "concave"),
        (14, 1, 0.1, "left", "convex"),
    ]
    return pd.DataFrame(rows, columns=["neuron", "layer", "info_bits", "pref_side", "pref_conformation"])


@pytest.fixture
def annotations() -> pd.DataFrame:
    """Seven anchor-layer-2 PNGs (one unlabelled); see the module docstring for the layout."""
    rows = [
        (2, 10, "left-convex", 0.95),
        (2, 11, "right-concave", 0.75),
        (2, 13, "right-concave", 0.60),
        (2, 10, "left-convex", 0.40),
        (2, 99, "left-convex", 0.99),
        (2, 12, "right-concave", 0.80),
        (2, 11, None, 0.95),
    ]
    return pd.DataFrame(rows, columns=["layer", "l_id", "png_pref_label", "png_pref_f1"])


def test_feature_label_and_conformation(neuron_information):
    labels = completeness.feature_label(neuron_information)
    assert list(labels) == ["left-convex", "left-convex", "right-concave", "right-concave", "left-convex"]
    assert completeness.conformation_of("left-convex") == "convex"
    assert list(completeness.conformation_of(labels)) == [
        "convex", "convex", "concave", "concave", "convex"
    ]


def test_informative_pool_threshold(neuron_information):
    # At 2/3 bit the pool is {10, 11, 13}; at 0.5 bit neuron 12 (info_bits == 0.5) joins.
    pool_23 = completeness.informative_pool(neuron_information, anchor_layer=2, info_threshold=2 / 3)
    assert set(pool_23["neuron"]) == {10, 11, 13}
    pool_05 = completeness.informative_pool(neuron_information, anchor_layer=2, info_threshold=0.5)
    assert set(pool_05["neuron"]) == {10, 11, 12, 13}
    # The pool draws from layer anchor-1 only; an anchor of 3 would look at layer 2 (empty here).
    assert completeness.informative_pool(neuron_information, anchor_layer=3, info_threshold=2 / 3).empty


def _counts(res, tier):
    """Per-label (pool, recruited_any, recruited_matched) for a tier, indexed by label."""
    return res.counts[res.counts["tier"] == tier].set_index("label")


def test_recruitment_counts_are_hand_checkable(annotations, neuron_information):
    res = completeness.recruitment_tables(
        annotations, neuron_information, _LABELS, anchor_layers=[2], info_thresholds=[2 / 3]
    )
    # Pool @ 2/3 bit: left-convex = {10, 11} (2); right-concave = {13} (1). Same for every tier.
    for tier in ("structural", "f1>0.5", "f1>0.7", "f1>0.9"):
        c = _counts(res, tier)
        assert c.loc["left-convex", "pool"] == 2
        assert c.loc["right-concave", "pool"] == 1

    # STRUCTURAL (any labelled PNG): recruited L ids in pool = {10, 11, 13}.
    s = _counts(res, "structural")
    assert s.loc["left-convex", "recruited_any"] == 2      # {10, 11}
    assert s.loc["right-concave", "recruited_any"] == 1    # {13}
    # Label-matched: own-label PNGs are #1 (10->left-convex) and #3 (13->right-concave); #2 is a
    # mismatch, #4 reuses 10, #5/#6 are outside the pool. Matched pool ids = {10, 13}.
    assert s.loc["left-convex", "recruited_matched"] == 1  # {10}
    assert s.loc["right-concave", "recruited_matched"] == 1  # {13}

    # F1 > 0.5 admits PNGs #1, #2, #3, #6 (>0.5); recruited pool ids = {10, 11, 13}.
    c05 = _counts(res, "f1>0.5")
    assert c05.loc["left-convex", "recruited_any"] == 2
    assert c05.loc["right-concave", "recruited_any"] == 1

    # F1 > 0.7 admits #1, #2, #6 (and #5, outside pool); recruited pool ids = {10, 11}.
    c07 = _counts(res, "f1>0.7")
    assert c07.loc["left-convex", "recruited_any"] == 2
    assert c07.loc["right-concave", "recruited_any"] == 0  # 13's only PNG (#3) has F1 0.60

    # F1 > 0.9 admits #1 (and #5, outside pool); recruited pool ids = {10}.
    c09 = _counts(res, "f1>0.9")
    assert c09.loc["left-convex", "recruited_any"] == 1
    assert c09.loc["right-concave", "recruited_any"] == 0


def test_recruitment_is_monotone_across_tiers(annotations, neuron_information):
    res = completeness.recruitment_tables(
        annotations, neuron_information, _LABELS, anchor_layers=[2], info_thresholds=[2 / 3]
    )
    order = ["structural", "f1>0.5", "f1>0.7", "f1>0.9"]
    wide = res.counts.pivot_table(index="label", columns="tier", values="recruited_any")[order]
    assert (np.diff(wide.to_numpy(), axis=1) <= 0).all()
    # And every recruited count is bounded by its pool, with matched <= any.
    assert (res.counts["recruited_any"] <= res.counts["pool"]).all()
    assert (res.counts["recruited_matched"] <= res.counts["recruited_any"]).all()


def test_coverage_drops_at_the_strict_tip(annotations, neuron_information):
    res = completeness.recruitment_tables(
        annotations, neuron_information, _LABELS, anchor_layers=[2], info_thresholds=[2 / 3]
    )
    cov = res.coverage.set_index("tier")
    # Both labels are informative in the pool at every tier.
    assert (cov["cov_informative_neuron"] == 2).all()
    # Structural PNGs carry both labels; at F1 > 0.9 only left-convex survives (#1), so
    # right-concave becomes uncovered -- coverage reads 2 -> 1 with the gap named.
    assert cov.loc["structural", "cov_recruited_label"] == 2
    assert cov.loc["structural", "missing_label"] == ""
    assert cov.loc["f1>0.9", "cov_recruited_label"] == 1
    assert cov.loc["f1>0.9", "missing_label"] == "right-concave"


def test_agreement_any_vs_matched(annotations, neuron_information):
    res = completeness.recruitment_tables(
        annotations, neuron_information, _LABELS, anchor_layers=[2], info_thresholds=[2 / 3]
    )
    ag = res.agreement.set_index("tier")
    # Structural: any = {10, 11, 13} (3), matched = {10, 13} (2) -> agreement 2/3.
    assert ag.loc["structural", "n_recruited_any"] == 3
    assert ag.loc["structural", "n_recruited_matched"] == 2
    assert ag.loc["structural", "agreement"] == pytest.approx(2 / 3)
    # F1 > 0.9: any = matched = {10} -> agreement 1.0.
    assert ag.loc["f1>0.9", "n_recruited_any"] == 1
    assert ag.loc["f1>0.9", "agreement"] == pytest.approx(1.0)


def test_lower_info_threshold_enlarges_pool(annotations, neuron_information):
    res = completeness.recruitment_tables(
        annotations, neuron_information, _LABELS,
        anchor_layers=[2], info_thresholds=[2 / 3, 0.5],
    )
    pool = res.counts.groupby("threshold")["pool"].sum() / len(completeness.DEFAULT_TIERS)
    # 2/3 bit: {10, 11, 13} = 3 distinct; 0.5 bit additionally admits neuron 12 -> 4.
    assert pool.loc[2 / 3] == 3
    assert pool.loc[0.5] == 4
    # At 0.5 bit, neuron 12 (right-concave) is now recruited structurally by PNG #6.
    c = res.counts[(np.isclose(res.counts["threshold"], 0.5)) & (res.counts["tier"] == "structural")]
    assert c.set_index("label").loc["right-concave", "recruited_any"] == 2  # {12, 13}


def test_unlabelled_pngs_are_ignored(annotations, neuron_information):
    # Adding more copies of the unlabelled PNG #7 (neuron 11) must not change any count.
    extra = pd.concat([annotations, annotations.iloc[[6]], annotations.iloc[[6]]], ignore_index=True)
    a = completeness.recruitment_tables(annotations, neuron_information, _LABELS, anchor_layers=[2])
    b = completeness.recruitment_tables(extra, neuron_information, _LABELS, anchor_layers=[2])
    pd.testing.assert_frame_equal(a.counts, b.counts)


# --- per-circuit label alignment ------------------------------------------------------


@pytest.fixture
def alignment_annotations() -> pd.DataFrame:
    """Anchor-layer-2 PNGs with per-role label / informativeness columns for the alignment check.

        #  layer  png_pref_label   f1     l_informative  l_pref_label   h_pref_label    note
        1    2     left-convex     0.95      True         left-convex    left-convex     L & H aligned
        2    2     right-concave   0.80      True         left-convex    right-concave   L mismatch, H aligned
        3    2     left-convex     0.60      True         left-convex    left-convex     below F1 > 0.7
        4    2     left-convex     0.99      False        left-convex    left-convex     L uninformative
        5    2     (unlabelled)    0.99      True         left-convex    left-convex     no png_pref_label
        6    3     left-convex     0.99      True         left-convex    right-concave   other anchor layer
    """
    rows = [
        (2, "left-convex", 0.95, True, "left-convex", "left-convex"),
        (2, "right-concave", 0.80, True, "left-convex", "right-concave"),
        (2, "left-convex", 0.60, True, "left-convex", "left-convex"),
        (2, "left-convex", 0.99, False, "left-convex", "left-convex"),
        (2, None, 0.99, True, "left-convex", "left-convex"),
        (3, "left-convex", 0.99, True, "left-convex", "right-concave"),
    ]
    return pd.DataFrame(
        rows,
        columns=["layer", "png_pref_label", "png_pref_f1", "l_informative", "l_pref_label", "h_pref_label"],
    )


def test_circuit_label_alignment(alignment_annotations):
    align = completeness.circuit_label_alignment
    # Anchor 2, F1 > 0.7, informative-L circuits: PNGs #1 and #2 qualify (#3 below the cut, #4
    # uninformative, #5 unlabelled, #6 a different layer). L aligns on #1 only -> 1/2.
    assert align(alignment_annotations, "L", f1_threshold=0.7, anchor_layer=2) == pytest.approx(0.5)
    # Same population, role H: #1 and #2 both align (H prefers each PNG's label) -> 2/2.
    assert align(alignment_annotations, "H", f1_threshold=0.7, anchor_layer=2) == pytest.approx(1.0)
    # Role label is case-insensitive.
    assert align(alignment_annotations, "l", f1_threshold=0.7, anchor_layer=2) == pytest.approx(0.5)
    # Dropping the informativeness restriction admits #4 (uninformative L, aligned):
    # {#1, #4} aligned of {#1, #2, #4} -> 2/3.
    assert align(
        alignment_annotations, "L", f1_threshold=0.7, anchor_layer=2, restrict_informative=None
    ) == pytest.approx(2 / 3)
    # All anchor layers: #6 (layer 3, informative, aligned) joins -> {#1, #6} of {#1, #2, #6} -> 2/3.
    assert align(alignment_annotations, "L", f1_threshold=0.7) == pytest.approx(2 / 3)
    # No circuit clears the cut -> nan.
    assert np.isnan(align(alignment_annotations, "L", f1_threshold=0.99, anchor_layer=2))
    # Unknown role rejected.
    with pytest.raises(ValueError):
        align(alignment_annotations, "X", f1_threshold=0.7)


# --- gated real-data reproduction -----------------------------------------------------

_EXPERIMENTS = {
    "N3P2": "n3p2/train_n3p2_lrate_0_04_181023",
    "N4P2": "n4p2/train_n4p2_lrate_0_02_181023",
}
_MODEL_TYPE = "ALL"


def _real_tables(experiment):
    if not (io.EXPT_DIR / experiment).exists():
        pytest.skip(f"{experiment} experiment data not available.")
    try:
        tabs = reuse.load_tables(reuse.open_store(experiment, _MODEL_TYPE, checkpoint=-1))
    except FileNotFoundError as exc:
        pytest.skip(f"Persisted reuse tables not available: {exc}")
    info = tabs["neuron_information"]
    labels = sorted(completeness.feature_label(info).unique())
    return tabs["hfb_annotations"], info, labels


@pytest.mark.parametrize("experiment", _EXPERIMENTS.values(), ids=_EXPERIMENTS.keys())
def test_real_invariants(experiment):
    ann, info, labels = _real_tables(experiment)
    res = completeness.recruitment_tables(
        ann, info, labels, anchor_layers=[2, 3, 4], info_thresholds=[2 / 3]
    )
    # Bounded fractions / distinct-neuron counts: recruited <= pool, label-matched <= any-label.
    assert (res.counts["recruited_any"] <= res.counts["pool"]).all()
    assert (res.counts["recruited_matched"] <= res.counts["recruited_any"]).all()
    # Coverage stays within [0, n_labels].
    n_lab = len(labels)
    assert res.coverage["cov_informative_neuron"].between(0, n_lab).all()
    assert res.coverage["cov_recruited_label"].between(0, n_lab).all()
    # Recruitment is monotone non-increasing across the funnel tiers.
    order = [name for name, _ in completeness.DEFAULT_TIERS]
    wide = res.counts.pivot_table(
        index=["anchor_layer", "label"], columns="tier", values="recruited_any"
    )[order]
    assert (np.diff(wide.to_numpy(), axis=1) <= 0).all()
