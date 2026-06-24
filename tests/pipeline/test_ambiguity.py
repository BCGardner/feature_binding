# type: ignore
"""Verification tests for role ambiguity / resolution (`ambiguity.py`).

A symmetric synthetic fixture populates the ``l_*`` / ``h_*`` / ``b_*`` columns identically so
the derived quantities (distinct-label count, the three-way match category, the per-layer
divergence/match fraction and the high-F1 mismatch fractions) are hand-checkable for each of
the three circuit roles, exercised by parameterised tests. A gated suite asserts the
partition/range invariants on the representative-trial annotation table for the binder.
"""

import numpy as np
import pandas as pd
import pytest

from hsnn.pipeline import reuse
from hsnn.pipeline.reuse import ambiguity
from hsnn.utils import io

_EXPERIMENT = "n4p2/train_n4p2_lrate_0_02_181023"
_MODEL_TYPE = "ALL"
_CHECKPOINT = -1

_ROLES = ("L", "H", "B")
_ROLE_NAME = {"L": "low", "H": "high", "B": "binder"}


@pytest.fixture
def symmetric_ann() -> pd.DataFrame:
    """Six PNGs (one unlabelled) with the same hand-checkable layout for every role.

    Each row's per-role columns (``{prefix}_layer``, ``{prefix}_id``, ``{prefix}_pref_label``,
    ``{prefix}_informative``) carry the same scenario, with the low-level neuron one layer
    below the anchor. Neuron ``id=30`` anchors two PNGs of *different* labels (distinct count
    2); ``id=31`` is uninformative; ``id=40`` and ``id=41`` are an informative match and an
    informative mismatch at the next anchor layer. The unlabelled PNG shares ``id=31`` and must
    not perturb the labelled-PNG categories or counts. Because the layout is identical across
    roles, the expected values are the same for L, H and B (only the category *labels* differ).
    """
    # (anchor_layer, neuron_id, informative, role_pref_label, png_pref_label, f1)
    scenario = [
        (2, 30, True, "left-convex", "left-convex", 0.95),
        (2, 30, True, "left-convex", "left-concave", 0.85),
        (2, 31, False, "right-convex", "right-convex", 0.60),
        (3, 40, True, "top-convex", "top-convex", 0.92),
        (3, 41, True, "bottom-concave", "top-concave", 0.95),
        (2, 31, False, "right-convex", None, np.nan),
    ]
    records = []
    for anchor, nid, informative, role_pref, png, f1 in scenario:
        record = dict(trial_id="T", layer=anchor, png_pref_label=png, png_pref_f1=f1)
        for role, prefix in (("L", "l"), ("H", "h"), ("B", "b")):
            record[f"{prefix}_layer"] = anchor - 1 if role == "L" else anchor
            record[f"{prefix}_id"] = nid
            record[f"{prefix}_pref_label"] = role_pref
            record[f"{prefix}_informative"] = informative
        records.append(record)
    return pd.DataFrame(records)


@pytest.mark.parametrize("role", _ROLES)
def test_distinct_label_count(symmetric_ann, role):
    counts = ambiguity.distinct_label_count(symmetric_ann, role)
    # id=30 anchors {left-convex, left-concave} -> 2 for both its rows; id=31 anchors
    # {right-convex} (the unlabelled sibling adds nothing) -> 1; id=40 and id=41 -> 1 each.
    assert list(counts) == [2, 2, 1, 1, 1, 1]


@pytest.mark.parametrize("role", _ROLES)
def test_match_categories_labels(role):
    name = _ROLE_NAME[role]
    assert ambiguity.match_categories(role) == (
        f"{name}_matched",
        f"{name}_mismatched",
        f"{name}_uninformative",
    )


def test_match_categories_default_is_binder():
    assert ambiguity.match_categories() == ambiguity.match_categories("B")


@pytest.mark.parametrize("role", _ROLES)
def test_match_category_partition(symmetric_ann, role):
    matched, mismatched, uninformative = ambiguity.match_categories(role)
    cats = ambiguity.match_category(symmetric_ann, role)
    assert list(cats[:5]) == [matched, mismatched, uninformative, matched, mismatched]
    assert pd.isna(cats.iloc[5])  # unlabelled PNG -> unassigned
    labelled = symmetric_ann["png_pref_label"].notna()
    counts = cats[labelled].value_counts()
    assert counts.sum() == int(labelled.sum())  # partition, no overlap
    assert counts[matched] == 2 and counts[mismatched] == 2
    assert counts[uninformative] == 1


@pytest.mark.parametrize("role", _ROLES)
def test_label_divergence_fraction_ungated(symmetric_ann, role):
    div = ambiguity.label_divergence_fraction(symmetric_ann, role).set_index("layer")
    # Layer 2: left-convex(T), left-concave(F), right-convex(T) -> 2/3; n=3.
    assert div.loc[2, "match_fraction"] == pytest.approx(2 / 3)
    assert div.loc[2, "n_pngs"] == 3
    # Layer 3: top-convex(T), bottom-concave vs top-concave(F) -> 1/2; n=2.
    assert div.loc[3, "match_fraction"] == pytest.approx(0.5)
    assert div.loc[3, "n_pngs"] == 2


@pytest.mark.parametrize("role", _ROLES)
def test_label_divergence_fraction_informative_gated(symmetric_ann, role):
    div = ambiguity.label_divergence_fraction(
        symmetric_ann, role, require_informative=True
    ).set_index("layer")
    # Layer 2 drops the uninformative right-convex PNG: left-convex(T), left-concave(F) -> 1/2.
    assert div.loc[2, "match_fraction"] == pytest.approx(0.5)
    assert div.loc[2, "n_pngs"] == 2
    assert div.loc[3, "match_fraction"] == pytest.approx(0.5)


@pytest.mark.parametrize("role", _ROLES)
def test_high_f1_mismatch_fractions(symmetric_ann, role):
    name = _ROLE_NAME[role]
    hf = ambiguity.high_f1_mismatch_fractions(symmetric_ann, role=role).set_index(
        ["threshold", "layer"]
    )
    # >=0.9: {0.95 matched, 0.92 matched, 0.95 mismatched} -> n=3, 1 mismatched.
    assert hf.loc[(0.9, "overall"), "n_high_f1"] == 3
    assert hf.loc[(0.9, "overall"), f"n_{name}_mismatched"] == 1
    assert hf.loc[(0.9, "overall"), f"frac_{name}_mismatched"] == pytest.approx(1 / 3)
    assert hf.loc[(0.9, "overall"), f"frac_{name}_uninformative"] == pytest.approx(0.0)
    # >=0.8 additionally admits the 0.85 mismatched PNG -> n=4, 2 mismatched.
    assert hf.loc[(0.8, "overall"), "n_high_f1"] == 4
    assert hf.loc[(0.8, "overall"), f"frac_{name}_mismatched"] == pytest.approx(0.5)
    # The uninformative neuron (F1 0.60) never reaches a threshold.
    assert hf.loc[(0.7, "overall"), f"n_{name}_uninformative"] == 0


def test_pool_trials_adds_rows(symmetric_ann):
    pooled = ambiguity.pool_trials([symmetric_ann, symmetric_ann])
    assert len(pooled) == 2 * len(symmetric_ann)
    assert list(pooled.columns) == list(symmetric_ann.columns)


@pytest.mark.parametrize("role", _ROLES)
def test_trial_comparability(symmetric_ann, role):
    other = symmetric_ann.assign(trial_id="U")
    summary = ambiguity.trial_comparability(
        {"T": symmetric_ann, "U": other}, role
    ).set_index("trial_id")
    assert set(summary.index) == {"T", "U"}
    assert summary.loc["T", "n_labelled"] == 5
    assert summary.loc["T", "match_L2"] == pytest.approx(round(2 / 3, 3))
    assert summary.loc["T", "match_L3"] == pytest.approx(0.5)


def test_unknown_role_raises(symmetric_ann):
    with pytest.raises(ValueError, match="Unknown role"):
        ambiguity.match_category(symmetric_ann, "X")


# --- gated real-data invariants -------------------------------------------------------

@pytest.fixture(scope="module")
def real_ann() -> pd.DataFrame:
    if not (io.EXPT_DIR / _EXPERIMENT).exists():
        pytest.skip("N4P2 experiment data not available.")
    store = reuse.open_store(_EXPERIMENT, _MODEL_TYPE, checkpoint=_CHECKPOINT)
    try:
        return reuse.load_tables(store)["hfb_annotations"]
    except FileNotFoundError as exc:
        pytest.skip(f"Persisted reuse tables not available: {exc}")


def test_real_f1_in_unit_interval(real_ann):
    f1 = real_ann["png_pref_f1"].dropna()
    assert f1.min() >= 0.0 and f1.max() <= 1.0


@pytest.mark.parametrize("role", _ROLES)
def test_real_categories_partition_labelled(real_ann, role):
    labelled = real_ann["png_pref_label"].notna()
    cats = ambiguity.match_category(real_ann, role)
    assert cats[labelled].notna().all()
    assert cats[~labelled].isna().all()
    assert set(cats[labelled].unique()) <= set(ambiguity.match_categories(role))
    assert cats[labelled].value_counts().sum() == int(labelled.sum())


@pytest.mark.parametrize("role", _ROLES)
def test_real_distinct_count_at_least_one_for_labelled(real_ann, role):
    labelled = real_ann["png_pref_label"].notna()
    counts = ambiguity.distinct_label_count(real_ann, role)
    assert (counts[labelled] >= 1).all()
