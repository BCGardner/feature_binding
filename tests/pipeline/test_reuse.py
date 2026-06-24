# type: ignore
import numpy as np
import pandas as pd
import pytest

from hsnn.analysis import png
from hsnn.pipeline import reuse
from hsnn.utils import handler


@pytest.fixture
def polygrps():
    png_a = png.PNG(
        layers=np.array([3, 4, 4]), nrns=np.array([10, 20, 30]),
        lags=np.array([0.0, 3.0, 10.0]),
        times=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    )
    png_b = png.PNG(
        layers=np.array([3, 4, 4]), nrns=np.array([11, 21, 31]),
        lags=np.array([0.0, 2.0, 8.0]), times=np.array([1.0, 2.0, 3.0]),
    )
    png_c = png.PNG(  # constituents absent from neuron_information
        layers=np.array([2, 3, 3]), nrns=np.array([40, 41, 42]),
        lags=np.array([0.0, 1.0, 4.0]), times=np.array([1.0, 2.0]),
    )
    return {4: [png_a, png_b], 3: [png_c]}, png_a, png_b, png_c


@pytest.fixture
def png_label_metrics(polygrps):
    _, png_a, png_b, _ = polygrps
    return pd.DataFrame([
        # png_a: one valid candidate (precision > 0.5) -> 'left-convex'.
        {"png_id": hash(png_a), "side": "left", "conformation": "convex",
         "precision": 0.9, "recall": 0.8, "f1": 0.85},
        {"png_id": hash(png_a), "side": "top", "conformation": "convex",
         "precision": 0.4, "recall": 0.9, "f1": 0.55},
        # png_b: no candidate reaches precision > 0.5 -> null preferred label.
        {"png_id": hash(png_b), "side": "left", "conformation": "concave",
         "precision": 0.3, "recall": 0.7, "f1": 0.42},
    ])


@pytest.fixture
def neuron_information():
    return pd.DataFrame([
        {"layer": 3, "neuron": 10, "pref_side": "left", "pref_conformation": "convex",
         "info_bits": 0.9, "informative": True},
        {"layer": 4, "neuron": 20, "pref_side": "left", "pref_conformation": "convex",
         "info_bits": 0.8, "informative": True},
        {"layer": 4, "neuron": 30, "pref_side": "right", "pref_conformation": "concave",
         "info_bits": 0.7, "informative": True},
        {"layer": 3, "neuron": 11, "pref_side": "left", "pref_conformation": "convex",
         "info_bits": 0.5, "informative": False},
        {"layer": 4, "neuron": 21, "pref_side": "top", "pref_conformation": "concave",
         "info_bits": 0.6, "informative": False},
        {"layer": 4, "neuron": 31, "pref_side": "left", "pref_conformation": "convex",
         "info_bits": 0.9, "informative": True},
    ])


def _row(ann, png_obj):
    return ann.set_index("png_id").loc[hash(png_obj)]


def test_infer_num_reps_uses_global_max():
    # Detections span reps 0..9 across PNGs; no single PNG need cover them all.
    png_a = png.PNG(
        layers=np.array([3, 4, 4]), nrns=np.array([1, 2, 3]),
        lags=np.array([0.0, 1.0, 2.0]), times=np.array([1.0, 2.0, 3.0]),
        reps=np.array([0, 3, 5]), imgs=np.array([0, 1, 2]),
    )
    png_b = png.PNG(
        layers=np.array([3, 4, 4]), nrns=np.array([4, 5, 6]),
        lags=np.array([0.0, 1.0, 2.0]), times=np.array([1.0, 2.0]),
        reps=np.array([2, 9]), imgs=np.array([0, 1]),
    )
    assert reuse.infer_num_reps({4: [png_a, png_b]}) == 10


def test_infer_num_reps_requires_occurrences():
    png_a = png.PNG(
        layers=np.array([3, 4, 4]), nrns=np.array([1, 2, 3]),
        lags=np.array([0.0, 1.0, 2.0]), times=np.array([]),
    )
    with pytest.raises(ValueError):
        reuse.infer_num_reps({4: [png_a]})


def test_annotation_structure(polygrps, png_label_metrics, neuron_information):
    by_layer, png_a, png_b, png_c = polygrps
    ann = reuse.build_annotation_table(
        by_layer, png_label_metrics, neuron_information,
        experiment="n4p2/x", model_type="ALL", trial_id="trial0", checkpoint=-1,
    )
    assert len(ann) == 3
    a = _row(ann, png_a)
    assert (a["l_id"], a["h_id"], a["b_id"]) == (10, 20, 30)
    assert (a["l_layer"], a["h_layer"], a["b_layer"]) == (3, 4, 4)
    assert a["layer"] == 4
    assert a["lag_lh"] == 3.0 and a["lag_lb"] == 10.0 and a["lag_hb"] == 7.0
    assert a["span"] == 10.0
    assert a["n_occ"] == 5
    assert a["experiment"] == "n4p2/x" and a["model_type"] == "ALL"


def test_preferred_label_and_alignment(polygrps, png_label_metrics, neuron_information):
    by_layer, png_a, png_b, png_c = polygrps
    ann = reuse.build_annotation_table(
        by_layer, png_label_metrics, neuron_information,
        experiment="n4p2/x", model_type="ALL", trial_id="trial0", checkpoint=-1,
    )
    a, b, c = _row(ann, png_a), _row(ann, png_b), _row(ann, png_c)

    # png_a: highest-F1 candidate with precision > 0.5.
    assert a["png_pref_label"] == "left-convex"
    assert a["png_pref_f1"] == pytest.approx(0.85)
    assert a["l_pref_label"] == "left-convex"
    assert a["h_pref_label"] == "left-convex"
    assert a["b_pref_label"] == "right-concave"
    assert a["l_info"] == pytest.approx(0.9)
    assert a["align_lh"] is True or a["align_lh"] == True  # noqa: E712
    assert a["align_lh_png"] == True  # noqa: E712
    assert a["align_lhb_png"] == False  # noqa: E712

    # png_b: no valid PNG label -> null label and null *_png alignment flags.
    assert pd.isna(b["png_pref_label"])
    assert b["align_lh"] == False  # left-convex vs top-concave  # noqa: E712
    assert pd.isna(b["align_lh_png"])
    assert pd.isna(b["align_lhb_png"])

    # png_c: constituents absent from neuron_information -> null neuron labels/flags.
    assert pd.isna(c["l_pref_label"])
    assert pd.isna(c["align_lh"])


def test_audit_tables(polygrps, png_label_metrics, neuron_information):
    by_layer, png_a, png_b, png_c = polygrps
    ann = reuse.build_annotation_table(
        by_layer, png_label_metrics, neuron_information,
        experiment="n4p2/x", model_type="ALL", trial_id="trial0", checkpoint=-1,
    )
    audit = reuse.audit_tables({
        "hfb_annotations": ann,
        "png_label_metrics": png_label_metrics,
        "neuron_information": neuron_information,
    })
    assert audit.loc[4, "n_pngs"] == 2
    assert audit.loc[3, "n_pngs"] == 1
    assert audit.loc[4, "n_high_neurons"] == 2
    assert bool(audit.loc[4, "neuron_info_present"]) is True


def test_store_table_roundtrip(tmp_path, polygrps, png_label_metrics, neuron_information):
    (tmp_path / "checkpoint_000").mkdir()
    trial = handler.TrialView(tmp_path)
    store = handler.ArtifactStore(trial, ckpt_idx=-1)

    by_layer, *_ = polygrps
    ann = reuse.build_annotation_table(
        by_layer, png_label_metrics, neuron_information,
        experiment="n4p2/x", model_type="ALL", trial_id="trial0", checkpoint=-1,
    )
    tables = {
        "hfb_annotations": ann,
        "png_label_metrics": png_label_metrics,
        "neuron_information": neuron_information,
    }
    reuse.persist_tables(store, tables, overwrite=True)

    def _normalise_nulls(df):
        # parquet reads object-column nulls back as None; unify to NaN.
        return df.where(pd.notna(df), other=np.nan).reset_index(drop=True)

    loaded = reuse.load_tables(store)
    for name in reuse.TABLE_NAMES:
        assert (tmp_path / "checkpoint_000" / reuse.REUSE_SUBDIR /
                f"{name}.parquet").exists()
        pd.testing.assert_frame_equal(
            _normalise_nulls(loaded[name]),
            _normalise_nulls(tables[name]),
            check_like=True,
        )
