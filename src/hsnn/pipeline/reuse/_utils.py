"""Helpers for building the canonical HFB annotation table."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hsnn.analysis.png import PNG

_ROLE_COLUMNS = {
    "l": ("l_layer", "l_id"),
    "h": ("h_layer", "h_id"),
    "b": ("b_layer", "b_id"),
}


def png_record(
    png: PNG, experiment: str, model_type: str, trial_id: str, checkpoint: int
) -> dict:
    """Extracts provenance and structure for a single (lag-ordered) HFB PNG."""
    l_layer, h_layer, b_layer = (int(x) for x in png.layers)
    l_id, h_id, b_id = (int(x) for x in png.nrns)
    lags = png.lags
    return {
        "png_id": hash(png),
        "experiment": experiment,
        "model_type": model_type,
        "trial_id": trial_id,
        "checkpoint": checkpoint,
        "layer": h_layer,
        "l_id": l_id,
        "h_id": h_id,
        "b_id": b_id,
        "l_layer": l_layer,
        "h_layer": h_layer,
        "b_layer": b_layer,
        "lag_lh": float(lags[1] - lags[0]),
        "lag_lb": float(lags[2] - lags[0]),
        "lag_hb": float(lags[2] - lags[1]),
        "span": float(lags.max() - lags.min()),
        "n_occ": int(png.num_occ),
    }


def _label(side: pd.Series, conformation: pd.Series) -> pd.Series:
    """Combines side and conformation into a single feature-label string."""
    return side.astype(str) + "-" + conformation.astype(str)


def join_png_preferred_label(
    ann: pd.DataFrame, png_label_metrics: pd.DataFrame
) -> pd.DataFrame:
    """Adds the preferred PNG label (highest-F1 among precision > 0.5; else null)."""
    candidates = png_label_metrics[png_label_metrics["precision"] > 0.5]
    ann = ann.copy()
    if candidates.empty:
        ann["png_pref_label"] = np.nan
        ann["png_pref_f1"] = np.nan
        ann["png_pref_precision"] = np.nan
        ann["png_pref_recall"] = np.nan
        return ann
    best = candidates.loc[candidates.groupby("png_id")["f1"].idxmax()].copy()
    best["png_pref_label"] = _label(best["side"], best["conformation"])
    best = best.set_index("png_id")[
        ["png_pref_label", "f1", "precision", "recall"]
    ].rename(columns={
        "f1": "png_pref_f1",
        "precision": "png_pref_precision",
        "recall": "png_pref_recall",
    })
    return ann.merge(best, how="left", left_on="png_id", right_index=True)


def join_neuron_information(
    ann: pd.DataFrame, neuron_information: pd.DataFrame
) -> pd.DataFrame:
    """Adds per-role (L/H/B) preferred label, information and informative flag."""
    ann = ann.copy()
    info = neuron_information.copy()
    info["pref_label"] = _label(info["pref_side"], info["pref_conformation"])
    info = info.set_index(["layer", "neuron"])
    for role, (layer_col, id_col) in _ROLE_COLUMNS.items():
        keys = pd.MultiIndex.from_arrays([ann[layer_col], ann[id_col]])
        matched = info.reindex(keys)
        ann[f"{role}_pref_label"] = matched["pref_label"].to_numpy()
        ann[f"{role}_info"] = matched["info_bits"].to_numpy()
        ann[f"{role}_informative"] = matched["informative"].to_numpy()
    return ann


def _all_equal(columns: list[pd.Series]) -> pd.Series:
    """Element-wise equality across columns; null where any element is null."""
    frame = pd.concat([col.reset_index(drop=True) for col in columns], axis=1)
    notna = frame.notna().all(axis=1)
    equal = frame.eq(frame.iloc[:, 0], axis=0).all(axis=1)
    result = equal.astype("object")
    result[~notna] = np.nan
    return result


def add_alignment_flags(ann: pd.DataFrame) -> pd.DataFrame:
    """Derives the L/H, L/H/PNG and L/H/B/PNG label-agreement flags."""
    ann = ann.copy()
    ann["align_lh"] = _all_equal([ann["l_pref_label"], ann["h_pref_label"]]).to_numpy()
    ann["align_lh_png"] = _all_equal(
        [ann["l_pref_label"], ann["h_pref_label"], ann["png_pref_label"]]
    ).to_numpy()
    ann["align_lhb_png"] = _all_equal(
        [ann["l_pref_label"], ann["h_pref_label"], ann["b_pref_label"],
         ann["png_pref_label"]]
    ).to_numpy()
    return ann
