"""Role ambiguity and resolution over the HFB annotation table.

This module quantifies the **ambiguity** (a circuit neuron's own identity becomes
increasingly insufficient to specify the bound feature with depth) and supports the
**resolution** (the bound feature is nonetheless carried by the time-locked polychronous
pattern, i.e. high PNG F1, even where the neuron's own preferred feature differs).

The quantities are defined identically for each of the three circuit roles, selected by a
``role`` argument (defaulting to the binding neuron ``"B"``):

* ``"L"`` -- the low-level neuron (first-firing, lag index 0), keyed by ``(l_layer, l_id)``;
* ``"H"`` -- the high-level neuron (second-firing, lag index 1), keyed by ``(h_layer, h_id)``;
* ``"B"`` -- the binding neuron (third-firing, lag index 2), keyed by ``(b_layer, b_id)``.

All functions operate on the ``hfb_annotations`` table (one row per significant PNG; see
:mod:`hsnn.pipeline.reuse.api`), whose ``l_*`` / ``h_*`` / ``b_*`` columns are symmetric, and
return neutral pandas/numpy data. ``layer`` is the anchor layer (the high-level neuron's
layer); analysis is over **labelled** PNGs -- those carrying a non-null ``png_pref_label``.

The three match categories are defined on the full ``side-conformation`` label and named for
the role (``binder_matched`` / ``low_matched`` / ``high_matched`` and so on; see
:func:`match_categories`):

* ``*_matched`` -- the role neuron is informative and its own preferred label equals the PNG's;
* ``*_mismatched`` -- the role neuron is informative and the two differ;
* ``*_uninformative`` -- the role neuron is not informative (or has no preferred label).

A neuron's preferred label is an argmax over sides and is therefore present even when the
neuron is uninformative; the ``*_uninformative`` category is gated on the role's
``*_informative`` flag, not on a missing label.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

__all__ = [
    "Role",
    "match_categories",
    "distinct_label_count",
    "match_category",
    "label_divergence_fraction",
    "high_f1_mismatch_fractions",
    "pool_trials",
    "trial_comparability",
]

Role = Literal["L", "H", "B"]

_LABEL_COL = "png_pref_label"
_F1_COL = "png_pref_f1"
_LAYER_COL = "layer"


@dataclass(frozen=True)
class _RoleSpec:
    """Maps a circuit role to its annotation columns and category labels.

    ``prefix`` is the column prefix (``l`` / ``h`` / ``b``); ``name`` is the human role name
    used to qualify the match-category labels (``low`` / ``high`` / ``binder``).
    """

    prefix: str
    name: str

    @property
    def layer_col(self) -> str:
        return f"{self.prefix}_layer"

    @property
    def id_col(self) -> str:
        return f"{self.prefix}_id"

    @property
    def pref_label_col(self) -> str:
        return f"{self.prefix}_pref_label"

    @property
    def informative_col(self) -> str:
        return f"{self.prefix}_informative"

    @property
    def categories(self) -> tuple[str, str, str]:
        return (
            f"{self.name}_matched",
            f"{self.name}_mismatched",
            f"{self.name}_uninformative",
        )


_ROLE_SPECS: dict[str, _RoleSpec] = {
    "L": _RoleSpec("l", "low"),
    "H": _RoleSpec("h", "high"),
    "B": _RoleSpec("b", "binder"),
}


def _spec(role: Role) -> _RoleSpec:
    """Resolves the role specification, accepting case-insensitive role keys."""
    try:
        return _ROLE_SPECS[role.upper()]
    except (AttributeError, KeyError):
        raise ValueError(
            f"Unknown role {role!r}; expected one of {list(_ROLE_SPECS)}"
        ) from None


def match_categories(role: Role = "B") -> tuple[str, str, str]:
    """The match-category labels for a role, in (matched, mismatched, uninformative) order.

    Args:
        role: Circuit role (``"L"``, ``"H"`` or ``"B"``). Defaults to ``"B"`` (binder).

    Returns:
        The role's three category labels, e.g. ``("binder_matched", "binder_mismatched",
        "binder_uninformative")`` for the binder.
    """
    return _spec(role).categories


def distinct_label_count(ann: pd.DataFrame, role: Role = "B") -> pd.Series:
    """Counts the distinct PNG preferred labels among the PNGs each role neuron is part of.

    For each role neuron (keyed by the role's ``(layer, id)`` within a trial), counts the
    number of distinct non-null ``png_pref_label`` values across the PNGs in which it plays
    that role. Every PNG inherits its role neuron's count, so a higher count marks a more
    ambiguous neuron.

    Args:
        ann: The ``hfb_annotations`` table (one row per significant PNG).
        role: Circuit role (``"L"``, ``"H"`` or ``"B"``). Defaults to ``"B"`` (binder).

    Returns:
        An integer :class:`pandas.Series` aligned to ``ann.index`` giving each PNG's role
        neuron's distinct-label count.
    """
    spec = _spec(role)
    return ann.groupby(_role_keys(ann, spec), dropna=False)[_LABEL_COL].transform("nunique")


def match_category(ann: pd.DataFrame, role: Role = "B") -> pd.Series:
    """Assigns each labelled PNG to a role match category.

    Categories (see module docstring): ``*_matched``, ``*_mismatched`` or ``*_uninformative``,
    qualified by the role (see :func:`match_categories`). PNGs without a preferred label are
    left unassigned (null), so the categories partition the labelled PNGs without overlap.

    Args:
        ann: The ``hfb_annotations`` table.
        role: Circuit role (``"L"``, ``"H"`` or ``"B"``). Defaults to ``"B"`` (binder).

    Returns:
        A categorical :class:`pandas.Series` aligned to ``ann.index`` with the role's
        categories (:func:`match_categories`); null for unlabelled PNGs.
    """
    spec = _spec(role)
    matched, mismatched, uninformative = spec.categories
    labelled = ann[_LABEL_COL].notna()
    informative = _informative(ann, spec) & ann[spec.pref_label_col].notna()
    same = ann[spec.pref_label_col] == ann[_LABEL_COL]
    category = np.where(
        ~informative,
        uninformative,
        np.where(same, matched, mismatched),
    )
    category = np.where(labelled.to_numpy(), category, None)  # pyright: ignore[reportArgumentType]
    return pd.Series(
        pd.Categorical(category, categories=spec.categories), index=ann.index
    )


def label_divergence_fraction(
    ann: pd.DataFrame, role: Role = "B", *, require_informative: bool = False
) -> pd.DataFrame:
    """Per anchor layer, the fraction of labelled PNGs whose role neuron agrees with the PNG.

    For each anchor ``layer``, the fraction of labelled PNGs whose role neuron's own preferred
    label equals the PNG's preferred label. This is the complement of label *divergence*: it
    falls with depth as neurons grow ambiguous. Pass ``require_informative=True`` to restrict
    to PNGs whose role neuron is informative.

    Args:
        ann: The ``hfb_annotations`` table.
        role: Circuit role (``"L"``, ``"H"`` or ``"B"``). Defaults to ``"B"`` (binder).
        require_informative: If ``True``, restrict to PNGs with an informative role neuron.
            Defaults to ``False``.

    Returns:
        A frame with columns ``layer``, ``match_fraction`` and ``n_pngs`` (the denominator).
    """
    spec = _spec(role)
    sub = ann[ann[_LABEL_COL].notna()].dropna(subset=[spec.pref_label_col])
    if require_informative:
        sub = sub[_informative(sub, spec)]
    same = (sub[spec.pref_label_col] == sub[_LABEL_COL]).astype(float)
    grouped = same.groupby(sub[_LAYER_COL])
    return pd.DataFrame(
        {"match_fraction": grouped.mean(), "n_pngs": grouped.size()}
    ).reset_index()


def high_f1_mismatch_fractions(
    ann: pd.DataFrame,
    thresholds: Sequence[float] = (0.9, 0.8, 0.7),
    role: Role = "B",
) -> pd.DataFrame:
    """Among high-F1 labelled PNGs, the fraction whose role neuron mismatches the PNG feature.

    For each F1 threshold, restricts to labelled PNGs with ``png_pref_f1`` at or above the
    threshold and reports, per anchor layer and overall, the fraction that are mismatched and
    the fraction that are uninformative (with counts). These are the circuits that are
    strongly feature-selective yet whose role neuron does not itself prefer that feature -- the
    direct evidence that selectivity resides in the pattern.

    The fractions are computed on a single trial's table; report mean and spread across trials
    by calling per trial and combining (see :func:`pool_trials`).

    Args:
        ann: The ``hfb_annotations`` table.
        thresholds: F1 thresholds to evaluate. Defaults to ``(0.9, 0.8, 0.7)``.
        role: Circuit role (``"L"``, ``"H"`` or ``"B"``). Defaults to ``"B"`` (binder).

    Returns:
        A long-form frame with columns ``threshold``, ``layer`` (each anchor layer plus
        ``"overall"``), ``n_high_f1`` and the role-qualified ``n_{name}_mismatched``,
        ``frac_{name}_mismatched``, ``n_{name}_uninformative`` and ``frac_{name}_uninformative``
        (e.g. ``n_binder_mismatched`` for the binder).
    """
    spec = _spec(role)
    _, mismatched, uninformative = spec.categories
    sub = ann[ann[_LABEL_COL].notna()].copy()
    sub["match_category"] = match_category(sub, role)
    layers = sorted(int(layer) for layer in sub[_LAYER_COL].unique())
    rows = []
    for threshold in thresholds:
        high = sub[sub[_F1_COL] >= threshold]
        for layer in [*layers, "overall"]:
            group = high if layer == "overall" else high[high[_LAYER_COL] == layer]
            n = len(group)
            n_mismatch = int((group["match_category"] == mismatched).sum())
            n_uninf = int((group["match_category"] == uninformative).sum())
            rows.append(
                {
                    "threshold": threshold,
                    "layer": layer,
                    "n_high_f1": n,
                    f"n_{spec.name}_mismatched": n_mismatch,
                    f"frac_{spec.name}_mismatched": n_mismatch / n if n else np.nan,
                    f"n_{spec.name}_uninformative": n_uninf,
                    f"frac_{spec.name}_uninformative": n_uninf / n if n else np.nan,
                }
            )
    return pd.DataFrame(rows)


def pool_trials(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Concatenates per-trial frames into one pooled frame.

    Pooling adds samples rather than summing counts, so it is appropriate for value
    distributions (F1, information) but not for per-trial scalars or counts.

    Args:
        frames: Per-trial frames sharing a schema.

    Returns:
        The row-wise concatenation with a fresh integer index.
    """
    return pd.concat(list(frames), ignore_index=True)


def trial_comparability(
    annotations: Mapping[str, pd.DataFrame], role: Role = "B"
) -> pd.DataFrame:
    """Summarises the per-trial divergence trend and F1 medians used to justify pooling.

    Pooling value distributions across trials is only sound when the trials are comparable.
    This reports, per trial, the labelled-PNG count, the median PNG F1, and the per-layer
    match fraction (the divergence trend), so an outlier trial can be surfaced rather than
    pooled silently.

    Args:
        annotations: Mapping of ``trial_id -> hfb_annotations`` frame.
        role: Circuit role (``"L"``, ``"H"`` or ``"B"``) whose divergence trend is reported.
            Defaults to ``"B"`` (binder).

    Returns:
        One row per trial with columns ``trial_id``, ``n_labelled``, ``f1_median`` and
        ``match_L{layer}`` for each anchor layer.
    """
    rows = []
    for trial_id, ann in annotations.items():
        labelled = ann[ann[_LABEL_COL].notna()]
        divergence = label_divergence_fraction(ann, role).set_index(_LAYER_COL)[
            "match_fraction"
        ]
        row: dict[str, object] = {
            "trial_id": trial_id,
            "n_labelled": int(len(labelled)),
            "f1_median": float(labelled[_F1_COL].median()),
        }
        for layer, value in divergence.items():
            row[f"match_L{int(layer)}"] = round(float(value), 3)  # pyright: ignore[reportArgumentType]
        rows.append(row)
    return pd.DataFrame(rows)


def _role_keys(ann: pd.DataFrame, spec: _RoleSpec) -> list[str]:
    """The grouping columns identifying one role neuron within a trial.

    ``trial_id`` is included when present so that the same ``(layer, id)`` in different trials
    is never conflated; on a single-trial frame it may be absent.
    """
    return [c for c in ("trial_id", spec.layer_col, spec.id_col) if c in ann.columns]


def _informative(ann: pd.DataFrame, spec: _RoleSpec) -> pd.Series:
    """Boolean role-informative flag, treating null as not informative."""
    return ann[spec.informative_col].fillna(False).astype(bool)
