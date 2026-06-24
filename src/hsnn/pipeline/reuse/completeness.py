"""Completeness / coverage of the informative low-level feature representation.

Quantifies how completely the network's *informative* low-level (**L**) neurons are recruited
into labelled binding circuits, and how that recruitment thins as the binding circuit's feature
selectivity (PNG F1) is tightened. All functions operate on the neutral per-trial reuse tables
(``hfb_annotations`` and ``neuron_information``) and return tidy :class:`pandas.DataFrame`\\ s.

The quantities, for an anchor layer *l* (the high-level neuron's layer, ``hfb_annotations.layer``;
the L neuron lies in layer *l*-1), an informativeness threshold ``tau`` on ``info_bits``, and a
recruitment tier:

* **Informative pool** (the denominator) -- the distinct layer *l*-1 neurons whose
  stimulus-specific information reaches ``tau``, partitioned by each neuron's own preferred
  label. Reuse is collapsed: a neuron is counted once however many PNGs it joins.
* **Recruitment at a tier** -- the informative pool neurons that are the L neuron of at least one
  *labelled* significant PNG anchored at *l* satisfying the tier. The **structural** tier is any
  labelled PNG (no F1 cut); an **F1 > c** tier additionally requires ``png_pref_f1 > c``. Each
  stricter tier is a subset of the looser one, so per-label recruitment is monotone
  non-increasing across the tier order. Two definitions are recorded: **any-label** (the PNG may
  carry any preferred label) and **label-matched** (the PNG's preferred label equals the
  neuron's own).

A complementary per-circuit view is :func:`circuit_label_alignment`: rather than collapsing reuse
to count distinct neurons, it asks, over the labelled PNGs themselves, how often a constituent
neuron's own preferred label equals the circuit's -- an occurrence-weighted measure of how
reliably a role sets the bound feature.

Anchor layer 1 is normally excluded by the caller: its L neuron sits in the fixed Poisson input
layer, whose informativeness is not assessed.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "Tier",
    "DEFAULT_TIERS",
    "RecruitmentTables",
    "feature_label",
    "conformation_of",
    "informative_pool",
    "recruitment_tables",
    "circuit_label_alignment",
]

Tier = tuple[str, "float | None"]
"""A recruitment tier: (name, f1_cut) where f1_cut is None for structural tier."""


DEFAULT_TIERS: tuple[Tier, ...] = (
    ("structural", None),
    ("f1>0.5", 0.5),
    ("f1>0.7", 0.7),
    ("f1>0.9", 0.9),
)


@dataclass(frozen=True)
class RecruitmentTables:
    """The three tidy completeness frames for one detection trial.

    Attributes:
        counts: One row per ``(anchor_layer, threshold, tier, label)``, with columns ``pool``,
            ``recruited_any`` and ``recruited_matched`` (counts of distinct neurons), plus
            ``conformation``.
        coverage: One row per ``(anchor_layer, threshold, tier)``, with ``cov_informative_neuron``
            (distinct labels in the pool), ``cov_recruited_label`` (distinct labels carried by a
            tier PNG) and ``missing_label`` (a ``";"``-joined list of uncovered labels).
        agreement: One row per ``(anchor_layer, threshold, tier)``, with ``n_recruited_any``,
            ``n_recruited_matched``, ``n_both`` and ``agreement`` (matched-among-any fraction).
    """

    counts: pd.DataFrame
    coverage: pd.DataFrame
    agreement: pd.DataFrame


def feature_label(neuron_information: pd.DataFrame) -> pd.Series:
    """Combines each neuron's preferred side and conformation into one feature label.

    Args:
        neuron_information: The ``neuron_information`` table, with ``pref_side`` and
            ``pref_conformation`` columns.

    Returns:
        A string :class:`pandas.Series` ``"{pref_side}-{pref_conformation}"`` aligned to the
        input index (e.g. ``"left-convex"``).
    """
    return (
        neuron_information["pref_side"].astype(str)
        + "-"
        + neuron_information["pref_conformation"].astype(str)
    )


def conformation_of(label: "pd.Series | str") -> "pd.Series | str":
    """Extracts the conformation token (the part after the final hyphen) from a feature label.

    Args:
        label: A single ``"{side}-{conformation}"`` string, or a Series of them.

    Returns:
        The conformation token (e.g. ``"convex"``); a string for a string input, else a Series.
    """
    if isinstance(label, str):
        return label.rsplit("-", 1)[-1]
    return label.str.rsplit("-", n=1).str[-1]


def _with_label(neuron_information: pd.DataFrame) -> pd.DataFrame:
    """Returns ``neuron_information`` with a ``pref_label`` column, deriving it if absent."""
    if "pref_label" in neuron_information.columns:
        return neuron_information
    return neuron_information.assign(pref_label=feature_label(neuron_information))


def informative_pool(
    neuron_information: pd.DataFrame, anchor_layer: int, info_threshold: float
) -> pd.DataFrame:
    """The informative low-level pool for an anchor layer: the recruitment denominator.

    Args:
        neuron_information: The ``neuron_information`` table (columns ``neuron``, ``layer``,
            ``info_bits`` and either ``pref_label`` or ``pref_side``/``pref_conformation``).
        anchor_layer: The anchor layer *l*; the pool lives in layer *l*-1.
        info_threshold: The informativeness threshold ``tau`` (bits) on ``info_bits``.

    Returns:
        A frame of the distinct informative layer *l*-1 neurons, with columns ``neuron`` and
        ``pref_label``.
    """
    info = _with_label(neuron_information)
    pool = info[(info["layer"] == anchor_layer - 1) & (info["info_bits"] >= info_threshold)]
    return pool[["neuron", "pref_label"]].reset_index(drop=True)


def _tier_selection(anchored: pd.DataFrame, cut: "float | None") -> pd.DataFrame:
    """The labelled anchored PNGs satisfying a tier (all of them for the structural tier)."""
    return anchored if cut is None else anchored[anchored["png_pref_f1"] > cut]


def recruitment_tables(
    annotations: pd.DataFrame,
    neuron_information: pd.DataFrame,
    labels: Sequence[str],
    *,
    anchor_layers: Iterable[int],
    info_thresholds: Iterable[float] = (2 / 3,),
    tiers: Iterable[Tier] = DEFAULT_TIERS,
) -> RecruitmentTables:
    """Per-label recruitment counts, coverage and any-vs-matched agreement for one trial.

    Sweeps the anchor layers, informativeness thresholds and recruitment tiers, counting the
    distinct informative low-level neurons (reuse collapsed) recruited as the L neuron of a
    labelled PNG at each tier. Recruitment numerators intersect the pool's neuron-id set with the
    PNGs' ``l_id`` set, so a reused neuron is counted once.

    Args:
        annotations: The ``hfb_annotations`` table (one row per significant PNG), with columns
            ``layer``, ``l_id``, ``png_pref_label`` and ``png_pref_f1``.
        neuron_information: The ``neuron_information`` table (see :func:`informative_pool`).
        labels: The feature labels partitioning the pool denominator (the per-label rows).
        anchor_layers: The anchor layers to evaluate.
        info_thresholds: Informativeness thresholds (bits). Defaults to ``(2/3,)``.
        tiers: The recruitment tiers as ``(name, f1_cut)`` pairs, ordered loosest-first.
            Defaults to :data:`DEFAULT_TIERS`.

    Returns:
        A :class:`RecruitmentTables` bundling the per-label ``counts`` frame and the per-tier
        ``coverage`` and ``agreement`` frames.
    """
    info = _with_label(neuron_information)
    labels = list(labels)
    label_set = set(labels)
    tiers = list(tiers)

    count_rows, cov_rows, agree_rows = [], [], []
    for layer in anchor_layers:
        anchored = annotations[
            (annotations["layer"] == layer) & (annotations["png_pref_label"].notna())
        ]
        for tau in info_thresholds:
            pool = informative_pool(info, layer, tau)
            own = dict(zip(pool["neuron"], pool["pref_label"]))
            pool_ids = set(pool["neuron"])
            den_by_label = {
                lab: set(pool.loc[pool["pref_label"] == lab, "neuron"]) for lab in labels
            }
            cov_informative = int(pool["pref_label"].nunique())
            for tier_name, cut in tiers:
                sel = _tier_selection(anchored, cut)
                recruited_ids = set(sel["l_id"])
                # Label-matched: the L neuron's own preferred label equals the PNG's.
                pairs = sel[["l_id", "png_pref_label"]].drop_duplicates()
                matched_ids = {
                    row.l_id for row in pairs.itertuples()
                    if own.get(row.l_id) == row.png_pref_label
                }
                for label in labels:
                    den = den_by_label[label]
                    count_rows.append({
                        "anchor_layer": layer, "threshold": tau, "tier": tier_name,
                        "label": label, "conformation": conformation_of(label),
                        "pool": len(den),
                        "recruited_any": len(den & recruited_ids),
                        "recruited_matched": len(den & matched_ids),
                    })
                any_set, matched_set = pool_ids & recruited_ids, pool_ids & matched_ids
                agree_rows.append({
                    "anchor_layer": layer, "threshold": tau, "tier": tier_name,
                    "n_recruited_any": len(any_set), "n_recruited_matched": len(matched_set),
                    "n_both": len(any_set & matched_set),
                    "agreement": (len(any_set & matched_set) / len(any_set)) if any_set else np.nan,
                })
                present = set(sel["png_pref_label"].dropna().unique())
                cov_rows.append({
                    "anchor_layer": layer, "threshold": tau, "tier": tier_name,
                    "cov_informative_neuron": cov_informative,
                    "cov_recruited_label": len(present & label_set),
                    "missing_label": ";".join(sorted(label_set - present)),
                })
    return RecruitmentTables(
        counts=pd.DataFrame(count_rows),
        coverage=pd.DataFrame(cov_rows),
        agreement=pd.DataFrame(agree_rows),
    )


def _role_prefix(role: str) -> str:
    """The lower-case column prefix (``l``/``h``/``b``) for a constituent role label."""
    prefix = role.lower()
    if prefix not in {"l", "h", "b"}:
        raise ValueError(f"unknown role {role!r}; expected one of 'L', 'H', 'B'")
    return prefix


def circuit_label_alignment(
    annotations: pd.DataFrame,
    role: str,
    *,
    f1_threshold: float,
    anchor_layer: "int | None" = None,
    restrict_informative: "str | None" = "L",
) -> float:
    """Per-circuit fraction whose preferred label matches a constituent neuron's own.

    Over the labelled PNGs exceeding ``f1_threshold`` in ``png_pref_f1`` (optionally restricted to
    ``anchor_layer`` and to circuits whose ``restrict_informative`` neuron is informative), returns
    the fraction whose ``png_pref_label`` equals the ``role`` neuron's own preferred label.

    This is a per-circuit (occurrence-weighted) quantity, complementary to the reuse-collapsed,
    per-neuron recruitment counts of :func:`recruitment_tables`: a neuron appearing in several
    circuits contributes once per circuit. Fixing the population by a single role's informativeness
    (by default the low-level neuron) lets the three roles be compared over the same circuits, with
    ``role`` selecting only which neuron's label is tested.

    Args:
        annotations: The ``hfb_annotations`` table for one trial, with columns ``layer``,
            ``png_pref_label``, ``png_pref_f1`` and, for each role *x* referenced,
            ``{x}_pref_label`` and ``{x}_informative``.
        role: The constituent whose label is tested: ``"L"``, ``"H"`` or ``"B"``
            (case-insensitive).
        f1_threshold: Lower bound (exclusive) on ``png_pref_f1``.
        anchor_layer: If given, restrict to PNGs anchored at this layer; otherwise all layers.
        restrict_informative: If given (a role), restrict to circuits whose that-role neuron is
            informative; ``None`` applies no informativeness restriction.

    Returns:
        The per-circuit alignment fraction, or ``nan`` if no circuit satisfies the selection.
    """
    tested = _role_prefix(role)
    sub = annotations[
        annotations["png_pref_label"].notna() & (annotations["png_pref_f1"] > f1_threshold)
    ]
    if anchor_layer is not None:
        sub = sub[sub["layer"] == anchor_layer]
    if restrict_informative is not None:
        gate = _role_prefix(restrict_informative)
        sub = sub[sub[f"{gate}_informative"].astype(bool)]
    if len(sub) == 0:
        return float("nan")
    return float((sub["png_pref_label"] == sub[f"{tested}_pref_label"]).mean())
