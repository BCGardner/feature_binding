"""Role co-membership / role-switching matrices over the HFB annotation table.

Quantifies how often a neuron that fills one circuit role (low-level **L**, high-level
**H**, binding **B**) *also* fills another role somewhere in its repertoire. All functions
operate on the ``hfb_annotations`` table (one row per significant PNG,
with L/H/B ids and layers) and return neutral data (Python sets, pandas frames).

Membership is **structural**: a neuron fills a role if it occupies that role in at least
one significant PNG, regardless of F1 or label. A neuron is identified by its composite
``(layer, neuron)`` key -- the same physical neuron regardless of which role column it
appears in.

Roles are lag-ordered (PNGs lag-ordered): index 0 = L (layer ``l-1``), index 1 = H
(layer ``l``), index 2 = B (layer ``l``). A PNG's **anchor layer** ``l`` is the layer of
its index-1 (H) neuron (``hfb_annotations.layer``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "ROLES",
    "RoleComembershipResult",
    "role_comembership",
    "role_reuse_counts",
]

ROLES = ("L", "H", "B")
_ROLE_LAYER_COL = {"L": "l_layer", "H": "h_layer", "B": "b_layer"}
_ROLE_ID_COL = {"L": "l_id", "H": "h_id", "B": "b_id"}
_INPUT_LAYER = 0  # Poisson stimulus layer (never a role).

Key = tuple[int, int]  # (layer, neuron) composite key = one physical neuron


@dataclass(frozen=True)
class RoleComembershipResult:
    """The role co-membership outputs for a single anchor layer.

    Attributes:
        proportions: 3x3 row-conditional role-switching proportions, indexed and
            columned by ``('L', 'H', 'B')``.
        counts: 3x3 raw co-membership counts (intersection sizes); the within-layer
            off-diagonals are symmetric.
        row_counts: Per-role row-population size (``role -> n``).
        mask: 3x3 boolean frame, ``True`` where a cell is structurally impossible.
    """

    proportions: pd.DataFrame
    counts: pd.DataFrame
    row_counts: dict[str, int]
    mask: pd.DataFrame


def global_role_sets(ann: pd.DataFrame) -> dict[str, set[Key]]:
    """Builds each role's global ``(layer, neuron)`` key set across all PNGs.

    Spans every significant PNG, **including layer-1 anchored circuits**, so a neuron's
    role repertoire is complete. The low-level set excludes the Poisson input layer
    (``l_layer == 0``): those input neurons are never assigned a role.

    Args:
        ann: The ``hfb_annotations`` table (one row per significant PNG).

    Returns:
        Mapping ``role -> {(layer, neuron)}`` for roles ``'L'``, ``'H'``, ``'B'``.
    """
    sets: dict[str, set[Key]] = {}
    for role in ROLES:
        sub = ann[ann["l_layer"] != _INPUT_LAYER] if role == "L" else ann
        sets[role] = set(zip(sub[_ROLE_LAYER_COL[role]].astype(int),
                             sub[_ROLE_ID_COL[role]].astype(int)))
    return sets


def row_population(ann: pd.DataFrame, anchor_layer: int, role: str) -> set[Key]:
    """The ``(layer, neuron)`` keys filling ``role`` among layer-``anchor_layer`` PNGs.

    For role ``'L'`` these neurons lie in layer ``anchor_layer - 1``; for ``'H'`` and
    ``'B'`` they lie in ``anchor_layer``.

    Args:
        ann: The ``hfb_annotations`` table.
        anchor_layer: The anchor layer ``l`` (the high-level neuron's layer).
        role: One of ``'L'``, ``'H'`` or ``'B'``.

    Returns:
        The set of composite keys for that role's row population.
    """
    sub = ann[ann["layer"] == anchor_layer]
    return set(zip(sub[_ROLE_LAYER_COL[role]].astype(int),
                   sub[_ROLE_ID_COL[role]].astype(int)))


def role_reuse_counts(
    annotations: pd.DataFrame,
    layers: Sequence[int] = (2, 3, 4),
) -> pd.DataFrame:
    """Per-(anchor layer, role) structural-reuse counts for one trial.

    For each anchor layer ``l`` and role, counts the PNGs anchored at ``l`` (``N_l``) and
    the distinct neurons filling that role among them, and reports their ratio (the reuse
    factor, equal to the mean of the role's participation distribution). Reuse is purely
    structural: every significant PNG contributes exactly one neuron per role.

    Args:
        annotations: The ``hfb_annotations`` table for a single trial (one row per
            significant PNG).
        layers: Anchor layers to summarise (layer 1 excluded by default: its L neuron sits
            in the Poisson input layer).

    Returns:
        A tidy frame with columns ``layer``, ``role``, ``n_pngs`` (the PNG count ``N_l``),
        ``distinct_count`` (distinct neurons in that role) and ``reuse_factor``
        (``n_pngs / distinct_count``).
    """
    rows = []
    for layer in layers:
        anchored = annotations[annotations["layer"] == layer]
        n_pngs = len(anchored)
        for role in ROLES:
            distinct = int(anchored[_ROLE_ID_COL[role]].nunique())
            rows.append({
                "layer": int(layer),
                "role": role,
                "n_pngs": int(n_pngs),
                "distinct_count": distinct,
                "reuse_factor": n_pngs / distinct if distinct else np.nan,
            })
    return pd.DataFrame(rows)


def _switching_matrix(ann: pd.DataFrame, anchor_layer: int,
                      global_sets: Mapping[str, set[Key]]) -> pd.DataFrame:
    """Row-conditional role-switching proportions for one anchor layer.

    Cell ``(r1, r2)`` is ``|row_pop(r1) & GLOBAL(r2)| / |row_pop(r1)|`` -- the fraction of
    the layer's role-``r1`` neurons that also occupy role ``r2`` anywhere in their
    repertoire. Asymmetric off-diagonals; the diagonal is 1.0 by construction.

    Args:
        ann: The ``hfb_annotations`` table.
        anchor_layer: The anchor layer ``l``.
        global_sets: Output of :func:`global_role_sets`.

    Returns:
        A 3x3 frame indexed/columned by ``('L', 'H', 'B')``.
    """
    mat = pd.DataFrame(index=list(ROLES), columns=list(ROLES), dtype=float)
    for r1 in ROLES:
        pop = row_population(ann, anchor_layer, r1)
        n = len(pop)
        for r2 in ROLES:
            mat.loc[r1, r2] = len(pop & global_sets[r2]) / n if n else np.nan
    return mat


def _comembership_counts(ann: pd.DataFrame, anchor_layer: int,
                         global_sets: Mapping[str, set[Key]]) -> pd.DataFrame:
    """Raw co-membership counts (intersection sizes) for one anchor layer.

    Cell ``(r1, r2) = |row_pop(r1) & GLOBAL(r2)|``; the diagonal equals the row-population
    size. The within-layer off-diagonals (e.g. ``(H, B)`` and ``(B, H)``) are symmetric.

    Args:
        ann: The ``hfb_annotations`` table.
        anchor_layer: The anchor layer ``l``.
        global_sets: Output of :func:`global_role_sets`.

    Returns:
        A 3x3 integer-valued frame indexed/columned by ``('L', 'H', 'B')``.
    """
    mat = pd.DataFrame(index=list(ROLES), columns=list(ROLES), dtype=int)
    for r1 in ROLES:
        pop = row_population(ann, anchor_layer, r1)
        for r2 in ROLES:
            mat.loc[r1, r2] = len(pop & global_sets[r2])
    return mat


def impossible_mask(anchor_layer: int, top_layer: int) -> pd.DataFrame:
    """Structurally-impossible cells: the top-layer ``(H, L)`` and ``(B, L)`` only.

    A top-layer neuron can never be a low-level neuron (no higher layer to anchor), so
    those off-diagonal cells in the low-level *column* are impossible and greyed. The
    ``(L, L)`` diagonal is still 1.0.

    Args:
        anchor_layer: The anchor layer ``l``.
        top_layer: The top (highest) spiking layer; at this anchor layer the low-level
            column off-diagonals are structurally impossible.

    Returns:
        A boolean 3x3 frame, ``True`` where a cell is structurally impossible.
    """
    mask = pd.DataFrame(False, index=list(ROLES), columns=list(ROLES))
    if anchor_layer == top_layer:
        mask.loc[["H", "B"], "L"] = True
    return mask


def role_comembership(
    annotation: pd.DataFrame,
    anchor_layer: int,
    *,
    global_sets: Mapping[str, set[Key]] | None = None,
    top_layer: int | None = None,
) -> RoleComembershipResult:
    """Computes the role co-membership result for a single anchor layer.

    Args:
        annotation: The ``hfb_annotations`` table (one row per significant PNG).
        anchor_layer: The anchor layer ``l`` (the high-level neuron's layer).
        global_sets: Pre-computed output of :func:`global_role_sets`. If ``None``
            (default), it is computed from ``annotation``. Pass it explicitly to share
            one realisation across several anchor layers.
        top_layer: The top (highest) spiking layer, used to grey the structurally
            impossible low-level column. If ``None`` (default), it is taken as
            ``annotation['layer'].max()``.

    Returns:
        The :class:`RoleComembershipResult` bundling the row-conditional proportions, the
        symmetric raw co-membership counts, the per-role row-population sizes, and the
        structurally-impossible mask.
    """
    gs = global_sets if global_sets is not None else global_role_sets(annotation)
    top = top_layer if top_layer is not None else int(annotation["layer"].max())
    return RoleComembershipResult(
        proportions=_switching_matrix(annotation, anchor_layer, gs),
        counts=_comembership_counts(annotation, anchor_layer, gs),
        row_counts={r: len(row_population(annotation, anchor_layer, r)) for r in ROLES},
        mask=impossible_mask(anchor_layer, top),
    )
