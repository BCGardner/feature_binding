from typing import Iterable, Optional, Sequence, cast

import pandas as pd
from sqlalchemy import and_, or_, select

from hsnn.core.logger import get_logger
from hsnn.utils.handler import TrialView

from ..base import PNG
from ..refinery import Constrained, Match
from . import _utils
from .database import PNGDatabase
from .models import LagModel, OnsetModel, PNGModel

__all__ = [
    "get_or_create_db",
    "get_polygrps",
    "find_idx",
    "find_matching_index",
    "query_aligned_pngs",
]

logger = get_logger()


def get_or_create_db(
    trial: TrialView,
    chkpt_idx: Optional[int] = None,
    *,
    subdir: Optional[str] = None,
    sgnf: bool = False,
    engine_kwargs: Optional[dict] = None,
    **kwargs,
) -> PNGDatabase:
    if "amplitude" in kwargs:
        raise KeyError("Invalid key: 'amplitude'")
    db = PNGDatabase.from_trial(
        trial,
        chkpt_idx,
        subdir=subdir,
        sgnf=sgnf,
        engine_kwargs=engine_kwargs,
        **kwargs,
    )
    if not db.exists:
        db.create()
        logger.info(f"Created PNG database '{db.path}'")
    return db


def find_idx(query: PNG, polygrps: Sequence[PNG]) -> int:
    """Find the index of a PNG in a sequence by equality."""
    for idx, polygrp in enumerate(polygrps):
        if polygrp == query:
            return idx
    raise IndexError("No matching index found for the given query.")


def find_matching_index(
    indices: Sequence[tuple[int, int]], polygrps: Sequence[PNG]
) -> int:
    """
    Return the first index of a PNG in a sequence identified by layer-nrn pairs:
    `[(layer_1, nrn_1), ...]`.
    """
    queries = Match(indices)(polygrps)
    if len(queries):
        return find_idx(queries[0], polygrps)
    raise IndexError("No matching index found for the given query.")


def get_polygrps(
    database: PNGDatabase,
    syn_params: pd.DataFrame | None = None,
    *,
    w_min: float = 0.5,
    tol: float = 3.0,
    layer: int = 4,
    nrn_ids: Iterable = range(4096),
    index: int = 1,
) -> list[PNG]:
    """Gets detected PNGs from database, for a given `layer`, iterated `nrn_ids` w.r.t. `index`.

    If `syn_params` is provided, these PNGs are structurally constrained according to `w_min`, `tol`.
    """
    polygrps = database.get_pngs(layer, nrn_ids, index)
    if syn_params is not None:
        refine = Constrained(syn_params, w_min, tol)
        polygrps_tp = refine(polygrps)
        polygrps_fp = set(polygrps) - set(polygrps_tp)
        print(
            f"Dropped {len(polygrps_fp) / len(polygrps) * 100:.3f} % PNGs (false positives)"
        )
        return cast(list[PNG], polygrps_tp)

    return cast(list[PNG], polygrps)


def query_aligned_pngs(
    nrn_id: int,
    layer: int,
    index: int,
    target_times: Sequence[float],
    png_db: PNGDatabase,
    tol: float = 3.0,
) -> list[PNG]:
    """Query for PNGs containing a neuron which fires a spike aligned with a target
    timing.

    Args:
        nrn_id (int): PNG firing neuron.
        layer (int): Layer of firing neuron.
        index (int): The index of the PNG neuron.
        target_times (Sequence[float]): Target times to query against.
        png_db (PNGDatabase): The database of previously detected PNGs.
        tol (float, optional): Tolerance window to consider aligned timings. Defaults to 3.0.

    Returns:
        Sequence[PNG]: Recreated PNGs from selected records.
    """
    target_times = [float(t) for t in target_times]
    if nrn_id not in png_db.get_run_nrns(layer, index):
        logger.warning(
            f"L{layer} neuron {nrn_id} (position {index}) "
            "not recorded as a run in the database"
        )
        return []

    tolerance_conditions = [
        and_(
            (OnsetModel.time + LagModel.time) >= (target_time - tol),
            (OnsetModel.time + LagModel.time) <= (target_time + tol),
        )
        for target_time in target_times
    ]
    stmt = (
        select(PNGModel)
        .distinct()
        .join(LagModel)
        .join(OnsetModel)
        .where(
            LagModel.index == index,
            LagModel.layer == layer,
            LagModel.neuron == nrn_id,
            or_(*tolerance_conditions),
        )
    )

    with png_db.get_session() as session:
        png_entries = session.scalars(stmt).all()
        polygrps = _utils.recreate_all(png_entries)
    return cast(list[PNG], polygrps)
