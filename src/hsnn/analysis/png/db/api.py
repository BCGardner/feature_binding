from typing import Iterable, Optional, Sequence

import pandas as pd
from sqlalchemy import select, and_, or_

from hsnn.core.logger import get_logger
from hsnn.utils.handler import TrialView
from ..base import PNG
from ..refinery import Constrained
from .database import PNGDatabase
from .models import PNGModel, LagModel, OnsetModel
from . import _utils

__all__ = ["get_or_create_db", "get_polygrps", "query_aligned_pngs"]

logger = get_logger()


def get_or_create_db(trial: TrialView, chkpt_idx: Optional[int] = None, *,
                     subdir: Optional[str] = None, sgnf: bool = False,
                     engine_kwargs: Optional[dict] = None, **kwargs) -> PNGDatabase:
    if 'amplitude' in kwargs:
        raise KeyError("Invalid key: 'amplitude'")
    db = PNGDatabase.from_trial(trial, chkpt_idx, subdir=subdir, sgnf=sgnf,
                                engine_kwargs=engine_kwargs, **kwargs)
    if not db.exists:
        db.create()
        logger.info(f"Created PNG database '{db.path}'")
    return db


def get_polygrps(database: PNGDatabase, syn_params: pd.DataFrame, w_min: float = 0.5,
                 tol: float = 3.0, layer: int = 4, nrn_ids: Iterable = range(4096),
                 index: int = 1) -> Sequence[PNG]:
    """Gets detected PNGs from database, for a given `layer`, iterated `nrn_ids` w.r.t. `index`.
    These PNGs are structurally constrained according to `w_min`, `tol`.
    """
    refine = Constrained(syn_params, w_min, tol)
    polygrps = database.get_pngs(layer, nrn_ids, index)
    polygrps_tp = refine(polygrps)
    polygrps_fp = set(polygrps) - set(polygrps_tp)
    print(f"Dropped {len(polygrps_fp) / len(polygrps) * 100:.3f} % PNGs (false positives)")
    return polygrps_tp


def query_aligned_pngs(nrn_id: int, layer: int, index: int, target_times: Sequence[float],
                       png_db: PNGDatabase, tol: float = 3.0) -> Sequence[PNG]:
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
        logger.warning(f"L{layer} neuron {nrn_id} (position {index}) "
                       "not recorded as a run in the database")
        return []

    tolerance_conditions = [
        and_(
            (OnsetModel.time + LagModel.time) >= (target_time - tol),
            (OnsetModel.time + LagModel.time) <= (target_time + tol)
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
            or_(*tolerance_conditions)
        )
    )

    with png_db.get_session() as session:
        png_entries = session.scalars(stmt).all()
        polygrps = _utils.recreate_all(png_entries)
    return polygrps
