import logging
import os
from typing import Any, Iterable

import xarray as xr

from .. import io
from .trial import TrialView
from .artifacts import get_results_path

__all__ = ['load_results', "load_detections"]

logger = logging.getLogger(__name__)


def load_results(trial: TrialView, state: str | Iterable[str] = ('pre', 'post'),
                 **kwargs) -> dict[str, xr.DataArray]:
    keys = (state,) if isinstance(state, str) else state
    key_chkpt_mapping = {
        'pre': None,
        'post': -1,
    }
    if not set(keys).issubset(key_chkpt_mapping):
        raise ValueError(keys)
    results = {}
    for key in keys:
        chkpt = key_chkpt_mapping[key]
        results_path = get_results_path(trial, chkpt, **kwargs)
        logger.debug(f"{key}: {results_path}")
        if results_path.exists():
            results[key] = io.load_pickle(results_path)
        else:
            raise FileNotFoundError(f"'{key}': {results_path}")
    return results


def load_detections(trial: TrialView, state: str | Iterable[str] = ('pre', 'post'),
                    sgnf=True, **kwargs) -> dict[str, Any]:
    from hsnn.analysis.png.db import PNGDatabase

    keys = (state,) if isinstance(state, str) else state
    key_chkpt_mapping = {
        'pre': None,
        'post': -1,
    }
    if not set(keys).issubset(key_chkpt_mapping):
        raise ValueError(keys)
    results = {}
    for key in keys:
        chkpt = key_chkpt_mapping[key]
        db = PNGDatabase.from_trial(trial, chkpt, sgnf=sgnf, **kwargs)
        logger.debug(f"{key}: {db.path}")
        if os.path.exists(db.path):
            results[key] = db
        else:
            db.close()
            raise FileNotFoundError(db.path)
    return results
