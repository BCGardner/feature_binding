from typing import Any, Iterable, Optional, Sequence

import numpy as np

from ..base import PNG
from .models import PNGModel, LagModel, OnsetModel, RunModel

__all__ = ["recreate_all"]


def create_lag_models(polygrp: PNG) -> list[LagModel]:
    lag_models = []
    for i in range(len(polygrp.lags)):
        lag_model = LagModel(
            index=i, layer=int(polygrp.layers[i]), neuron=int(polygrp.nrns[i]),
            time=polygrp.lags[i]
        )
        lag_models.append(lag_model)
    return lag_models


def create_onset_models(polygrp: PNG) -> list[OnsetModel]:
    onset_models = []
    for i in range(len(polygrp.times)):
        time = float(polygrp.times[i])
        img = int(polygrp.imgs[i]) if polygrp.imgs is not None else None
        rep = int(polygrp.reps[i]) if polygrp.reps is not None else None
        time_rel = float(polygrp.times_rel[i]) if polygrp.times_rel is not None else None
        onset_models.append(
            OnsetModel(time=time, img=img, rep=rep, time_rel=time_rel)
        )
    return onset_models


def create_png_model(polygrp: PNG) -> PNGModel:
    span = polygrp.lags[-1] - polygrp.lags[0]
    return PNGModel(
        id=hash(polygrp),
        size=len(polygrp.nrns),
        span=span,
        lags=create_lag_models(polygrp),
        onsets=create_onset_models(polygrp)
    )


def _sorted_onsets_data(onsets: list[OnsetModel]) -> dict:
    def array_or_none(arr: list, dtype: Any,
                      indices: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        if None in arr:
            return None
        ret = np.array(arr, dtype=dtype)
        if indices is not None:
            return ret[indices]

    times = np.array([onset.time for onset in onsets], dtype=float)
    indices = np.argsort(times)
    imgs = array_or_none([onset.img for onset in onsets], dtype=int, indices=indices)
    reps = array_or_none([onset.rep for onset in onsets], dtype=int, indices=indices)
    times_rel = array_or_none([onset.time_rel for onset in onsets], dtype=float, indices=indices)
    return {
        'times': times[indices], 'imgs': imgs, 'reps': reps, 'times_rel': times_rel
    }


def _recreate_png(png_entry: PNGModel) -> PNG:
    layers = np.array([lag.layer for lag in png_entry.lags], dtype=int)
    nrns = np.array([lag.neuron for lag in png_entry.lags], dtype=int)
    lags = np.array([lag.time for lag in png_entry.lags], dtype=float)
    onsets_data = _sorted_onsets_data(png_entry.onsets)
    args = np.lexsort((nrns, layers, lags))
    return PNG(layers=layers[args], nrns=nrns[args], lags=lags[args], **onsets_data)


def recreate_all(png_entries: Iterable[PNGModel]) -> Sequence[PNG]:
    polygrps = [_recreate_png(png_entry) for png_entry in png_entries]
    return sorted(polygrps, key=lambda x: len(x.times), reverse=True)


def create_run_model(nrn_id: int, layer: int, index: int) -> RunModel:
    return RunModel(layer=int(layer), neuron=int(nrn_id), index=int(index))
