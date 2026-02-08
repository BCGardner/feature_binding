from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Optional, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from ... import ops
from .._types import RatesArray
from .. import measures
from .base import PNG

__all__ = [
    "get_occurrences_array",
    "get_onsets_array",
    "get_specific_measures",
    "get_sorted_measures_occ",
    "precision_recall",
    "f1_score",
    "get_thresholded_metrics",
]


def get_occurrences_array(polygrps: Sequence[PNG], num_reps: int, num_imgs: int, index: int,
                          duration: float, offset: Optional[float] = None,
                          description: str = 'PNG', nrn_cls: str = 'EXC') -> RatesArray:
    map_fn = partial(
        _get_png_occ, num_reps=num_reps, num_imgs=num_imgs,
        index=index, duration=duration, offset=offset,
        description=description, nrn_cls=nrn_cls
    )
    with ProcessPoolExecutor() as executor:
        png_occs = list(executor.map(map_fn, polygrps))
    return ops.concat_records(png_occs, 'png').transpose(..., 'png')


def get_onsets_array(polygrps: Sequence[PNG], num_reps: int, num_imgs: int,
                     null_img: int | None = None) -> xr.DataArray:
    """Get onset times for each PNG, with respect to each (`rep`, `img`).

    Args:
        polygrps (Sequence[PNG]): Sequence of PNGs containing `nrns` and `layers`.
        num_reps (int): Number of repetitions per image.
        num_imgs (int): Number of distinct images.
        null_img (int | None, optional): Drop PNGs associated with this image. Defaults to None.

    Returns:
        xr.DataArray: PNG onsets array, with dims (`png`, `rep`, `img`).
    """
    data = np.full((num_reps, num_imgs, len(polygrps)), np.nan)
    for i, polygrp in enumerate(polygrps):
        assert polygrp.reps is not None
        assert polygrp.times_rel is not None
        for j in range(num_imgs):
            idxs = _get_onset_indices(polygrp, j)
            reps = polygrp.reps[idxs]
            times_rel = polygrp.times_rel[idxs]
            data[reps, j, i] = times_rel
    onsets_array = xr.DataArray(
        data,
        dims=['rep', 'img', 'png'],
        coords=dict(
            rep=range(num_reps),
            img=range(num_imgs),
            png=range(len(polygrps))
        )
    )
    if null_img:
        return _drop_null_onsets(onsets_array, null_img=null_img)
    return onsets_array


def get_specific_measures(freq_array: RatesArray, targets: Optional[npt.ArrayLike] = None,
                          bin_width: Optional[int] = None) -> pd.DataFrame:
    return measures.get_specific_measures(freq_array, targets, bin_width, pk='png')


def get_sorted_measures_occ(freq_array: RatesArray, labels: pd.DataFrame, attribute: str,
                            target: int = 1, **kwargs) -> pd.DataFrame:
    return measures.get_sorted_measures_rates(
        freq_array, labels, attribute, target, apply_filter=False, pk='png', **kwargs
    )


def precision_recall(
    occ_array: RatesArray, labels: pd.DataFrame, side: str, target: int = 1,
    threshold: float = 0.0
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    occ_array = occ_array.transpose('rep', 'img', 'png')
    preds_array = occ_array.values > threshold  # Positive predictions
    # Positive / negative image indices
    ispositive = labels[side].values == target
    pos_idxs = np.flatnonzero(ispositive)
    neg_idxs = np.flatnonzero(~ispositive)
    # Predictions
    true_positives = preds_array[:, pos_idxs].sum((0, 1))
    false_negatives = (~preds_array[:, pos_idxs]).sum((0, 1))
    false_positives = preds_array[:, neg_idxs].sum((0, 1))
    # true_negatives = (~preds_array[:, neg_idxs]).sum((0, 1))
    # Metrics
    precision = true_positives / (true_positives + false_positives + 1E-16)
    recall = true_positives / (true_positives + false_negatives)
    return precision, recall


def f1_score(
    precision: npt.NDArray[np.float64], recall: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return 2 * precision * recall / (precision + recall)


def get_thresholded_metrics(
    occ_array: RatesArray, labels: pd.DataFrame, side: str, target: int = 1,
    threshold: float = 0.0, precision_min: float | None = 0.5
) -> pd.DataFrame:
    """Get precision, recall, F1-scores, for PNGs with min. precision.

    Args:
        occ_array (RatesArray): Occurrences array / frequency of PNG activations.
        labels (pd.DataFrame): Image annotations.
        side (str): Target attribute.
        target (int, optional): Target value (positive class). Defaults to 1.
        threshold (float, optional): Min. frequency for positive classification. Defaults to 0.0.
        precision_min (float, optional): Min. precision for inclusion. Defaults to 0.5.

    Returns:
        pd.DataFrame: Contains `precision`, `recall`, `F1-score`, indexed by PNG id.
    """
    precision, recall = precision_recall(
        occ_array, labels, side, target, threshold
    )
    if precision_min is not None:
        mask = precision > precision_min
    else:
        mask = precision >= 0
    precision, recall = precision[mask], recall[mask]
    scores = f1_score(precision, recall)
    return pd.DataFrame(
        data={'precision': precision, 'recall': recall, 'score': scores},
        index=occ_array['png'].values[mask]
    )


def _get_png_occ(polygrp: PNG, num_reps: int, num_imgs: int, index: int,
                 duration: float, offset: Optional[float] = None,
                 description: str = 'PNG', nrn_cls: str = 'EXC') -> RatesArray:
    assert polygrp.reps is not None
    assert polygrp.imgs is not None

    layer = polygrp.layers[index]
    nrn = polygrp.nrns[index]
    occ_array = xr.DataArray(
        np.zeros((num_reps, num_imgs)), dims=['rep', 'img'],
        coords=dict(nrn_cls=nrn_cls, layer=layer, nrn=nrn, index=index,
                    img=range(num_imgs), rep=range(num_reps)),
        attrs={'unit': 'hertz', 'description': f'{description} occurrence rates',
               'duration': duration, 'offset': offset}
    )
    coords, counts = np.unique(np.vstack((polygrp.reps, polygrp.imgs)).T,
                               axis=0, return_counts=True)
    for coord, count in zip(coords, counts):
        occ_array[*coord] = count / duration * 1E3
    return occ_array


def _get_onset_indices(polygrp: PNG, img: int) -> npt.NDArray[np.int64]:
    """Get indices of first PNG occurrences, gathered across repetitions of given image.
    """
    img_idxs = np.flatnonzero(polygrp.imgs == img)
    if len(img_idxs):
        reps = polygrp.reps[img_idxs]  # type: ignore
        _, reps_idxs = np.unique(reps, return_index=True)
        return img_idxs[reps_idxs]
    else:
        return np.array([], dtype=np.int64)


def _drop_null_onsets(onset_times: xr.DataArray, null_img: int,
                      max_reps: int | None = None) -> xr.DataArray:
    """Eliminate PNG entries which occur across more than `max_reps`
    separate repetitions in response to the image ID `null_img`.
    """
    _max_reps = 0 if max_reps is None else max_reps
    unique_ids = np.flatnonzero(
        onset_times.sel(img=null_img).notnull().sum('rep') <= _max_reps
    )
    return onset_times.sel(png=unique_ids)
