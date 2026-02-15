from copy import copy
from typing import Hashable, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from ._types import CountsArray, RatesArray
from .base import infer_rates

__all__ = [
    "get_specific_measures",
    "get_sorted_measures_rates",
    "get_specific_measures_side",
    "get_filtered_measures",
    "get_combined_measures",
]


def get_specific_measures(
    rates_array: RatesArray,
    targets: Optional[npt.ArrayLike] = None,
    bin_width: Optional[int] = None,
    pk: Hashable = "nrn",
) -> pd.DataFrame:
    _dims = {"rep", "img", pk}
    if set(rates_array.dims) != _dims:
        raise ValueError(f"Input array dims != {_dims}")
    counts_array = _as_spike_counts(rates_array)
    # Split individual measures according to these target stimuli
    if targets is None:
        targets = counts_array["img"].values
    else:
        targets = np.asarray(targets)
    unique_targets, weights = np.unique(targets, return_counts=True)
    masks = [np.asarray(targets == target) for target in unique_targets]
    # Get measures w.r.t. the primary key
    informs = []
    for _, group in counts_array.groupby(pk):
        count_freqs = _conditioned_count_freqs(group, bin_width=bin_width)
        count_freqs_ = np.array([np.mean(count_freqs[mask], axis=0) for mask in masks])
        informs.append(_specific_information(count_freqs_, weights=weights))
    informs_ = np.array(informs)
    return pd.DataFrame(
        data={target: informs_[:, i] for i, target in enumerate(unique_targets)},
        index=counts_array[pk].values,
    )


def get_sorted_measures_rates(
    rates_array: RatesArray,
    labels: pd.DataFrame,
    attribute: str,
    target: int = 1,
    apply_filter: bool = True,
    pk: Hashable = "nrn",
    **kwargs,
) -> pd.DataFrame:
    """Rank orders the single-entity specific information conveyed about a stimulus according to both
    the measure and frequencies.

    Args:
        rates_array (RatesArray): Firing rate responses, dims (rep, img, `pk`).
        labels (pd.DataFrame): Image annotations containing labels.
        attribute (str): The column name in `labels` that contains the target attribute.
        target (int, optional): The value of the target category measured. Defaults to 1.
        apply_filter (bool, optional): If True, filters the results to include only entities where
            the positive response rate is greater than the negative response rate. Defaults to True.
        pk (Hashable, optional): Dimension for which specific measures are determined. Defaults to 'nrn'.

    Returns:
        pd.DataFrame: Rank-ordered specific measures indexed by entity indices (`pk`),
        with columns `measure`, and mean firing rates `neg` and `pos` corresponding to
        negative and positive stimulus categories, respectively.
    """
    targets = np.asarray(labels[attribute])
    bin_width = kwargs.get("bin_width", None)
    bin_measures = kwargs.get("measures_bin", 0.01)
    bin_rates = kwargs.get("rates_bin", 1e3 / rates_array.attrs["duration"])
    # Binned measures
    specific_measures = get_specific_measures(rates_array, targets, bin_width, pk)[
        target
    ]
    nrn_idxs = np.array(specific_measures.index)
    specific_measures = _bin_values(specific_measures, bin_measures)
    # Binned rates
    pos_idxs = np.flatnonzero(targets == target)
    neg_idxs = np.setdiff1d(range(len(targets)), pos_idxs)
    rates_pos = _bin_values(
        rates_array.sel(img=pos_idxs).mean(dim=["rep", "img"]).values, bin_rates
    )
    rates_neg = _bin_values(
        rates_array.sel(img=neg_idxs).mean(dim=["rep", "img"]).values, bin_rates
    )
    # Sort by measure, -rates_neg, rates_pos
    sorted_data = sorted(
        zip(nrn_idxs, specific_measures, rates_neg, rates_pos),
        key=lambda x: (x[1], -x[2], x[3]),
        reverse=True,
    )
    df_sorted = pd.DataFrame(
        sorted_data, columns=[pk, "measure", "neg", "pos"]
    ).set_index(pk)
    if apply_filter:
        return df_sorted.query("pos > neg")
    else:
        return df_sorted


def get_specific_measures_side(
    rates_array: xr.DataArray,
    labels: pd.DataFrame,
    target: int,
) -> dict[str, pd.DataFrame]:
    """Get single-neuron specific information per boundary side."""
    attributes = labels.drop("image_id", axis=1).columns.tolist()
    measures_sides = {}
    for attribute in attributes:
        measures_sides[attribute] = get_sorted_measures_rates(
            rates_array, labels, attribute, target
        )
    return measures_sides


def get_filtered_measures(
    rates_array: RatesArray,
    labels: pd.DataFrame,
    attribute: str,
    target: int = 1,
    pk: Hashable = "nrn",
    **kwargs,
) -> pd.DataFrame:
    """Computes the single-entity specific information conveyed about a stimulus
    according to both the measure and frequencies.

    Args:
        rates_array: RatesArray
        labels: Image annotations containing labels.
        attribute: The column name in `labels` that contains the target attribute.
        target: The value of the target category measured. Defaults to 1.
        pk: Dimension for which specific measures are determined. Defaults to "nrn".

    Returns:
        Specific measures indexed by entity indices (`pk`),
        with columns `measure`, and mean firing rates `neg` and `pos` corresponding to
        negative and positive stimulus categories, respectively.
    """
    targets = np.asarray(labels[attribute])
    # Binned measures
    specific_measures = get_specific_measures(rates_array, targets, pk=pk)[target]
    nrn_idxs = np.array(specific_measures.index)
    # Firing rates (averaged)
    pos_idxs = np.flatnonzero(targets == target)
    neg_idxs = np.setdiff1d(range(len(targets)), pos_idxs)
    rates_pos = rates_array.sel(img=pos_idxs).mean(dim=["rep", "img"]).values
    rates_neg = rates_array.sel(img=neg_idxs).mean(dim=["rep", "img"]).values
    # Filtered measures
    df = pd.DataFrame(
        zip(nrn_idxs, specific_measures, rates_neg, rates_pos),
        columns=[pk, "measure", "neg", "pos"],
    ).set_index(pk)
    df.loc[df["pos"] <= df["neg"], "measure"] = 0.0
    return df


def get_combined_measures(
    records: xr.DataArray,
    labels: pd.DataFrame,
    target: int,
    duration: float,
    offset: float,
    layer: int = 4,
    nrn_cls: str = "EXC",
) -> pd.DataFrame:
    """Computes the single-neuron specific information conveyed about a stimulus from
    spike recordings with respect to all dataset attributes. Specific measures for
    positive firing rates which are less than negative rates are set to zero.

    Args:
        records: Spike recordings.
        labels: Dataset annotations, containing attributes.
        target: Target feature value conveyed (e.g. convex=1).
        duration: Observation period starting from offset.
        offset: Determines the start of the observation period.
        layer: Selected layer. Defaults to 4.
        nrn_cls: Selected neuron class. Defaults to 'EXC'.

    Returns:
        Information conveyed per attribute for all neurons.
    """
    attributes = labels.drop("image_id", axis=1).columns.tolist()

    rates_array = infer_rates(
        records.sel(layer=layer, nrn_cls=nrn_cls), duration, offset
    )
    measures_sides = {}
    for attribute in attributes:
        measures_sides[attribute] = get_filtered_measures(
            rates_array, labels, attribute, target
        )["measure"].values
    specific_measures = pd.DataFrame(
        data=measures_sides,
        index=pd.Index(rates_array["nrn"].values, name="nrn"),
    )
    specific_measures.columns = pd.Index(specific_measures.columns, name=target)
    return specific_measures


def _as_spike_counts(
    rates: RatesArray, duration: Optional[float] = None, conversion: float = 1e-3
) -> CountsArray:
    if duration is None:
        duration = rates.attrs["duration"]
    attrs = copy(rates.attrs)
    attrs.update(unit=None, description="Spike counts")
    data = np.array(rates * conversion * duration, dtype=int)
    return xr.DataArray(
        data, coords=rates.coords, dims=rates.dims, name="counts", attrs=attrs
    )


def _conditioned_count_freqs(
    spike_counts: CountsArray, bin_width: Optional[int] = None
) -> npt.NDArray[np.float64]:
    """Gets the fraction of trials where each choice (or range) of spike count occurs.

    Args:
        spike_counts (CountsArray): The number of spike responses, dims ('rep', 'img').
        bin_width (Optional[int], optional): Smoothes the count frequency estimates. Defaults to None.

    Raises:
        ValueError: spike_counts array must contain dims: {'rep', 'img'}.

    Returns:
        npt.NDArray[np.float64]: The conditional probabilities of all possible response, shape (num_img, num_bins).
    """
    _dims = {"rep", "img"}
    if set(spike_counts.dims) != _dims:
        raise ValueError(f"spike_counts dims != {_dims}")
    values = spike_counts.transpose("img", "rep").values
    if bin_width is None:
        minlength = values.max() + 1
        return (
            np.array(
                [np.bincount(counts, minlength=minlength) for counts in values],
                dtype=float,
            )
            / values.shape[-1]
        )
    else:
        bin_edges = np.arange(0, values.max() + bin_width, bin_width)
        freqs = []
        for vs in values:
            counts, _ = np.histogram(vs[vs > 0], bins=bin_edges)
            counts = np.append(np.sum(vs == 0), counts)
            freqs.append(counts / values.shape[-1])
        return np.array(freqs)


def _specific_information(
    pr_responses_cond: npt.ArrayLike, weights: Optional[npt.ArrayLike] = None
) -> npt.NDArray[np.float64]:
    """Gets the information conveyed about each stimulus by a single entity's responses, I(s, R).

    Args:
        pr_responses_cond (npt.ArrayLike): Stimulus-conditioned response probabilities, shape (num_stim, num_bins).
        weights (Optional[npt.ArrayLike], optional): Averaging weights (over stimuli). Defaults to None.

    Returns:
        npt.NDArray[np.float64]: Stimulus-specific information measures, shape (num_stim,).
    """
    pr_responses_cond_ = np.asarray(pr_responses_cond)
    if not pr_responses_cond_.ndim == 2:
        raise ValueError(f"expected ndim '2', got '{pr_responses_cond_.ndim}'")
    if not np.allclose(np.sum(pr_responses_cond, axis=1), 1):
        raise ValueError(
            "invalid argument for 'pr_responses_cond': all rows do not sum to one"
        )
    pr_rates = np.average(pr_responses_cond_, axis=0, weights=weights)
    return np.array(
        [_sum_log_numerical(probs_cond, pr_rates) for probs_cond in pr_responses_cond_]
    )


def _bin_values(values: npt.ArrayLike, bin_size: float) -> npt.NDArray[np.float64]:
    return np.round(np.array(values, dtype=float) / bin_size) * bin_size


def _sum_log_numerical(
    probs_cond: npt.NDArray[np.float64],
    probs_total: npt.NDArray[np.float64],
    epsilon: float = 1e-16,
) -> np.float64:
    mask = probs_cond > 0
    return np.sum(
        probs_cond[mask] * np.log2(probs_cond[mask] / (probs_total[mask] + epsilon))
    )
