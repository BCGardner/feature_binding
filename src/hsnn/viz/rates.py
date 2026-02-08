from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.axes import Axes

from hsnn.utils import labels_to_masks
from hsnn.core import SpikeRecord
from hsnn.analysis._types import RatesArray
from .base import setup_axes, set_figsize, SHAPECAT_LABEL_MAPPING
from ._utils import get_bar_data, get_violin_data

__all__ = [
    "topographic_rates",
    "hist_rates",
    "plot_contour_selectivity",
    "plot_image_selectivity",
]


def topographic_rates(
    records: xr.DataArray,
    plot_ticks: bool = True,
    cmap: str = "inferno",
    vmax: Optional[float] = None,
    figsize: tuple = (6, 12),
) -> Axes:
    records = records.transpose("layer", "nrn_cls")
    ax: Axes
    f, axes = plt.subplots(len(records), 2, figsize=figsize)
    for rec, ax in zip(records.values.flat, axes.flat):
        assert isinstance(rec, SpikeRecord)
        if rec is not None:
            shape = [int(np.sqrt(rec.num_nrns))] * 2
            ax.imshow(rec.rates[1].reshape(shape), cmap=cmap, vmax=vmax)
            ax.set_title(str(rec.name))
            if not plot_ticks:
                ax.set_xticks([])
                ax.set_yticks([])
        else:
            ax.remove()
    f.tight_layout()
    # f.subplots_adjust(wspace=0.4, hspace=0.4)
    return axes


def hist_rates(
    records: xr.DataArray,
    bins: Optional[Any] = 50,
    density: bool = True,
    xmax: Optional[Sequence] = None,
    figsize: tuple = (8, 10),
    yticks: bool = True,
    xlabel: str = "",
) -> Axes:
    records = records.transpose("layer", "nrn_cls")
    f, axes = plt.subplots(len(records), 2, figsize=figsize)
    rec: SpikeRecord
    ax: Axes
    for (i, j), ax in np.ndenumerate(axes):
        rec = records[i, j].item()
        if rec is not None:
            rates = rec.rates[1][rec.rates[1] > 0]
            ax.hist(rates, bins, density=density)
            ax.set_title(str(rec.name))
            if xmax is not None:
                ax.set_xlim(*[0, xmax[j]])
            if not yticks:
                ax.set_yticks([])
        else:
            ax.remove()
    if len(xlabel):
        for ax in axes[-1, :]:
            ax.set_xlabel(xlabel)
    else:
        f.suptitle("Firing rates (Hz)", size="x-large")
    f.tight_layout()
    return axes


def plot_contour_selectivity(
    rates: RatesArray,
    nrn_id: int,
    labels: pd.DataFrame,
    rates_pre: RatesArray | None = None,
    bar_width: float = 0.8,
    violinplot: bool = True,
    show_xlabels: bool = True,
    grid: bool = True,
    alpha: float = 0.2,
    rotate_xticklabels: bool = True,
    axes: Optional[Axes] = None,
    figsize: Optional[tuple] = None,
) -> Axes:
    assert set(rates.dims) == {"img", "nrn", "rep"}
    axes = setup_axes(axes)
    masks = labels_to_masks(labels)

    def _plot_bar(rates: RatesArray, split: bool = False, left_aligned: bool = False):
        factor = 1 / 2 + 0.05
        rates_combined = get_bar_data(rates, nrn_id, masks)
        xs = np.arange(len(rates_combined))
        width = bar_width / 2 if split else bar_width
        label = "Untrained" if left_aligned else "Trained"
        if split:
            offset = width * factor
            xs = xs - offset if left_aligned else +xs + offset
        axes.bar(xs, rates_combined, width, label=label)

    if violinplot:
        rates_combined = get_violin_data(rates, nrn_id, masks)
        axes.violinplot(rates_combined)
    else:
        if rates_pre is not None:
            _plot_bar(rates_pre, split=True, left_aligned=True)
            _plot_bar(rates, split=True, left_aligned=False)
        else:
            _plot_bar(rates)

    # axes.bar(range(len(r_avs)), r_avs, yerr=r_stdevs, error_kw=dict(lw=1, capsize=5, capthick=1))
    axes.set_title(f"L{int(rates.layer)} neuron #{nrn_id}: Contour selectivity")
    axes.set_ylabel("Firing rate (Hz)")
    if violinplot and grid:
        axes.grid(True, alpha=alpha)
    if show_xlabels:
        xticklabels = [
            f"{key[0].capitalize()}: {SHAPECAT_LABEL_MAPPING[key[1].item()].capitalize()}"
            for key in masks.keys()
        ]
        xticklabel_kwargs = (
            {"rotation": 45, "ha": "right"} if rotate_xticklabels else {}
        )
        if violinplot:
            axes.set_xticks(np.arange(1, len(xticklabels) + 1))
            axes.set_xticklabels(xticklabels, **xticklabel_kwargs)
        else:
            axes.set_xticks(
                np.arange(len(xticklabels)), xticklabels, **xticklabel_kwargs
            )
    else:
        axes.set_xticklabels("")
    if figsize is not None:
        set_figsize(*figsize, axes=axes)
        f = plt.gcf()
        f.tight_layout()
    return axes


def plot_image_selectivity(
    rates: RatesArray,
    nrn_id: int,
    grid: bool = True,
    alpha: float = 0.2,
    axes: Optional[Axes] = None,
    figsize: Optional[tuple] = None,
) -> Axes:
    assert set(rates.dims) == {"img", "nrn", "rep"}
    axes = setup_axes(axes)
    axes.violinplot(rates.sel(nrn=nrn_id).values)
    axes.set_xticks(np.arange(1, len(rates.coords["img"]) + 1))
    axes.set_xticklabels(rates.coords["img"].values)
    if grid:
        axes.grid(True, alpha=alpha)
    axes.set_title(f"L{int(rates.layer)} neuron #{nrn_id}: Image selectivity")
    axes.set_ylabel("Firing rate (Hz)")
    axes.set_xlabel("Image ID")
    if figsize:
        set_figsize(*figsize, axes=axes)
    return axes
