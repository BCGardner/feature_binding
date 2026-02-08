from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..utils import labels_to_masks

__all__ = [
    "setup_journal_env",
    "save_figure",
    "imshow_cbar",
    "plot_images",
    "plot_errorband",
    "plot_shape_columns",
]

FIG_SIZE = (12, 6)
SHAPECAT_LABEL_MAPPING = {
    0: "concave",
    1: "convex",
}
JOURNAL_RCPARAMS = {
    "ps.fonttype": 42,
    "axes.labelsize": "small",
    "axes.titlesize": "medium",
    "xtick.labelsize": "small",
    "ytick.labelsize": "small",
    "legend.fontsize": "small",
    "figure.titlesize": "medium",
    "grid.color": "lightgray",
}


def setup_axes(axes: Optional[Axes] = None) -> Axes:
    if axes is None:
        axes = plt.subplot()
    return axes


def set_figsize(width: float, height: float, axes: Optional[Axes] = None) -> None:
    if axes is not None:
        f = axes.figure
    else:
        f = plt.gcf()
    if isinstance(f, Figure):
        f.set_size_inches(width, height)


def setup_journal_env(d: dict | None = None):
    _JOURNAL_RCPARAMS = JOURNAL_RCPARAMS.copy()
    if d is not None:
        _JOURNAL_RCPARAMS.update(d)
    for k, v in _JOURNAL_RCPARAMS.items():
        plt.rcParams[k] = v
    print("Setup journal printing specs")


def save_figure(
    figure: Figure,
    file_name: str | Path,
    format: str | None = None,
    dpi: int = 300,
    overwrite: bool = False,
    **kwargs,
) -> None:
    _file_name = Path(file_name).resolve()
    _format = _file_name.suffix[1:] if format is None else format
    if not _format:
        raise ValueError(f"Invalid format: '{format}'")
    if not _file_name.exists() or overwrite:
        figure.savefig(_file_name, format=_format, dpi=dpi, **kwargs)
        print(f"Saved: '{file_name}'")
    else:
        print(f"Skipped save: '{file_name}'")


def imshow_cbar(
    arr: npt.ArrayLike,
    axes: Optional[Axes] = None,
    cmax: Optional[float] = None,
    cmap: str | Colormap = plt.cm.Greys_r,  # type: ignore
    diverging: bool = False,
    label: Optional[str] = None,
    attach_cbar: bool = True,
    **kwargs,
) -> Axes:
    axes = setup_axes(axes)
    im = axes.imshow(arr, cmap=cmap, **kwargs)
    if attach_cbar:
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label=label)
    cmax_ = np.max(np.abs(arr)) if cmax is None else cmax
    if diverging:
        im.set_clim(-cmax_, cmax_)
    else:
        im.set_clim(None, cmax)
    return axes


def plot_images(
    data: Sequence[np.ndarray],
    num_rows: int = 2,
    plot_ticks: bool = True,
    titles: Optional[Sequence] = None,
    cmap: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: tuple = FIG_SIZE,
) -> npt.NDArray[Any]:
    f, axes = plt.subplots(
        num_rows, len(data) // num_rows, sharex=True, sharey=True, figsize=figsize
    )
    ax: Axes
    for idx, (sample, ax) in enumerate(zip(data, axes.flat)):
        ax.imshow(sample, cmap=cmap, vmin=vmin, vmax=vmax)
        if titles is not None:
            ax.set_title(titles[idx])
        if not plot_ticks:
            ax.set_xticks([])
            ax.set_yticks([])
    f.subplots_adjust(wspace=0.05, hspace=0.05)
    return axes


def plot_errorband(
    data: npt.ArrayLike,
    ylim: Optional[tuple] = None,
    xlim: Optional[tuple] = None,
    xscale: str = "linear",
    grid: bool = True,
    axes: Optional[Axes] = None,
    **plot_kwargs,
) -> Axes:
    xticks = plot_kwargs.pop("xticks", None)
    axes = setup_axes(axes)
    data = np.atleast_2d(data)
    xs = range(len(data)) if xticks is None else xticks
    avs, stdevs = np.mean(data, axis=1), np.std(data, axis=1)
    axes.plot(xs, avs, **plot_kwargs)
    axes.fill_between(xs, avs - stdevs, avs + stdevs, alpha=0.3)
    if ylim is not None:
        axes.set_ylim(*ylim)
    if xlim is not None:
        axes.set_xlim(*xlim)
    axes.set_xscale(xscale)
    if grid:
        axes.grid(True, alpha=0.5)
    return axes


def plot_shape_columns(
    data: Sequence[np.ndarray],
    labels: pd.DataFrame,
    side: str,
    target: int,
    highlight_target: bool = True,
    figsize: tuple = (9, 6),
    linewidth: float = 3,
    **plot_kwargs,
) -> npt.NDArray[Any]:
    masks = labels_to_masks(labels)
    # Gather contour-target images as NDArray with shape (num_rows x num_cols, H, W)
    images_grid = []
    target_mask = []
    for mask in masks.values():
        # Prioritise gathering target sides first
        indices = np.flatnonzero(mask)
        args = np.argsort(np.array(labels[side][indices] == target))[::-1]  # type: ignore
        indices = indices[args]
        images_grid.append([data[i] for i in indices])
        target_mask.append(np.array(labels[side][indices] == target))  # type: ignore
    target_mask = np.array(target_mask).T
    images_grid_ = np.asarray(images_grid).transpose([2, 3, 1, 0])
    images_grid_ = images_grid_.reshape(128, 128, -1).transpose(2, 0, 1)
    num_rows = len(images_grid_) // len(masks)
    # Plot columns of categorised images
    cmap = plot_kwargs.pop("cmap", "gray")
    axes = plot_images(
        list(images_grid_),
        num_rows,
        plot_ticks=False,
        figsize=figsize,
        cmap=cmap,
        **plot_kwargs,
    )
    if highlight_target:
        for ax, highlight in zip(axes.ravel(), target_mask.ravel()):
            if highlight:
                for spine in ax.spines.values():
                    spine.set_edgecolor("red")
                    spine.set_linewidth(linewidth)
    f = plt.gcf()
    keys = list(masks.keys())
    ax: Axes
    for idx, ax in enumerate(axes[-1, :]):
        key = keys[idx]
        xlabel = f"{key[0].capitalize()}: {SHAPECAT_LABEL_MAPPING[key[1].item()].capitalize()}"
        ax.set_xlabel(xlabel)
    f.tight_layout()
    return axes
