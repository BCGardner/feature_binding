from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from scipy.interpolate import griddata

from .base import setup_axes, set_figsize

__all__ = [
    "plot_loss_contour",
    "plot_specific_measures"
]


def plot_loss_contour(zs: npt.ArrayLike, xs: npt.ArrayLike, ys: npt.ArrayLike,
                      xlim: tuple, ylim: tuple, xlabel: Optional[str] = None,
                      ylabel: Optional[str] = None, label: str = 'loss',
                      logscale: bool = True, grid_resolution: int = 100, method: str = 'linear',
                      contour_levels: int = 15, cmap: str = 'viridis') -> Axes:
    grid_x, grid_y = np.mgrid[xlim[0]:xlim[1]:complex(grid_resolution),
                              ylim[0]:ylim[1]:complex(grid_resolution)]
    if logscale:
        zs = np.log10(zs)
    # Interpolate the loss values onto this grid
    grid_z = griddata((xs, ys), zs, (grid_x, grid_y), method=method)
    f, axes = plt.subplots()
    im = axes.contourf(grid_x, grid_y, grid_z, contour_levels, cmap=cmap)
    f.colorbar(im, label=label)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.scatter(xs, ys, color='red', marker='o')
    f.set_size_inches(10, 8)
    plt.show()
    return axes


def plot_specific_measures(specific_measures: npt.ArrayLike, max_info: Optional[float] = None,
                           layer: Optional[str] = None, side: Optional[str] = None,
                           grid: bool = True, axes: Optional[Axes] = None,
                           figsize: Optional[tuple] = None) -> Axes:
    axes = setup_axes(axes)
    ranked_measures = np.sort(specific_measures)[::-1].round(8)
    axes.plot(ranked_measures)
    axes.set_xscale('log')
    axes.set_xlabel('Neuron rank')
    axes.set_ylabel('I(s, R)')
    if max_info is not None:
        axes.set_ylim(*[0.0, 1.05 * max_info])
    if grid:
        axes.grid(True, alpha=0.4)
    if layer and side:
        axes.set_title(f'L{layer} neuron selectivity: {side} contour')
    if figsize is not None:
        set_figsize(*figsize, axes=axes)
    return axes
