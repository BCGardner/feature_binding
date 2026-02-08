from itertools import combinations, product
from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Patch, Ellipse, Rectangle

from .base import setup_axes, set_figsize
from ._utils import update_dict
from hsnn import ops
from hsnn.analysis import get_centroid, get_centroid_stdevs
from hsnn.analysis.png import PNG

__all__ = [
    "hist_weights",
    "plot_traceback",
    "create_bbox",
    "plot_traceback_imageset"
]

pidx = pd.IndexSlice


def hist_weights(syn_params: pd.DataFrame, projs: Optional[Sequence[str]] = None,
                 bins: int | Sequence[float] = 50, annotations: bool = True, show_yticks: bool = False,
                 text_kwargs: dict | None = None, hist_kwargs: dict | None = None,
                 figsize: Optional[tuple] = None) -> Axes:
    """Plot histograms of synaptic weights for each layer-projection combination.

    Args:
        syn_params (pd.DataFrame): DataFrame indexed by ('layer', 'proj', 'pre', 'post')
            containing a 'w' column with synaptic weights.
        projs (Optional[Sequence[str]], optional): Projections to include (e.g., 'FF', 'E2E', 'FB').
            Defaults to all available projections in syn_params.
        bins (int | Sequence[float], optional): Number of bins or explicit bin edges passed to
            Axes.hist. Defaults to 50.
        annotations (bool, optional): Whether to show example annotations in select panels.
            Defaults to True.
        show_yticks (bool, optional): Whether to display y-axis ticks. Defaults to False.
        text_kwargs (dict | None, optional): Extra kwargs forwarded to the annotation text
            (e.g., fontsize, bbox). Defaults to None.
        hist_kwargs (dict | None, optional): Extra kwargs forwarded to Axes.hist
            (e.g., density, color). Defaults to None; uses {'density': True} by default.
        figsize (Optional[tuple], optional): Figure size (width, height) in inches. When None,
            it scales with the number of projections. Defaults to None.

    Returns:
        np.ndarray[Axes]: Grid of axes with shape (n_layers, n_projs).
    """
    layers = syn_params.index.unique('layer').values
    proj_text_map = {
        'FF':   'Feedforward',
        'E2E':  'Lateral',
        'FB':   'Feedback'
    }
    if projs is None:
        projs_avail = syn_params.index.unique('proj')
        projs = [proj for proj in proj_text_map.keys() if proj in projs_avail]
    if figsize is None:
        figsize = (10/3 * len(projs), 8)

    weights_ser: pd.DataFrame = syn_params.loc[pidx[layers, projs], ['w']]  # type: ignore
    weights_ser = weights_ser.droplevel(['pre', 'post'])
    groups = weights_ser.groupby(['layer', 'proj'])

    _hist_kwargs = hist_kwargs or {'density': True}
    f, axes = plt.subplots(len(layers), len(projs), sharex=True, sharey=False,
                           figsize=figsize)
    ax: Axes
    for ax, index in zip(axes.flat, product(layers, projs)):
        if index in groups.indices:
            group = groups.get_group(index)
            values = np.asarray(group).ravel()
            values = values[~np.isnan(values)]
            ax.hist(values, bins, **_hist_kwargs)
        else:
            ax.axis('off')

    for ax, proj in zip(axes[0, :], projs):
        ax.set_title(proj_text_map[proj], pad=15, fontweight='bold')  # pad=20, fontsize=14

    for ax in axes[-1, :]:
        ax.set_xlabel('W')
        ax.set_xticks([0.0, 0.5, 1.0])

    for ax, layer in zip(axes[:, 0], layers):
        ax.set_ylabel(f"L{layer}", rotation=0, labelpad=15)  # labelpad=20, fontsize=12

    if not show_yticks:
        for ax in axes.flatten():
            ax.set_yticks([])

    # Annotations
    if annotations:
        _text_kwargs = text_kwargs or {}
        x, y = 0.15, 0.86
        _axes_text(axes[0, 0], x, y, r'e.g. $\mathrm{L0} \rightarrow \mathrm{L1}$',
                   **_text_kwargs)
        try:
            _axes_text(axes[0, 1], x, y, r'$\mathrm{L1} \leftrightarrow \mathrm{L1}$',
                       **_text_kwargs)
        except IndexError:
            pass
        try:
            _axes_text(axes[1, 2], x, y, r'$\mathrm{L2} \rightarrow \mathrm{L1}$',
                       **_text_kwargs)
        except IndexError:
            pass
    f.tight_layout()
    return axes


def plot_traceback(sensitivities: pd.Series, image: npt.NDArray,
                   nrn_id: Optional[int] = None, layer_shape: Optional[Sequence] = None,
                   sensitivities_max: Optional[float] = None, color: str = 'blue',
                   cmap: str = 'gray', vmin: float = 0.0, vmax: float = 255.0,
                   axes: Optional[Axes] = None, figsize: Optional[tuple] = None,
                   sel_kwargs: Optional[dict] = None, nrn_kwargs: Optional[dict] = None) -> Axes:
    """Overlay per-pixel sensitivities on an image as alpha-scaled scatter points.

    Alpha is sensitivities normalized to [0, 1]: divide by sensitivities_max (if given)
    or by the series maximum. Raises ValueError if any alpha > 1. Optionally marks the
    source neuron coordinate decoded from nrn_id and layer_shape.

    Args:
        sensitivities: Series with MultiIndex ('X', 'Y') containing sensitivity magnitudes.
        image: 2D (H, W) or 3D (H, W, C) array used as the background.
        nrn_id: Flat neuron index to highlight; ignored if None.
        layer_shape: (H, W) of the neuron layer; required with nrn_id.
        sensitivities_max: Denominator for alpha normalisation; if None, uses series max.
        color: Color of the sensitivity points.
        cmap: Colormap for the background image.
        vmin: Minimum intensity for image display.
        vmax: Maximum intensity for image display.
        axes: Axes to draw on; created if None.
        figsize: Figure size when axes is created.
        sel_kwargs: Extra kwargs for the sensitivity scatter (e.g., s, edgecolors).
        nrn_kwargs: Extra kwargs for the neuron marker.

    Raises:
        ValueError: If any computed alpha exceeds 1.0.

    Returns:
        Axes: Matplotlib axes with the composed traceback plot.
    """
    axes = setup_axes(axes)
    input_shape = image.shape
    xs = sensitivities.index.get_level_values('X').values
    ys = sensitivities.index.get_level_values('Y').values
    if len(sensitivities):
        if sensitivities_max:
            alphas = np.array(sensitivities / sensitivities_max)
        else:
            alphas = np.array(sensitivities / np.max(sensitivities))
        if alphas.max() > 1.0:
            raise ValueError(f"alpha value greater than 1.0 ({alphas.max()})")
    else:
        alphas = np.array([], dtype=float)
    # Plot sensitivites as heatmap over image
    axes.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    colors = [color for _ in xs]
    if len(alphas):
        plot_kwargs = {
            's': 25,
            'edgecolors': 'k',
            'linewidths': 0.2
        }
        plot_kwargs = update_dict(plot_kwargs, sel_kwargs)
        axes.scatter(xs, ys, c=colors, alpha=alphas, **plot_kwargs)  # type: ignore
    # Plot originating neuron coord
    if nrn_id and layer_shape:
        nrn_kwargs = {} if nrn_kwargs is None else nrn_kwargs
        _plot_neuron_coord(nrn_id, layer_shape, input_shape, axes, **nrn_kwargs)
    axes.set_xticks([])
    axes.set_yticks([])
    if figsize is not None:
        set_figsize(*figsize, axes=axes)
    return axes


def create_bbox(sensitivities: pd.Series, scale: float = 4.0, width: float | None = None,
                height: float | None = None, shape: str = 'rectangle', **kwargs) -> Patch | None:
    centroid = get_centroid(sensitivities)
    centroid_stdevs = get_centroid_stdevs(sensitivities)
    if centroid is not None:
        x_av, y_av = centroid
    else:
        return None
    if centroid_stdevs is not None:
        x_stdev, y_stdev = centroid_stdevs
    else:
        return None

    _kwargs: dict = {'edgecolor': 'red', 'facecolor': 'none'}
    _kwargs.update(**kwargs)
    if width and height:
        _width, _height = width, height
    else:
        _width, _height = (scale * x_stdev, scale * y_stdev)
    if shape == 'rectangle':
        xy = (x_av - _width / 2, y_av - _height / 2)
        return Rectangle(xy, width=_width, height=_height, **_kwargs)
    elif shape == 'ellipse':
        return Ellipse((x_av, y_av), _width, _height, **_kwargs)
    elif shape == 'circle':
        radius = max(_width, _height)
        return Ellipse((x_av, y_av), radius, radius, **_kwargs)
    else:
        raise ValueError(f"Invalid shape '{shape}'")


def plot_traceback_imageset(sensitivities_array: Sequence[pd.Series], imageset: Sequence[np.ndarray],
                            nrn_id: Optional[int] = None, layer_shape: Optional[Sequence] = None,
                            num_rows: int = 2, figsize: Optional[tuple] = None, **kwargs) -> Any:
    if len(imageset) % num_rows != 0:
        raise ValueError(f"Invalid num_rows: {num_rows}")
    f, axes = plt.subplots(
        num_rows, len(imageset) // num_rows, sharex=True, sharey=True,
        figsize=figsize
    )
    if 'sensitivities_max' not in kwargs:
        kwargs['sensitivities_max'] = \
            np.max([np.max(sens) for sens in sensitivities_array if len(sens)])
    for idx, ax in enumerate(axes.flatten()):
        plot_traceback(
            sensitivities_array[idx], imageset[idx], nrn_id=nrn_id, layer_shape=layer_shape,
            axes=ax, **kwargs
        )
    f.tight_layout()
    return axes


def print_png(polygrp: PNG, syn_params: pd.DataFrame):
    """Print PNG attributes as human-friendly text.

    Args:
        polygrp (PNG): PNG conforming to a three-neuron HFB circuit.
        syn_params (pd.DataFrame): Contains `w` and `delay`,
            index: (`layer`, `proj`, `pre`, `post`).
    """
    # 1) LOW -> HIGH
    # 2) LOW -> BIND
    # 3) HIGH -> BIND
    assert len(polygrp.layers) == 3, "PNG must conform to a three-neuron HFB circuit"
    idx_pairs = list(combinations(range(3), 2))
    for i, j in idx_pairs:
        layer_pre = polygrp.layers[i]
        layer_post = polygrp.layers[j]
        nrn_pre = polygrp.nrns[i]
        nrn_post = polygrp.nrns[j]
        t_diff = polygrp.lags[j] - polygrp.lags[i]
        proj = ops.layers_to_proj(layer_post, layer_pre)

        conn_vals = syn_params.xs((layer_post, proj, nrn_pre, nrn_post))
        print(f"{proj}: L{layer_pre} {nrn_pre} -> L{layer_post} {nrn_post}")
        print(f"t_diff: {t_diff}; delay: {conn_vals['delay']}; w: {conn_vals['w']:.3f}")
        print()


def _axes_text(axes: Axes, x: float, y: float, text: str, **kwargs):
    text_kwargs = {
        'fontsize': 12,
        'va': 'top',
        'ha': 'left',
        'bbox': dict(facecolor='yellow', alpha=0.1)
    }
    text_kwargs.update(**kwargs)
    axes.text(x, y, text, transform=axes.transAxes, **text_kwargs)


def _plot_neuron_coord(nrn_id: int, layer_shape: Sequence,
                       plot_shape: Optional[Sequence] = None,
                       axes: Optional[Axes] = None, **kwargs) -> Axes:
    axes = setup_axes(axes)
    nrn_coord = np.array(
        np.unravel_index([nrn_id], layer_shape)
    )[::-1].squeeze()
    if plot_shape is not None:
        nrn_coord = nrn_coord * plot_shape[0] / layer_shape[0]
    plot_kwargs = {
        's': 64,
        'c': 'red',
        'edgecolors': 'k',
        'linewidths': 1.0
    }
    plot_kwargs.update(**kwargs)
    axes.scatter(*nrn_coord, **plot_kwargs)
    return axes
