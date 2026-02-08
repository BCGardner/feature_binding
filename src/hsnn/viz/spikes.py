from typing import Optional

import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.axes import Axes

from hsnn.core import SpikeRecord
from hsnn.core.types import SpikeEvents, SpikeTrains
from hsnn import ops
from .base import setup_axes

__all__ = [
    "plot_raster",
    "plot_raster_xr"
]


def plot_raster(recording: SpikeEvents | SpikeTrains, duration: float, t_start: float = 0,
                relative_timings: bool = False, axes: Optional[Axes] = None,
                color: str = 'k', xlabel: str = 'Time [ms]', ylabel: str = 'Neuron ID',
                marker: str ='.', **kwargs) -> Axes:
    spike_events = ops.as_spike_events(recording)
    spike_ids, spike_times = ops.mask_recording(spike_events, duration, t_start, relative_timings)
    axes = setup_axes(axes)
    _plot_kwargs: dict = {
        'markersize': 10,
        'markeredgewidth': 0,
    }
    _plot_kwargs.update(**kwargs)
    axes.plot(spike_times, spike_ids, marker, color=color, **_plot_kwargs)
    if relative_timings:
        axes.set_xlim(*[0.0, duration])
    else:
        axes.set_xlim(*[t_start, t_start + duration])
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    return axes


def plot_raster_xr(records: xr.DataArray, duration: Optional[float] = None, t_start: float = 0,
                   relative_timings: bool = False, markersize: float = 5,
                   figsize: tuple[float, float] = (8, 10)) -> Axes:
    dims = set(records.dims)
    assert dims.issubset({'layer', 'nrn_cls'})
    duration = float(records.item(0).duration) if duration is None else duration
    f, axes = plt.subplots(len(records['layer']), sharex=True, figsize=figsize)
    col_exc = 'r' if 'nrn_cls' in dims else 'k'
    for idx, (layer, rec_grps) in enumerate(records.groupby('layer')):
        ax = axes[idx]
        rec: SpikeRecord
        if 'nrn_cls' in dims:
            rec = rec_grps.sel(nrn_cls='EXC').item()
        else:
            rec = rec_grps.item()
        num_nrns = rec.num_nrns
        plot_raster(rec.spike_events, duration, t_start, relative_timings,
                    axes=ax, markersize=markersize, color=col_exc)
        if 'nrn_cls' in dims:
            rec = rec_grps.sel(nrn_cls='INH').item()
            if rec is not None:
                spike_events = (rec.spike_events[0] + num_nrns, rec.spike_events[1])
                plot_raster(spike_events, duration, t_start,
                            relative_timings, axes=ax,
                            markersize=markersize, color='b')
                num_nrns += rec.num_nrns
        ax.set_ylim([0, num_nrns])
        ax.set_ylabel(f'L{layer}', rotation=0, labelpad=20)
        ax.set_xlabel('')
    axes[-1].set_xlabel('Time [ms]')
    f.tight_layout()
    return axes
