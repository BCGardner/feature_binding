import pytest

import numpy as np
import xarray as xr

from hsnn.core import SpikeRecord
from hsnn import ops


@pytest.fixture(scope='module')
def spike_records() -> xr.DataArray:
    num_nrns = 3
    duration = 20
    data = np.array([[SpikeRecord(num_nrns, duration, [0, 0, 2], [15, 17, 18]),
                      SpikeRecord(num_nrns, duration, [0, 2],    [12, 19])],
                     [SpikeRecord(num_nrns, duration, [0, 2, 2], [14, 16, 17]),
                      SpikeRecord(num_nrns, duration, [0, 1],    [13, 14])]], dtype=object)
    return xr.DataArray(
        data=data,
        coords=dict(rep=[0, 1], img=[0, 1]),
        dims=['img', 'rep'],
        attrs=dict(unit='msecond')
    )


def test_concatenate_spike_trains(spike_records: xr.DataArray):
    # All
    spike_trains = ops.concatenate_spike_trains(spike_records)
    assert np.array_equal(spike_trains[0], [15., 17., 32., 54., 73.])
    assert np.array_equal(spike_trains[1], [74.])
    assert np.array_equal(spike_trains[2], [18., 39., 56., 57.])
    # Select
    spike_trains = ops.concatenate_spike_trains(spike_records, [0])
    assert np.array_equal(spike_trains[0], [15., 17., 32., 54., 73.])
    # Select (subset)
    spike_trains = ops.concatenate_spike_trains(spike_records, [0], 10, 10)
    assert np.array_equal(spike_trains[0], [5., 7., 12., 24., 33.])
    # Select (subset, separation)
    spike_trains = ops.concatenate_spike_trains(spike_records, [0], 10, 10, 50)
    assert np.array_equal(spike_trains[0], [5., 7., 62., 124., 183.])
