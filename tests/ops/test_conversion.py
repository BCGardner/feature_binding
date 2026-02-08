import pytest
from typing import Sequence

import numpy as np
import xarray as xr

from hsnn import ops
from ._utils import get_data


@pytest.mark.parametrize("test_input, expected", get_data('as_spike_events'))
def test_as_spike_events(test_input: dict, expected: Sequence):
    spike_events = ops.as_spike_events(test_input)
    assert np.array_equal(spike_events[0], expected[0])
    assert np.array_equal(spike_events[1], expected[1])


@pytest.mark.parametrize("test_input, expected", get_data('as_spike_trains'))
def test_as_spike_trains(test_input: Sequence, expected: dict):
    spike_trains = ops.as_spike_trains(tuple(test_input))
    assert np.array_equal(list(spike_trains.keys()), list(expected.keys()))
    for nrn in expected.keys():
        assert np.array_equal(spike_trains[nrn], expected[nrn])


@pytest.mark.parametrize("test_input, expected", get_data('get_rates'))
def test_get_rates(test_input: Sequence, expected: dict):
    spike_events, duration, num_nrns = test_input
    rates = ops.get_rates(tuple(spike_events), duration, num_nrns)
    assert np.array_equal(rates[0], expected[0])
    assert np.array_equal(rates[1], expected[1])
