import pytest
from typing import Sequence

import numpy as np
import xarray as xr

from hsnn import ops
from ._utils import get_data


@pytest.mark.parametrize("test_input, expected", get_data('get_submask'))
def test_get_submask(test_input: Sequence, expected: Sequence):
    mask = ops.get_submask(*test_input)
    assert np.array_equal(mask, expected)


@pytest.mark.parametrize("test_input, expected", get_data('mask_recording'))
def test_mask_recording(test_input: Sequence, expected: Sequence):
    recording = ops.mask_recording(*test_input)
    for arr1, arr2 in zip(recording, expected):
        assert np.array_equal(arr1, arr2)


@pytest.mark.parametrize("test_input, expected", get_data('mask_rates'))
def test_mask_rates(test_input: Sequence, expected: Sequence):
    recording = ops.mask_rates(*test_input)
    for arr1, arr2 in zip(recording, expected):
        assert np.array_equal(arr1, arr2)
