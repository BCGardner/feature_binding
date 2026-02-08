import numpy as np
from brian2.units import umetre

from hsnn.core._brian2.groups import helper


def test_get_spatial_coords():
    # 2D coordinates
    shape = (2, 2)
    spatial_span = 100 * umetre
    xs, ys = helper.get_spatial_coords(shape, spatial_span=spatial_span)
    assert np.array_equal(xs, [0, 50, 0, 50] * umetre)
    assert np.array_equal(ys, [0, 0, 50, 50] * umetre)
    # 3D coordinates
    xs, ys, zs = helper.get_spatial_coords(shape, num_channels=2, spatial_span=spatial_span)
    assert np.array_equal(xs, [0, 50, 0, 50, 0, 50, 0, 50] * umetre)
    assert np.array_equal(ys, [0, 0, 50, 50, 0, 0, 50, 50] * umetre)
    assert np.array_equal(zs, [0, 0, 0, 0, 1, 1, 1, 1])
