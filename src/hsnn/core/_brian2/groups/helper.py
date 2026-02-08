from enum import Enum
from typing import Any, Dict, Optional, Sequence

import numpy as np
from brian2 import Group

from ..symbols import process_symbols


def get_spatial_coords(shape: Sequence, num_channels: Optional[int] = None,
                       spatial_span: float = 128) -> Sequence[np.ndarray]:
    assert 0 < len(shape) < 3, "invalid shape"
    shape = tuple(shape)
    if len(shape) < 2:
        shape = (1,) + shape
    spatial_dims = shape[::-1]

    coords = []
    for idx, grid in enumerate(np.meshgrid(*[range(size) for size in spatial_dims])):
        vec = grid.flatten() * spatial_span / spatial_dims[idx]
        if num_channels is not None:
            vec = np.tile(vec, num_channels)
        coords.append(vec)
    if num_channels is None:
        return coords
    else:
        coords.append(np.repeat(range(num_channels), np.prod(shape)))
        return coords


def process_namespaces(namespaces: Dict[str, Dict[Enum, dict]]) -> Dict[str, Dict[Enum, dict]]:
    dst = {}
    for grp_key, symbol_tables in namespaces.items():
        dst[grp_key] = {grp_cls: process_symbols(symbol_table)
                        for grp_cls, symbol_table in symbol_tables.items()}
    return dst


def process_kwargs(defaults: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    defaults.update(**process_symbols(kwargs))
    return defaults


def set_group_attr(group: Group, **kwargs) -> None:
    for key, val in kwargs.items():
        if not hasattr(group, key):
            group.add_attribute(key)
        setattr(group, key, val)
