from typing import Any, Mapping, Sequence

import numpy as np

import hsnn.simulation.functional as F
from hsnn import analysis
from hsnn.core import INetwork
from hsnn.analysis._types import RatesDatabase
from .base import get_nested


def get_rates_db(network: INetwork, data: Sequence[np.ndarray], duration: float,
                 duration_relax: float, offset: float, reps: int,
                 conditional_fired: bool) -> RatesDatabase:
    records = F.infer_batch_reps(
        network, data, duration, duration_relax, reps=reps
    )
    return analysis.create_rates_db(
        records.sel(layer=slice(1, None), nrn_cls='EXC'),
        duration=duration-offset, offset=offset,
        conditional_fired=conditional_fired
    )


def parse_scale_factor(expr: str, cfg: Mapping) -> Any:
    """Parses a string expression to obtain a scaled config parameter.

    Args:
        expr (str): Expression of form "`number` * `key_path`".
        cfg (Mapping): Config containing `key_path`.

    Returns:
        Any: Scaled config parameter: `number` * `key_path`.
    """
    substrings = [substring.strip() for substring in expr.split('*')]
    if len(substrings) != 2:
        raise ValueError(f"Input '{expr}' must contain exactly one '*'")

    factor, src_path = substrings
    number: Any
    if factor.isdigit():
        number = int(factor)
    else:
        try:
            number = float(factor)
        except ValueError:
            raise ValueError(f"'{factor}' is not a valid number")
    return number * get_nested(src_path, cfg)
