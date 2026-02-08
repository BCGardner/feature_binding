import numpy as np
import numpy.typing as npt
from typing import Optional

from hsnn.analysis._types import RatesArray


def update_dict(dst: dict, src: Optional[dict] = None) -> dict:
    src = {} if src is None else src
    dst.update(src)
    return dst


def get_bar_data(rates: RatesArray, nrn_id: int, masks: dict) -> npt.NDArray[np.float_]:
    combined_rates = []
    # r_stdevs = []
    for mask in masks.values():
        img_ids = np.flatnonzero(mask)
        rates_ = rates.sel(img=img_ids, nrn=nrn_id).values
        combined_rates.append(np.mean(rates_))
        # r_stdevs.append(np.std(rates_))
    return np.array(combined_rates)


def get_violin_data(rates: RatesArray, nrn_id: int, masks: dict) -> npt.NDArray[np.float_]:
    rates_combined = []
    for mask in masks.values():
        img_ids = np.flatnonzero(mask)
        rates_combined.append(
            np.array(rates.sel(img=img_ids, nrn=nrn_id)).flatten())
    return np.stack(rates_combined, axis=1)
