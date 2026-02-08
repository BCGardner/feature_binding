from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import xarray as xr

__all__ = [
    "RatesArray",
    "CountsArray",
    "OccurrencesArray",
    "RatesDatabase",
    "DeltaTuple"
]


RatesArray: TypeAlias = xr.DataArray
"Rates array, with dims (`rep`, `img`, `pk`)."
CountsArray: TypeAlias = xr.DataArray
"Counts array, with dims (`rep`, `img`, `pk`)."
OccurrencesArray: TypeAlias = xr.DataArray
"Fraction of reps with an occurrence array, with dims (`img`, `pk`)."
RatesDatabase: TypeAlias = xr.Dataset
"Rates dataset, containing arrays: (`rates`, `stdevs`, `occurs`)."
DeltaTuple: TypeAlias = tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]
