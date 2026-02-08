from typing import Optional

import numpy.typing as npt
import numpy as np
from scipy import stats


def uniform(points: npt.NDArray[np.float_], p_conn: float = 1,
            num_conn: Optional[int] = None) -> npt.NDArray[np.int_]:
    if num_conn is None:
        return np.flatnonzero(np.random.rand(len(points)) < p_conn)
    else:
        return np.random.choice(len(points), size=num_conn, replace=False)


def gaussian(points: npt.NDArray[np.float_], point_ref: npt.NDArray[np.float_],
             stdev: float, p_conn: float = 1, num_conn: Optional[int] = None) -> npt.NDArray[np.int_]:
    distances = np.linalg.norm(points - point_ref, axis=1)
    if num_conn is None:
        probs = p_conn * np.exp(-0.5 * (distances / stdev)**2)
        return np.flatnonzero(np.random.rand(len(points)) < probs)
    else:
        probs = stats.norm.pdf(distances, scale=stdev)
        probs /= probs.sum()
        return np.random.choice(len(distances), size=num_conn, replace=False, p=probs)
