import numpy as np
import numpy.typing as npt

__all__ = [
    "multiplicative_jitter"
]


def multiplicative_jitter(
    values: npt.ArrayLike, sigma: float, dt: float = 0.1,
    seed: np.random.Generator | int | None = None
) -> npt.NDArray[np.float64]:
    """Applies multiplicative jitter according to a provided array.

    Args:
        array (npt.ArrayLike): Array of float values.
        sigma (float): Standard deviation of the jitter (fractional amplitude).
        dt (float, optional): Time resolution. Defaults to 0.1.
        seed (np.random.Generator | int | None, optional): Seed. Defaults to None.

    Returns:
        npt.NDArray[np.float64]: Multiplicative jittered array.
    """
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    jitter_factor = rng.normal(loc=1.0, scale=sigma, size=len(values))
    new_delays = values * jitter_factor
    new_delays = np.maximum(new_delays, dt)
    return dt * np.round(new_delays / dt)
