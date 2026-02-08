from typing import Optional, Sequence

import numpy as np
import pandas as pd

from . import _utils
from .base import PNG, Refine

__all__ = [
    "Compose",
    "DropRepeating",
    "Merge",
    "FilterBySpan",
    "FilterLayers",
    "FilterIndex",
    "Match",
    "FilterMinOcc",
    "Structural",
    "Connected",
    "Constrained",
    "PassThrough",
]


class Compose(Refine):
    def __init__(self, refines: Sequence[Refine]) -> None:
        self._refines = refines

    def __call__(self, pngs: Sequence[PNG]) -> Sequence[PNG]:
        for refine in self._refines:
            pngs = refine(pngs)
        return pngs

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        format_string += repr(self._refines) + ")"
        return format_string


class DropRepeating(Refine):
    """Drops PNGs containing neurons which individually contribute more than one spike."""

    def __call__(self, pngs: Sequence[PNG]) -> Sequence[PNG]:
        return _utils.drop_repeating(pngs)


class Merge(Refine):
    """Merges together PNGs according to their lags within a given tolerance (in ms)."""

    def __init__(
        self, tol: float = 3.0, strategy: str = "mode", overlaps: bool = False
    ) -> None:
        self.tol = tol
        self.strategy = strategy
        self.overlaps = overlaps

    def __call__(self, pngs: Sequence[PNG]) -> Sequence[PNG]:
        return _utils.merge_pngs(pngs, self.tol, self.strategy, self.overlaps)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(tol={self.tol})"


class FilterBySpan(Refine):
    """Filter PNGs by their empirical temporal span (max_lag - min_lag).

    Args:
        max_span: Maximum allowed temporal span in ms (inclusive).
    """
    def __init__(self, max_span: float) -> None:
        self.max_span = max_span

    def __call__(self, pngs: Sequence[PNG]) -> Sequence[PNG]:
        return [png for png in pngs if self._get_span(png) <= self.max_span]

    @staticmethod
    def _get_span(png: PNG) -> float:
        """Compute temporal span from lags."""
        return float(np.max(png.lags) - np.min(png.lags))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(max_span={self.max_span})"


class FilterLayers(Refine):
    """Filters to PNGs containing neurons matching provided layer IDs."""

    def __init__(self, layers: Sequence[int]) -> None:
        self.layers = layers

    def __call__(self, pngs: Sequence[PNG]) -> Sequence[PNG]:
        return [
            png
            for png in pngs
            if len(png.layers) == len(self.layers) and np.all(png.layers == self.layers)
        ]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(layers={self.layers})"


class FilterIndex(Refine):
    """Filters to PNGs containing the select index `(layer, nrn)`, optionally
    also positional.
    """

    def __init__(self, index: tuple[int, int], position: Optional[int] = None) -> None:
        self.index = index
        self.position = position

    def __call__(self, pngs: Sequence[PNG]) -> Sequence[PNG]:
        return [png for png in pngs if self._isin(png)]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(index={self.index}, position={self.position})"
        )

    def _isin(self, png: PNG) -> bool:
        if self.position is None:
            return self.index in list(zip(png.layers, png.nrns))
        cmp = (png.layers[self.position], png.nrns[self.position])
        return self.index == cmp


class Match(Refine):
    """Gets PNGs matching the ordered indices `[(layer_1, nrn_1), ...]`."""

    def __init__(self, indices: Sequence[tuple[int, int]]) -> None:
        self.indices = indices

    def __call__(self, pngs: Sequence[PNG]) -> Sequence[PNG]:
        return [png for png in pngs if self._match(png)]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(indices={self.indices}"

    def _match(self, png: PNG) -> bool:
        if len(png.layers) != len(self.indices):
            return False
        png_indices = list(zip(png.layers, png.nrns))
        for idx, idx_ref in zip(png_indices, self.indices):
            if idx != idx_ref:
                return False
        return True


class FilterMinOcc(Refine):
    """Filters to PNGs which reoccur at least a minimum number of times."""

    def __init__(self, min_occ: int) -> None:
        self.min_occ = min_occ

    def __call__(self, pngs: Sequence[PNG]) -> Sequence[PNG]:
        return [png for png in pngs if len(png.times) >= self.min_occ]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(min_occ={self.min_occ})"


class Structural(Refine):
    """Filters to structural PNGs, checked against provided projections."""

    def __init__(
        self,
        syn_params: pd.DataFrame,
        projs: Sequence[str] = ("FF", "E2E"),
        tol: float = 3.0,
    ) -> None:
        self.projs = projs
        self.tol = tol
        self._syn_params = syn_params

    def __call__(self, pngs: Sequence[PNG]) -> Sequence[PNG]:
        return [
            png
            for png in pngs
            if _utils.isstructural(png, self._syn_params, self.projs, self.tol)
        ]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(projs={self.projs}, tol={self.tol})"


class Connected(Refine):
    """Filters to PNGs where all neuron pairs have synaptic connections.

    Only verifies connection existence, not weight or delay constraints.
    For HFB triplets with structure [L-1, L, L], checks:
    - Low -> High (feedforward)
    - Low -> Bind (feedforward)
    - High -> Bind (lateral E2E)
    """

    def __init__(self, syn_params: pd.DataFrame) -> None:
        assert syn_params.index.names == ["layer", "proj", "pre", "post"]
        self._syn_params = syn_params

    def __call__(self, pngs: Sequence[PNG]) -> Sequence[PNG]:
        return [png for png in pngs if _utils.isconnected(png, self._syn_params)]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Constrained(Refine):
    """Filters to synaptically constrained PNGs, checked against minimum weight and delay tolerance."""

    def __init__(
        self, syn_params: pd.DataFrame, w_min: float = 0.5, tol: float = 3.0
    ) -> None:
        assert syn_params.index.names == ["layer", "proj", "pre", "post"]
        assert {"w", "delay"}.issubset(syn_params.columns)
        self.w_min = w_min
        self.tol = tol
        self._syn_params = syn_params

    def __call__(self, pngs: Sequence[PNG]) -> Sequence[PNG]:
        return [
            png
            for png in pngs
            if _utils.isconstrained(png, self._syn_params, self.w_min, self.tol)
        ]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(w_min={self.w_min}, tol={self.tol})"


class PassThrough(Refine):
    def __call__(self, pngs: Sequence[PNG]) -> Sequence[PNG]:
        return [png for png in pngs]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
