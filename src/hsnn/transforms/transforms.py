from typing import Optional, Sequence

import cv2
import numpy as np

from ._base import BaseTransform

__all__ = ["ClipValues", "Compose", "Resize", "Threshold", "ThresholdToMax",
           "SquareCrop", "CentreCrop", "GaussianNoise"]


class ClipValues(BaseTransform):
    def __init__(self, v_min: Optional[float] = None,
                 v_max: Optional[float] = None) -> None:
        self.v_min = v_min
        self.v_max = v_max

    def transform(self, image: np.ndarray) -> np.ndarray:
        return np.clip(image, self.v_min, self.v_max)


class Compose(BaseTransform):
    def __init__(self, *transforms: BaseTransform) -> None:
        self._transforms = transforms

    def transform(self, image: np.ndarray) -> np.ndarray:
        for transform in self._transforms:
            image = transform(image)
        return image


class Resize(BaseTransform):
    def __init__(self, resize_dims: Sequence[int], inter: int = cv2.INTER_AREA) -> None:
        self.resize_dims = resize_dims
        self.inter = inter

    def transform(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, self.resize_dims, interpolation=self.inter)


class Threshold(BaseTransform):
    def __init__(self, thresh: int, maxval: int = 255,
                 op_flag: int = cv2.THRESH_BINARY) -> None:
        self.thresh = thresh
        self.maxval = maxval
        self.op_flag = op_flag

    def transform(self, image: np.ndarray) -> np.ndarray:
        _, dst = cv2.threshold(image, self.thresh, self.maxval, self.op_flag)
        return dst


class ThresholdToMax(BaseTransform):
    def __init__(self, thresh: int, maxval: int = 255) -> None:
        self.thresh = thresh
        self.maxval = maxval

    def transform(self, image: np.ndarray) -> np.ndarray:
        image = image.copy()
        image[image >= self.thresh] = self.maxval
        return image


class SquareCrop(BaseTransform):
    def transform(self, image: np.ndarray) -> np.ndarray:
        w, h = image.shape[1], image.shape[0]

        # largest square crop of portrait or landscape frames
        w2 = h if h < w else w
        h2 = w2

        x = int(w/2 - w2/2)
        y = int(h/2 - h2/2)

        return image[y:y+h2, x:x+w2]


class CentreCrop(BaseTransform):
    def __init__(self, width: int, height: Optional[int] = None) -> None:
        self.width = width
        self.height = height if height is not None else width

    def transform(self, image: np.ndarray) -> np.ndarray:
        w, h = image.shape[1], image.shape[0]

        x = int(w/2 - self.width/2)
        y = int(h/2 - self.height/2)

        return image[y:y+self.height, x:x+self.width]


class GaussianNoise(BaseTransform):
    def __init__(self, amplitude: float) -> None:
        self.amplitude = amplitude

    def transform(self, image: np.ndarray) -> np.ndarray:
        if self.amplitude < 0:
            raise ValueError("Noise amplitude must be non-negative.")

        noise = np.random.normal(0, self.amplitude, image.shape)
        return np.clip(image + noise, 0, 255).astype(np.uint8)
