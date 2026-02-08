from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np

from ._base import BaseLoader
from ..transforms import BaseTransform

__all__ = ['ImageSet']


class ImageLoader(BaseLoader):
    def __init__(self, flag: int = cv2.IMREAD_GRAYSCALE):
        self.flag: int = flag

    def load(self, file_path: str | Path) -> np.ndarray:
        if isinstance(file_path, Path):
            file_path = str(file_path.resolve())
        return cv2.imread(file_path, self.flag)


class ImageSet(Sequence):
    def __init__(self, image_dir: str | Path,
                 image_loader: BaseLoader = ImageLoader(),
                 transform: Optional[BaseTransform] = None,
                 pattern: str = '*.png'):
        self.image_paths = self._get_image_paths(image_dir, pattern)
        self.transform = transform
        self._image_loader = image_loader

    def __getitem__(self, index):  # type: ignore
        image_path = self.image_paths[index]
        image = self._image_loader.load(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.image_paths)

    def _get_image_paths(self, image_dir: str | Path, pattern: str = '*.png') -> List[str]:
        return [str(image_path.resolve()) for image_path in sorted(Path(image_dir).glob(pattern))]
