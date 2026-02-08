from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseLoader(ABC):
    @abstractmethod
    def load(self, file_path: str | Path) -> np.ndarray:
        raise NotImplementedError()
