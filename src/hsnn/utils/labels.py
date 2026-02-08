import numpy as np
import numpy.typing as npt
import pandas as pd

__all__ = [
    "labels_to_masks",
    "get_target_ids",
    "get_unique_id"
]


def labels_to_masks(labels: pd.DataFrame) -> dict[tuple[str, np.int64], npt.NDArray[np.bool_]]:
    if 'image_id' in labels:
        labels = labels.drop('image_id', axis=1)
    masks = {}
    for col in labels.columns[::-1]:
        targets = np.array(labels[col])
        unique_targets = np.unique(targets)
        for target in unique_targets:
            masks[(col, target)] = np.asarray(targets == target)
    return masks


def get_target_ids(labels: pd.DataFrame, attribute: str, target: int = 1) -> npt.NDArray[np.intp]:
    return np.flatnonzero(labels[attribute] == target)


def get_unique_id(labels: pd.DataFrame, attribute: str, target: int = 1) -> int:
    mask = (labels[attribute] == target) & (labels.drop('image_id', axis=1).sum(axis=1) == 1)
    return mask.argmax().item()
