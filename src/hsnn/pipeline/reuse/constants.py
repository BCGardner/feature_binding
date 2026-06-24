"""Constants for the reuse annotation-table pipeline."""

from __future__ import annotations

__all__ = [
    "TABLE_NAMES",
    "REUSE_SUBDIR",
    "DEFAULT_DURATION",
    "DEFAULT_OFFSET",
]

TABLE_NAMES = ("hfb_annotations", "png_label_metrics", "neuron_information")
REUSE_SUBDIR = "reuse"
DEFAULT_DURATION = 200.0
DEFAULT_OFFSET = 50.0
