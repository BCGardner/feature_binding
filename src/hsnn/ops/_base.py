from typing import Sequence

from hsnn.core.types import SpikeEvents, SpikeTrains


def assert_recording(recording: SpikeEvents | SpikeTrains) -> None:
    if isinstance(recording, dict):
        return
    if isinstance(recording, Sequence) and len(recording) == 2:
        return
    raise TypeError(f"invalid argument '{recording}'")
