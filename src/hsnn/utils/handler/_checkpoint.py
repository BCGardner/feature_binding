from dataclasses import dataclass, field
from pathlib import Path

__all__ = ['CheckpointView']


@dataclass
class CheckpointView:
    path: Path = field(repr=False)
    name: str = field(init=False)

    def __post_init__(self):
        self.path = self.path.resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"'{self.path}' does not exist.")
        self.name = self.path.name

    @property
    def store_path(self) -> Path:
        ret = self.path / 'netstate.pkl'
        if not ret.exists():
            raise FileNotFoundError(f"'{ret}' does not exist.")
        return ret
