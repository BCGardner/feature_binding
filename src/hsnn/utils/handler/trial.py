from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


from .. import io
from ._checkpoint import CheckpointView

__all__ = ['TrialView']


@dataclass
class TrialView:
    path: Path = field(repr=False)
    name: str = field(init=False)

    def __post_init__(self):
        self.path = self.path.resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"'{self.path}' does not exist.")
        self.name = self.path.name
        self._checkpoints = [CheckpointView(path) for path in
                             sorted(Path.glob(self.path, 'checkpoint*'))]

    @property
    def config(self) -> dict[str, Any]:
        return io.as_dict(self.path / 'config.yaml')

    @property
    def checkpoints(self) -> list[CheckpointView]:
        return self._checkpoints

    def get_checkpoint_step(self, step: int) -> CheckpointView:
        for checkpoint in self._checkpoints:
            if step == int(checkpoint.name.split('_')[-1]):
                return checkpoint
        else:
            raise KeyError(f"No checkpoint with '{step}' steps.")
