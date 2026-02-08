from typing import Mapping
from omegaconf import OmegaConf
from pathlib import Path

import xarray as xr

from .trial import TrialView, CheckpointView
from . import artifacts
from .. import io

__all__ = ["ArtifactStore"]

_ARTIFACT_EXT_MAPPING = {
    'results': '.pkl.gz',
    'config': '.yaml'
}


class ArtifactStore:
    def __init__(self, trial: TrialView, ckpt_idx: int | None = None) -> None:
        self._trial = trial
        self.set_checkpoint(ckpt_idx)

    def __repr__(self) -> str:
        return f"ArtifactStore(trial={self._trial.name}, ckpt_idx={self._ckpt_idx})"

    @property
    def logdir(self) -> Path:
        if self._ckpt is None:
            return self._trial.path
        return self._ckpt.path

    @property
    def trial(self) -> TrialView:
        return self._trial

    @property
    def checkpoint(self) -> CheckpointView | None:
        return self._ckpt

    def set_checkpoint(self, ckpt_idx: int | None = None) -> None:
        self._ckpt_idx = ckpt_idx
        self._ckpt = None if ckpt_idx is None else self._trial.checkpoints[ckpt_idx]

    def list_artifacts(self, subdir: str | None = None) -> list[str]:
        logdir = self._resolve_dir(subdir)
        if not logdir.exists():
            raise FileNotFoundError(f"Directory does not exist: {logdir}")
        return sorted([f.name for f in logdir.iterdir() if f.is_file()])

    def list_subdirs(self, subdir: str | None = None) -> list[Path]:
        logdir = self._resolve_dir(subdir)
        if not logdir.exists():
            raise FileNotFoundError(f"Directory does not exist: {logdir}")
        return sorted([f for f in logdir.iterdir() if f.is_dir()])

    def remove_artifact(self, fname: str, subdir: str | None = None) -> None:
        fpath = self._resolve_dir(subdir) / fname
        if not fpath.exists():
            raise FileNotFoundError(f"File does not exist: {fpath}")
        fpath.unlink()
        if not fpath.exists():
            print(f"Removed artifact: {fpath}")
        else:
            raise OSError(f"Failed to remove file: {fpath}")

    def load_results(self, subdir: str | None = None, **kwargs) -> xr.DataArray:
        ext = _ARTIFACT_EXT_MAPPING['results']
        fpath = artifacts.get_results_path(
            self._trial, self._ckpt_idx, subdir=subdir, ext=ext, **kwargs
        )
        return io.load_pickle(fpath)

    def save_results(self, data: xr.DataArray, subdir: str | None = None,
                     overwrite: bool = False, **kwargs) -> None:
        ext = _ARTIFACT_EXT_MAPPING['results']
        fpath = artifacts.get_results_path(
            self._trial, self._ckpt_idx, subdir=subdir, ext=ext, **kwargs
        )
        if not overwrite and fpath.exists():
            raise FileExistsError(f"File already exists: {fpath}")
        return io.save_pickle(data, fpath, parents=True)

    def load_config(self, subdir: str | None = None) -> dict:
        ext = _ARTIFACT_EXT_MAPPING['config']
        fpath = self._resolve_dir(subdir) / f'config{ext}'
        self._check_path(fpath)
        return io.as_dict(fpath)

    def save_config(self, config: Mapping, subdir: str | None = None,
                    overwrite: bool = False) -> None:
        ext = _ARTIFACT_EXT_MAPPING['config']
        fpath = self._make_logdir(subdir) / f'config{ext}'
        if not overwrite and fpath.exists():
            raise FileExistsError(f"File already exists: {fpath}")
        OmegaConf.save(config, fpath)

    def _resolve_dir(self, subdir: str | None) -> Path:
        if subdir is None:
            return self.logdir
        return Path.resolve(self.logdir / subdir)

    def _make_logdir(self, subdir: str | None) -> Path:
        logdir = self._resolve_dir(subdir)
        if subdir is not None:
            logdir.mkdir(exist_ok=True)
        return logdir

    def _check_path(self, fpath: Path) -> None:
        if not fpath.exists():
            raise FileNotFoundError(f"File not found: {fpath}")
