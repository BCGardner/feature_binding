from pathlib import Path
from typing import Optional

from .. import io
from .trial import TrialView

__all__ = [
    'get_results_path',
    'get_hfb_path',
    'get_artifact_path'
]


def get_results_path(trial: TrialView, chkpt: Optional[int] = None, *,
                     subdir: Optional[str] = None, ext: str = '.pkl.gz', **kwargs) -> Path:
    return _get_artifact_path(
        'inference', trial, chkpt, subdir=subdir, ext=ext, **kwargs
    )


def get_hfb_path(trial: TrialView, chkpt: Optional[int] = None, *,
                 subdir: Optional[str] = None, sgnf: bool = False, **kwargs) -> Path:
    return _get_artifact_path(
        'hfb', trial, chkpt, subdir=subdir, ext='.db', sgnf=sgnf, **kwargs
    )


def get_artifact_path(base_name: str, trial: TrialView, chkpt: Optional[int] = None, *,
                      subdir: Optional[str] = None, ext: str = '.pkl', **kwargs) -> Path:
    return _get_artifact_path(
        base_name, trial, chkpt, subdir=subdir, ext=ext, **kwargs
    )


def _get_artifact_path(base_name: str, trial: TrialView, chkpt: Optional[int] = None, *,
                       subdir: Optional[str] = None, ext: Optional[str] = None, **kwargs) -> Path:
    fname = _get_artifact_name(base_name, chkpt, ext, **kwargs)
    relative_path = _get_relative_path(fname, subdir)
    if chkpt is not None:
        artifact_path = trial.checkpoints[chkpt].path / relative_path
    else:
        artifact_path = trial.path / relative_path
    return artifact_path.resolve()


def _get_artifact_name(base_name: str, chkpt: Optional[int] = None,
                       ext: Optional[str] = None, **kwargs) -> str:
        name = base_name if chkpt is not None else f'{base_name}_init'
        return io.formatted_name(name, ext, **kwargs)


def _get_relative_path(fname: str, subdir: Optional[str] = None) -> Path:
    relative_path = Path(fname) if subdir is None else Path(subdir) / fname
    conditions = [
        not relative_path.is_absolute(),
        not relative_path.parts[0].startswith('checkpoint'),
        len(relative_path.parts) <= 2
    ]
    if not all(conditions):
        raise ValueError(f"Invalid relative path: '{relative_path}'.")
    return relative_path
