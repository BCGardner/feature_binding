import json
import re
from dataclasses import dataclass, field, InitVar
from pathlib import Path
from typing import Optional

import pandas as pd
from tbparse import SummaryReader

from .. import io
from .trial import TrialView

__all__ = ['ExperimentHandler', 'get_closest_samples']

_PATTERNS = [
    r'projections/([^/]+)/namespace/eta',
    r'ray/tune/([^/]+)'
]


class MetaData:
    def __init__(self, logdir: Path):
        self._path = logdir / 'metadata.json'
        self._analysis = {}
        self.load()

    def __repr__(self) -> str:
        model_types = list(self._analysis.keys())
        return f"{__class__.__name__}(models={model_types})"

    @property
    def path(self) -> Path:
        return self._path

    def get_trials_dict(self, model_type: str) -> dict[str, list[str]]:
        return self._analysis.get(model_type, {})

    def set_trials_dict(self, model_type: str, key: str, trial_names: list[str]):
        trials_dict = self.get_trials_dict(model_type)
        trials_dict[key] = trial_names
        self._analysis[model_type] = trials_dict
        self.save()

    def remove(self, model_type: str, key: Optional[str] = None):
        if key is not None:
            del self._analysis[model_type][key]
        else:
            del self._analysis[model_type]
        self.save()

    def load(self):
        if self._path.exists():
            with open(self._path, 'r') as fp:
                self._analysis = json.load(fp)

    def save(self):
        with open(self.path, 'w') as fp:
            json.dump(self._analysis, fp, indent=4)


@dataclass
class ExperimentHandler:
    logdir: Path = field(repr=False)
    basedir: InitVar[Path] = field(default=io.EXPT_DIR)
    concise_names: bool = field(repr=False, default=True)
    name: str = field(init=False)
    num_trials: int = field(init=False)
    metadata: MetaData = field(init=False, repr=False)

    def __post_init__(self, basedir):
        self.logdir = Path(basedir / self.logdir).resolve()
        if not self.logdir.exists():
            raise FileNotFoundError(f"'{self.logdir}' does not exist.")
        self.name = self.logdir.name
        self._reader = SummaryReader(str(self.logdir), pivot=True,
                                     extra_columns={'dir_name'})
        trialdirs = sorted([path for path in self.logdir.glob('*') if path.is_dir()])
        self._trials = {trialdir.name: TrialView(trialdir) for trialdir in trialdirs}
        self.num_trials = len(self)
        self.metadata = MetaData(self.logdir)

    def __getitem__(self, key: int | str | tuple) -> TrialView:
        if isinstance(key, int):
            return list(self._trials.values())[key]
        elif isinstance(key, str):
            return self._trials[key]
        elif isinstance(key, tuple):
            idx2dir = self.index_to_dir
            return self[idx2dir.loc[key]]
        else:
            raise KeyError(f"'{key}'")

    def __len__(self) -> int:
        return len(self._trials)

    @property
    def scalars(self) -> pd.DataFrame:
        if not hasattr(self, '_scalars'):
            self._scalars = self._reader.scalars
            self._scalars.columns = [_concise_string(col) for col in self._scalars.columns]
        return self._scalars.copy()

    @property
    def hparams(self) -> pd.DataFrame:
        if not hasattr(self, '_hparams'):
            self._hparams = self._reader.hparams
            if self.concise_names:
                self._hparams.columns = [_concise_string(col) for col in self._hparams.columns]
        return self._hparams.copy()

    @property
    def index_to_dir(self) -> pd.Series:
        df = self._merge_tables(drop_dir=False).droplevel('step')
        df = df.reset_index().set_index(df.index.names + ['sample'])['dir_name']
        return df.drop_duplicates().sort_index()

    @property
    def dir_to_index(self) -> pd.DataFrame:
        return self.index_to_dir.reset_index().set_index('dir_name')

    @property
    def trial_names(self) -> list[str]:
        return list(self._trials.keys())

    def get_timeseries(self, scalar: str = 'loss', max_info: Optional[float] = None,
                       reduce: bool = False) -> pd.DataFrame:
        df_ts = self._merge_tables(scalar=scalar, drop_dir=True)
        index_cols = df_ts.index.names
        df_ts = df_ts.pivot_table(scalar, index=index_cols, columns='sample')
        if scalar == 'loss' and max_info is not None:
            df_ts = max_info - df_ts
        if reduce:
            return df_ts.agg(['mean', 'std'], axis=1)
        return df_ts

    def get_summary(self, iter_idx: int, scalar: str = 'loss',
                    max_info: Optional[float] = None, reduce: bool = False) -> pd.DataFrame | pd.Series:
        df = self.get_timeseries(scalar, max_info, reduce)
        step = df.index.unique('step')[iter_idx]
        return df.xs(step, level='step')

    def _merge_tables(self, scalar: str = 'loss', drop_dir: bool = True) -> pd.DataFrame:
        hparams = self.hparams
        scalars = self.scalars
        index_cols = hparams.drop('dir_name', axis=1).columns.to_list() + ['step']
        if drop_dir:
            df = hparams.merge(scalars[['step', scalar, 'dir_name']],
                               on='dir_name').drop('dir_name', axis=1)
        else:
            df = hparams.merge(scalars[['step', scalar, 'dir_name']], on='dir_name')
        df = df.set_index(index_cols)
        df['sample'] = df.groupby(level=list(range(len(df.index.levels)))).cumcount() # type: ignore
        return df


def get_closest_samples(df: pd.DataFrame) -> pd.Index:
    closest_samples = df.sub(df.mean(axis=1), axis=0).abs().idxmin(axis=1)
    closest_samples.name = 'sample'
    closest_samples = closest_samples.reset_index()
    return closest_samples.set_index(list(closest_samples.columns)).index


def _concise_string(s: str) -> str:
    for pattern in _PATTERNS:
        match = re.search(pattern, s)
        if match:
            return match.group(1)
    return s
