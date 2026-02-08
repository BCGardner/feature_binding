from copy import copy
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Type

import numpy as np
from ray import air
from ray import tune
from ray.tune.experiment.trial import Trial
from ray.tune import ResultGrid

from ._base import _overwrite_dir_prompt
from ._searchers import SearchAlg, DefaultSearcher, BayesOpt
from .tuning import override_config, TrainSNN
from ..utils import io

__all__ = ["ExperimentRunner"]

_CHECKPOINT_CFG: dict[str, Any] = {
    'num_to_keep': None,
    'checkpoint_frequency': 0,
    'checkpoint_at_end': True
}

_SEARCH_ALG_MAPPING: dict[Optional[str], Type[SearchAlg]] = {
    None: DefaultSearcher,
    'bayesopt': BayesOpt
}


def _trial_dirname_creator(trial: Trial) -> str:
    return f"{trial.trainable_name}_{trial.trial_id}"


class ExperimentRunner:
    def __init__(self, config: dict, results_dir: Optional[str | Path] = None) -> None:
        tuning_cfg: dict = config['tuning']
        training_cfg: dict = config['training']
        self._checkpoint_kwargs = training_cfg.pop('checkpointing', {})
        self._hyper_params = tuning_cfg.get('hyper_params', {})
        self.expt_name = tuning_cfg.get('expt_name', None)
        self.results_dir = io.EXPT_DIR if results_dir is None else results_dir
        self.config = override_config(config, self._hyper_params)
        self._prepare_data(training_cfg['data'])

    def run(self, trainable: Type[tune.Trainable] = TrainSNN) -> ResultGrid:
        tune_config = self._get_tune_config()
        run_config = self._get_run_config()
        self._setup_expt_dir()
        tuner = tune.Tuner(
            tune.with_parameters(trainable, data=self.data, labels=self.labels),
            tune_config=tune_config,
            run_config=run_config,
            param_space=self.config
        )
        results = tuner.fit()
        io.save_pickle(results.get_dataframe(), Path(results._local_path) / 'results.pkl')
        return results

    def _prepare_data(self, data_cfg: Mapping):
        data, labels = io.get_dataset(data_cfg, return_annotations=True)
        self.data: Sequence[np.ndarray] = list(data)  # Apply transforms once
        self.labels: np.ndarray = labels.iloc[:, 1:].values

    def _get_tune_config(self) -> tune.TuneConfig:
        tuning_cfg: dict = self.config['tuning']
        search_alg = self._create_search_alg()
        return tune.TuneConfig(
            metric='loss', mode='min', search_alg=search_alg,
            num_samples=tuning_cfg.get('num_samples', 1),
            trial_dirname_creator=_trial_dirname_creator
        )

    def _get_run_config(self) -> air.RunConfig:
        training_cfg: dict = self.config['training']
        tuning_cfg: dict = self.config['tuning']
        stop = {'training_iteration': training_cfg.get('training_iteration', 1)}
        if 'stop' in tuning_cfg:
            stop['loss'] = tuning_cfg['stop']['threshold']
        checkpoint_config = self._get_checkpoint_config()
        parameter_columns = [key.replace('.', '/') for key in self._hyper_params.keys()]
        return air.RunConfig(
            self.expt_name, str(self.results_dir),
            stop=stop,
            checkpoint_config=checkpoint_config,
            progress_reporter=tune.CLIReporter(parameter_columns=parameter_columns),
            verbose=1
        )

    def _get_checkpoint_config(self) -> air.CheckpointConfig:
        checkpoint_cfg = copy(_CHECKPOINT_CFG)
        checkpoint_cfg.update(**self._checkpoint_kwargs)
        return air.CheckpointConfig(**checkpoint_cfg)

    def _setup_expt_dir(self) -> None:
        if self.expt_name is not None:
            expt_dir: Path = Path(self.results_dir) / self.expt_name
            if expt_dir.is_dir():
                if not _overwrite_dir_prompt(expt_dir):
                    raise RuntimeError("Cancel experiment")

    def _create_search_alg(self) -> Any:
        tuning_cfg: dict = self.config['tuning']
        search_cfg: dict | str = tuning_cfg.get('search_alg', {})
        if isinstance(search_cfg, Mapping):
            name = search_cfg.get('name', None)
            kwargs = search_cfg.get('kwargs', {})
        else:
            name = search_cfg
            kwargs = {}
        factory = _SEARCH_ALG_MAPPING[name](self.config)
        return factory.create(**kwargs)
