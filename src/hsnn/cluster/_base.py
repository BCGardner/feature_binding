import shutil
from pathlib import Path
from typing import Mapping, Optional

import pandas as pd
import ray

from ..core.backends import create_network
from ..utils import io
from .. import simulation as sim

__all__ = ["SimulatorActor", "setup_ray"]


@ray.remote
class SimulatorActor(sim.Simulator):
    def __init__(self, model_cfg: str | Path | Mapping, randomised_state: bool = True,
                 seed_val: Optional[int] = None, backend: str = 'brian2') -> None:
        cfg = io.as_dict(model_cfg)
        sim_kwargs = sim.get_sim_kwargs(cfg.get('simulation', {}))
        network = create_network(cfg, backend)
        super().__init__(network, **sim_kwargs)
        if randomised_state:
            self._network.seed_val = seed_val

    def restore(self, file_path: str | Path, name: str = 'default') -> None:
        self._network.restore(name, file_path)

    def get_duration(self) -> float:
        return self.duration

    def set_duration(self, value: float) -> None:
        self.duration = value

    def get_duration_relax(self) -> float:
        return self.duration_relax

    def set_duration_relax(self, value: float) -> None:
        self.duration_relax = value

    def get_syn_params(self, **kwargs) -> Optional[pd.DataFrame]:
        return self._network.get_syn_params(**kwargs)

    def restore_delays(self, file_path: Path) -> None:
        syn_params = pd.read_parquet(file_path)
        self._network.set_delays(syn_params)


def _overwrite_dir_prompt(expt_dir: Path) -> bool:
    while True:
        ret = input(f"Overwrite existing directory '{expt_dir.name}' y/[n]? ")
        if ret.lower() not in {'y', 'n', ''}:
            print(f"Invalid input '{ret}'")
        else:
            break
    if ret.lower() == 'y':
        shutil.rmtree(expt_dir)
        return True
    return False


def setup_ray(num_cpus: int | None = None):
    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_cpus=num_cpus)
