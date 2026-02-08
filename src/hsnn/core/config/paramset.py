from dataclasses import dataclass, field, InitVar
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from omegaconf import DictConfig, OmegaConf

from .parser import ConfigParser
from ..definitions import NeuronClass, Projection


def _as_dict(cfg: str | Path | Mapping) -> dict:
    if isinstance(cfg, (str, Path)):
        cfg_ = OmegaConf.to_container(OmegaConf.load(cfg))
    elif isinstance(cfg, DictConfig):
        cfg_ = OmegaConf.to_container(cfg)
    else:
        cfg_ = dict(cfg)
    assert isinstance(cfg_, dict)
    return cfg_


@dataclass
class ModelParams:
    model_cfg: InitVar[str | Path | Mapping]

    network: Dict[str, Any] = field(init=False)
    encoder: Dict[str, Any] = field(init=False, repr=False)
    topology: Dict[str, Dict[NeuronClass, tuple]] = field(init=False, repr=False)
    projections: Sequence[Dict[Projection, dict]] = field(init=False, repr=False)
    namespaces: Dict[str, Dict[Enum, dict]] = field(init=False, repr=False)

    def __post_init__(self, model_cfg: str | Path | Mapping) -> None:
        cfg = _as_dict(model_cfg)
        self.network = cfg['network']
        self.encoder = cfg['encoder']
        self.topology = ConfigParser.parse_topology(cfg['topology'])
        self.projections = ConfigParser.parse_projections(cfg['projections'],
                                                          self.network['num_hidden'])
        self.namespaces = ConfigParser.parse_namespaces(cfg['namespaces'])
