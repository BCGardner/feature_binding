import re
from enum import Enum
from typing import Dict, Mapping, MutableSequence, Optional, Sequence

from ..definitions import NeuronClass, SynapseClass, Projection

NAMESPACE_TYPE_MAPPING = {
    'neurons': NeuronClass,
    'synapses': SynapseClass
}


def _split_symbol(symbol: str) -> tuple[str, Optional[int]]:
    """Split symbol at the end pattern '[*]' where * is an integer, returning each part.
    """
    match = re.search(r'\[(-?\d+)\]$', symbol)
    if match:
        return symbol[:match.start()], int(match.group(1))
    else:
        return symbol, None


class ProjectionSet(Sequence):
    def __init__(self, paramset: Mapping, num_hidden: int) -> None:
        self._paramset = paramset
        self._num_items = num_hidden + 1

    def __getitem__(self, index):
        ret = {}
        for proj_type, proj_params in self._paramset.items():
           ret[proj_type] = {}
           for key, kwargs in proj_params.items():  # e.g. connect_kwargs, namespace, ...
               ret[proj_type][key] = self._get_elements(index, **kwargs)
        return ret

    def __len__(self) -> int:
        return self._num_items

    def _get_elements(self, index, **kwargs):
        dst = {}
        for key, val in kwargs.items():  # e.g. namespace kwargs
            name, idx = _split_symbol(key)
            if idx is not None:
                idx = idx + len(self) if idx < 0 else idx
                if idx == index:
                    dst[name] = val
            elif isinstance(val, MutableSequence):
                dst[name] = val[:self._num_items][index]
            else:
                dst[name] = val
        return dst


class ConfigParser:
    @staticmethod
    def parse_namespaces(cfg: Mapping) -> Dict[str, Dict[Enum, dict]]:
        assert set(cfg.keys()) == set(NAMESPACE_TYPE_MAPPING.keys()), "missing namespace(s)"

        namespaces: Dict[str, Dict[Enum, dict]] = {}
        for grp_key, symbol_maps in cfg.items():  # e.g. {'neurons', ...}
            group_types = NAMESPACE_TYPE_MAPPING[grp_key]  # NeuronClass
            namespaces[grp_key] = {}
            for key, symbol_map in symbol_maps.items():
                group_type = group_types[key]  # {NeuronClass.EXC, ...}
                namespaces[grp_key][group_type] = dict(symbol_map)
        return namespaces

    @staticmethod
    def parse_topology(cfg: Mapping) -> Dict[str, Dict[NeuronClass, tuple]]:
        paramset: Dict[str, Dict[NeuronClass, tuple]] = {}
        for layer_key, kwargs in cfg.items():
            paramset[layer_key] = {}
            for key, val in kwargs.items():
                group_key = NeuronClass[key]
                paramset[layer_key][group_key] = tuple(val)
        return paramset

    @staticmethod
    def parse_projections(cfg: Mapping, num_hidden: int) -> Sequence[Dict[Projection, dict]]:
        paramset = {}
        for proj_key, symbol_maps in cfg.items():  # e.g. 'FF'
            proj_type = Projection[proj_key]  # Projection.FF
            paramset[proj_type] = {key: dict(kwargs) for key, kwargs in symbol_maps.items()}
        return ProjectionSet(paramset, num_hidden)
