from copy import deepcopy
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from brian2 import CodeRunner, Group, NeuronGroup, Synapses, Equations

from ...definitions import NeuronClass, SynapseClass, SynEvent
from ._base import GroupFactory
from . import helper as hp
from .codeblock import CodeBlock

__all__ = ["COBAFactory"]


_DYNAMICS = Equations('''
dv/dt = (g_l * (V_0 - v) + g_e * (V_e - v) + g_i * (V_i - v))/C_m : volt (unless refractory)
dg_e/dt = -g_e/tau_e : siemens
dg_i/dt = -g_i/tau_i : siemens
''')

_ADAPTIVE = Equations('''
dv/dt = (g_l * (V_0 - v) + g_e * (V_e - v) + g_i * (V_i - v) + g_a * (V_a - v))/C_m : volt (unless refractory)
dg_e/dt = -g_e/tau_e : siemens
dg_i/dt = -g_i/tau_i : siemens
dg_a/dt = -g_a/tau_a : siemens
''')


_SYN_EVENTS: Dict[SynEvent, Dict[NeuronClass, CodeBlock]] = {
    SynEvent.ON_PRE: {
        NeuronClass.EXC: CodeBlock('g_e_post += eta * w'),
        NeuronClass.INH: CodeBlock('g_i_post += eta * w')
    },
    SynEvent.ON_POST: {
        NeuronClass.EXC: CodeBlock(),
        NeuronClass.INH: CodeBlock()
    },
}


class COBAFactory(GroupFactory):
    def __post_init__(self) -> None:
        self.state_records.update({
            'neurons':  ['v', 'g_e', 'g_i']
        })

    def create_neurons(self, num_nrns: Sequence, identifier: NeuronClass,
                       name: str = 'neurongroup*', **kwargs) -> NeuronGroup:
        assert len(num_nrns) == 2, "2D square shape required"
        assert num_nrns[0] == num_nrns[1], "2D square shape required"
        attrs_init = kwargs.pop('attrs_init', {})
        namespace = self._namespaces['neurons'][identifier]
        isadaptive = {'V_a', 'eta_a', 'tau_a'}.issubset(namespace)
        equations = _ADAPTIVE if isadaptive else _DYNAMICS
        equations += self.spatial_eqs
        reset = 'v = V_h; g_a += eta_a' if isadaptive else 'v = V_h'
        group = NeuronGroup(
            np.prod(num_nrns), equations, self._integ_method, threshold='v > V_thr',
            reset=reset, refractory='tau_r', namespace=namespace, name=name, **kwargs # type: ignore
        )
        attrs_init = hp.process_kwargs({'v': 'V_0'}, **attrs_init)
        xs, ys = hp.get_spatial_coords(num_nrns, spatial_span=self._spatial_span)
        hp.set_group_attr(group, identifier=identifier, x=xs, y=ys, **attrs_init)
        self._update_network(group)
        return group

    def create_synapses(self, source: NeuronGroup, target: NeuronGroup,
                        identifier: SynapseClass, name: str = 'synapses*', **kwargs) -> Synapses:
        namespace = deepcopy(self._namespaces['synapses'][identifier])
        namespace.update(**hp.process_symbols(kwargs.pop('namespace', {})))
        connect_kwargs = kwargs.pop('connect_kwargs', {})
        attrs_init = kwargs.pop('attrs_init', {})
        model = self.synapses_model
        if identifier == SynapseClass.PLASTIC:
            model += self._plasticity.model
        on_pre, on_post = self._get_synapse_events(source.identifier, target.identifier, identifier)
        group = Synapses(source, target, model, on_pre=on_pre, on_post=on_post,
                         namespace=namespace, method=self._integ_method, name=name, **kwargs)
        self._connector.connect(group, **connect_kwargs)
        defaults = {
            'w':        'rand() * w_max',
            'delay':    'ceil(delay_max / msecond * rand()) * msecond'
        }
        attrs_init = hp.process_kwargs(defaults, **attrs_init)
        hp.set_group_attr(group, identifier=identifier, **attrs_init)
        self._update_network(group)
        return group

    def create_clamper(self, group: Group, active: bool = False, name: Optional[Any] = None) -> CodeRunner:
        return self._create_coderunner(group, 'v = V_0', active=active, name=name)

    def _get_synapse_events(self, source_cls: NeuronClass, target_cls: NeuronClass,
                            syn_cls: SynapseClass) -> Tuple[Optional[str], Optional[str]]:
        on_pre = _SYN_EVENTS[SynEvent.ON_PRE][source_cls]
        on_post = _SYN_EVENTS[SynEvent.ON_POST][target_cls]
        if syn_cls == SynapseClass.PLASTIC:
            on_pre += self._plasticity.on_pre
            on_post += self._plasticity.on_post
        return on_pre.value, on_post.value
