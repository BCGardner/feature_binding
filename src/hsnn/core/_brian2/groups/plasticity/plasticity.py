from brian2 import Equations

from ._base import PlasticityRule
from ..codeblock import CodeBlock

__all__ = ["STDPRule"]


class STDPRule(PlasticityRule):
    _model = Equations('''
    dC/dt = -C/tau_C : 1 (event-driven)
    dD/dt = -D/tau_D : 1 (event-driven)
    ''')
    _on_pre = CodeBlock('''
    C = clip(C + alpha_C*(1-C), 0, 1)
    w = clip(w - rho * w**mu * D, 0, w_max)
    ''')
    _on_post = CodeBlock('''
    D = clip(D + alpha_D*(1-D), 0, 1)
    w = clip(w + rho * (w_max - w)**mu * C, 0, w_max)
    ''')

    @property
    def model(self) -> Equations:
        return self._model

    @property
    def on_pre(self) -> CodeBlock:
        return self._on_pre

    @property
    def on_post(self) -> CodeBlock:
        return self._on_post
