from hsnn.simulation._base import _get_uid
from hsnn.core.interfaces import INetwork

__all__ = ["MonitorContext", "StateContext", "ClampContext"]


class MonitorContext:
    def __init__(self, network: INetwork, monitor_spikes: bool = True,
                 monitor_states: bool = False) -> None:
        self.network = network
        self.monitor_spikes = monitor_spikes
        self.monitor_states = monitor_states

    def __enter__(self):
        self._initial_states = {
            'spikes': self.network.monitor_spikes,
            'states': self.network.monitor_states
        }
        self.network.monitor_spikes = self.monitor_spikes
        self.network.monitor_states = self.monitor_states
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.network.monitor_spikes = self._initial_states['spikes']
        self.network.monitor_states = self._initial_states['states']


class StateContext:
    def __init__(self, network: INetwork) -> None:
        self.network = network

    def __enter__(self):
        self.state_name = _get_uid(self.network.store_names, 'tmp')
        self.network.store(self.state_name)
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.network.restore(self.state_name)
        self.network.remove_store(self.state_name)


class ClampContext:
    def __init__(self, network: INetwork) -> None:
        self.network = network

    def __enter__(self):
        self._initial_states = {
            'clamp': self.network.clamp_voltages,
            'lrate': self.network.lrate
        }
        self.network.clamp_voltages = True
        self.network.lrate = 0
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.network.clamp_voltages = self._initial_states['clamp']
        self.network.lrate = self._initial_states['lrate']
