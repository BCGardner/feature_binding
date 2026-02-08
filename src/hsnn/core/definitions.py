from enum import Enum

__all__ = ["NeuronClass", "SynapseClass", "Projection", "SynEvent"]


class NeuronClass(Enum):
    EXC = 1
    INH = 2


class SynapseClass(Enum):
    PLASTIC = 1
    FIXED = 2


class Projection(Enum):
    FF = 1
    E2I = 2
    I2E = 3
    FB = 4
    E2E = 5

class SynEvent(Enum):
    ON_PRE = 1
    ON_POST = 2
