from typing import Any, Dict, Mapping, MutableSequence, Optional

from brian2 import Quantity, msecond, mvolt, nfarad, nsiemens, hertz


UNIT_MAP = {
    'C':                nfarad,     # Capacitance
    'V':                mvolt,      # Voltage
    'v':                mvolt,
    'delay':            msecond,    # Axonal conduction delay
    'dt':               msecond,    # Clock step
    'duration':         msecond,    # Sample duration
    'eta':              nsiemens,   # Synaptic scaling factor
    'g':                nsiemens,   # Conductance
    'tau':              msecond,    # Time constant
    'rate':             hertz,      # Firing rate
}


def process_symbols(symbol_map: Mapping[str, Any]) -> Dict[str, Any]:
    """Processes a symbol map: eligible values are transformed into Brian2
    Quantities according to their keys.

    Args:
        symbol_map (Mapping[str, Any]): Symbol map linking symbolic names to values.

    Returns:
        Dict[str, Any]: Processed symbol map.
    """
    dst = dict(symbol_map)
    for key, val in dst.items():
        if isinstance(val, MutableSequence):
            dst[key] = [process_value(elem, key) for elem in val]
        else:
            dst[key] = process_value(val, key)
    return dst


def process_value(value: Any, symbol_name: str) -> Any:
    """Process a single value with an associated symbolic name.

    Args:
        value (Any): Value to be processed.
        symbol_name (str): Symbolic key.

    Returns:
        Any: Processed value.
    """
    if not isinstance(value, (Quantity, str)):
        associated_unit = retrieve_unit(symbol_name)
        if associated_unit is not None:
            return value * associated_unit
    return value


def retrieve_unit(symbol_name: str) -> Optional[Quantity]:
    """Provides the unit linked with a symbol name.
    Args:
        symbol_name (str): Symbolic key.

    Returns:
        Quantity: Brian2 unit.
    """
    for unit_key, unit_value in UNIT_MAP.items():
        if symbol_name.startswith(unit_key + '_') or symbol_name == unit_key:
            return unit_value  # type: ignore
    return None
