from typing import Any, Dict

import numpy as np

__all__ = ["assert_netstate_equal"]


def assert_netstate_equal(state1: Dict[str, Dict[str, Any]], state2: Dict[str, Dict[str, Any]]):
    state_keys = state1.keys()
    assert state2.keys() == state_keys, "state keys differ at network level"
    for state_key in state_keys:
        grp1 = state1[state_key]
        grp2 = state2[state_key]
        assert grp1.keys() == grp2.keys(), "state keys differ at group level"
        for grp_key in grp1.keys():
            assert np.array_equal(grp1[grp_key], grp2[grp_key]), "mismatched group values"
