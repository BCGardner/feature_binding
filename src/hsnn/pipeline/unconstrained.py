from typing import Sequence

import numpy as np
import pandas as pd
from hsnn.analysis.png import PNG
from hsnn.analysis.png.refinery import Connected

__all__ = ["filter_valid_hfb_triplets"]


def filter_valid_hfb_triplets(
    pngs: Sequence[PNG],
    syn_params: pd.DataFrame,
) -> tuple[list[PNG], dict]:
    """Filter unconstrained triplet PNGs to valid unique HFB circuits.

    Handles tied-lag ambiguity where High and Bind neurons have identical
    firing times, causing arbitrary ordering by neuron ID. When lags are tied:
    - If the reversed ordering exists in the collection, this is a duplicate artifact
    - If only one ordering exists, check if reversed connectivity is valid

    Args:
        pngs: Sequence of triplet PNGs with structure [L-1, L, L].
        syn_params: Network synaptic parameters with index [layer, proj, pre, post].

    Returns:
        Tuple of (valid_pngs, stats) where:
        - valid_pngs: List of PNGs with valid HFB connectivity (deduplicated)
        - stats: Dict with counts for n_total, n_valid, n_tied_lags,
                 n_tied_lag_duplicates, n_no_connectivity
    """
    # Build set of all (layers, nrns) keys in the collection
    all_keys = {_get_png_key(p) for p in pngs}
    key_to_png: dict[tuple, PNG] = {_get_png_key(p): p for p in pngs}

    valid_keys: set[tuple] = set()
    n_tied_lag_duplicates = 0
    n_no_connectivity = 0

    connected = Connected(syn_params)

    for png in pngs:
        key = _get_png_key(png)

        if len(connected([png])):
            # PNG passes connectivity as-is
            valid_keys.add(key)
        elif _has_tied_hb_lags(png):
            # Tied lags: check if reversed ordering would be valid
            reversed_key = _get_reversed_hb_key(png)

            if reversed_key in all_keys:
                # The valid ordering already exists -> this is a duplicate artifact
                n_tied_lag_duplicates += 1
            else:
                # Check if reversed ordering has valid connectivity
                if _check_reversed_connectivity(png, syn_params):
                    # Reversed ordering is valid -> count under reversed key
                    valid_keys.add(reversed_key)
                else:
                    # Neither ordering has valid connectivity
                    n_no_connectivity += 1
        else:
            # Distinct lags but fails connectivity
            n_no_connectivity += 1

    # Reconstruct valid PNGs from keys
    # For reversed keys that don't exist in original collection, we keep the
    # original PNG but note it represents the reversed circuit
    valid_pngs = []
    for key in valid_keys:
        if key in key_to_png:
            valid_pngs.append(key_to_png[key])
        else:
            # This was a reversed key - find the original and note it
            # The original has neurons [L, H, B] but valid circuit is [L, B, H]
            reversed_key = (key[0], (key[1][0], key[1][2], key[1][1]))
            if reversed_key in key_to_png:
                orig_png = key_to_png[reversed_key]
                # Create corrected PNG with reversed H/B neurons
                valid_pngs.append(PNG(
                    layers=orig_png.layers,
                    nrns=np.array([orig_png.nrns[0], orig_png.nrns[2], orig_png.nrns[1]]),
                    lags=orig_png.lags,  # lags are tied so order doesn't matter
                    times=orig_png.times,
                ))

    stats = {
        "n_total": len(pngs),
        "n_valid": len(valid_keys),
        "n_tied_lags": sum(1 for p in pngs if _has_tied_hb_lags(p)),
        "n_tied_lag_duplicates": n_tied_lag_duplicates,
        "n_no_connectivity": n_no_connectivity,
    }

    return valid_pngs, stats


def _get_png_key(png: PNG) -> tuple:
    """Get (layers, nrns) tuple key for a PNG."""
    return (tuple(png.layers), tuple(png.nrns))


def _get_reversed_hb_key(png: PNG) -> tuple:
    """Get key with High/Bind neurons (positions 1,2) reversed."""
    return (tuple(png.layers), (png.nrns[0], png.nrns[2], png.nrns[1]))


def _has_tied_hb_lags(png: PNG) -> bool:
    """Check if High and Bind neurons (positions 1,2) have identical lags."""
    return len(png.lags) >= 3 and png.lags[1] == png.lags[2]


def _check_reversed_connectivity(png: PNG, syn_params: pd.DataFrame) -> bool:
    """Check if the reversed High/Bind ordering has valid connectivity."""
    low_layer, high_layer, bind_layer = png.layers
    low_nrn = png.nrns[0]
    # Reversed: what was "bind" becomes "high", what was "high" becomes "bind"
    new_high, new_bind = png.nrns[2], png.nrns[1]

    low_to_new_high = (high_layer, "FF", low_nrn, new_high) in syn_params.index
    low_to_new_bind = (bind_layer, "FF", low_nrn, new_bind) in syn_params.index
    new_high_to_new_bind = (high_layer, "E2E", new_high, new_bind) in syn_params.index

    return low_to_new_high and low_to_new_bind and new_high_to_new_bind
