import numpy as np
import pandas as pd

from hsnn.analysis.measures import _summarise_information


def _frame(values: dict[int, list[float]], sides: list[str]) -> pd.DataFrame:
    return pd.DataFrame.from_dict(
        values, orient="index", columns=sides
    ).rename_axis("nrn")


def test_summarise_information_prefers_argmax_side():
    sides = ["left", "top"]
    convex = _frame({0: [0.9, 0.1], 1: [0.0, 0.0]}, sides)
    concave = _frame({0: [0.0, 0.0], 1: [0.2, 0.8]}, sides)

    summary = _summarise_information(convex, concave, info_threshold=2 / 3)

    # Neuron 0: strongest convex information on 'left'.
    assert summary.loc[0, "pref_side"] == "left"
    assert summary.loc[0, "pref_conformation"] == "convex"
    assert summary.loc[0, "info_bits"] == 0.9
    assert bool(summary.loc[0, "informative"]) is True

    # Neuron 1: strongest concave information on 'top', below threshold.
    assert summary.loc[1, "pref_side"] == "top"
    assert summary.loc[1, "pref_conformation"] == "concave"
    assert summary.loc[1, "info_bits"] == 0.8
    assert bool(summary.loc[1, "informative"]) is True


def test_summarise_information_threshold_and_conformation_tiebreak():
    sides = ["left"]
    # Below 2/3 bits -> not informative. Both conformations zero -> convex tiebreak.
    convex = _frame({0: [0.5], 1: [0.0]}, sides)
    concave = _frame({0: [0.0], 1: [0.0]}, sides)

    summary = _summarise_information(convex, concave, info_threshold=2 / 3)

    assert bool(summary.loc[0, "informative"]) is False
    assert summary.loc[0, "pref_conformation"] == "convex"
    # All-zero neuron: conformation defaults to convex, not informative.
    assert summary.loc[1, "info_bits"] == 0.0
    assert summary.loc[1, "pref_conformation"] == "convex"
    assert bool(summary.loc[1, "informative"]) is False

    assert list(summary.columns) == [
        "pref_side", "pref_conformation", "info_bits", "informative"
    ]
    np.testing.assert_array_equal(summary.index.values, [0, 1])
