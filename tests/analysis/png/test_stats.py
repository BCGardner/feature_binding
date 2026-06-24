import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hsnn.analysis.png import stats


@pytest.fixture
def labels() -> pd.DataFrame:
    return pd.DataFrame({
        "image_id": ["a", "b", "c", "d"],
        "left": [1, 1, 0, 0],
        "top": [1, 0, 1, 0],
    })


@pytest.fixture
def occ_array() -> xr.DataArray:
    # png 111 fires on images {0, 1}; png 222 fires on images {2, 3}.
    data = np.zeros((2, 4, 2))
    data[:, 0, 0] = 5.0
    data[:, 1, 0] = 5.0
    data[:, 2, 1] = 5.0
    data[:, 3, 1] = 5.0
    return xr.DataArray(
        data,
        dims=["rep", "img", "png"],
        coords={"rep": [0, 1], "img": range(4), "png": [111, 222]},
    )


def test_get_label_metrics_shape_and_labels(occ_array, labels):
    metrics = stats.get_label_metrics(occ_array, labels)
    # 2 sides x 2 conformations x 2 PNGs
    assert len(metrics) == 8
    assert set(metrics["conformation"]) == {"convex", "concave"}
    assert set(metrics["side"]) == {"left", "top"}
    assert set(metrics["png_id"]) == {111, 222}
    assert list(metrics.columns) == [
        "png_id", "side", "conformation", "precision", "recall", "f1"
    ]


def test_get_label_metrics_matches_precision_recall(occ_array, labels):
    metrics = stats.get_label_metrics(occ_array, labels)
    # png 111 perfectly separates the convex-left images.
    row = metrics.query("png_id == 111 and side == 'left' and conformation == 'convex'")
    assert row["precision"].item() == pytest.approx(1.0)
    assert row["recall"].item() == pytest.approx(1.0)
    assert row["f1"].item() == pytest.approx(1.0)

    # Equivalence with the underlying per-(side, target) primitive.
    precision, recall = stats.precision_recall(occ_array, labels, "left", target=1)
    convex_left = (
        metrics.query("side == 'left' and conformation == 'convex'")
        .set_index("png_id")
        .loc[[111, 222]]
    )
    np.testing.assert_allclose(convex_left["precision"], precision)
    np.testing.assert_allclose(convex_left["recall"], recall)


def test_get_label_metrics_unfiltered(occ_array, labels):
    # No precision filter: every PNG appears for every label, even precision 0.
    metrics = stats.get_label_metrics(occ_array, labels)
    counts = metrics.groupby(["side", "conformation"]).size()
    assert (counts == 2).all()
