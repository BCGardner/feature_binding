import pytest
from typing import Optional

import numpy as np
from sqlalchemy.exc import IntegrityError

import hsnn.analysis.png.db as polydb
from hsnn.analysis import png


@pytest.fixture(scope='module')
def polygrps() -> list[png.PNG]:
    return [
        png.PNG(
            layers=np.array([3, 3, 4]), nrns=np.array([1802, 1996, 1998]),
            lags=np.array([0., 3., 10.]), times=np.array([587., 9084., 9873.])
        ),
        png.PNG(
            layers=np.array([3, 4, 4]), nrns=np.array([1612, 1666, 1664]),
            lags=np.array([0., 3., 10.]), times=np.array([858., 6335., 17834.])
        ),
        png.PNG(
            layers=np.array([3, 4, 4]), nrns=np.array([2648, 2314, 2379]),
            lags=np.array([0., 4., 10.]), times=np.array([14781., 15315., 18556])
        ),
        png.PNG(
            layers=np.array([3, 4, 4]), nrns=np.array([1419, 1294, 1292]),
            lags=np.array([0., 4., 11.]), times=np.array([849., 7317., 7777.])
        ),
        png.PNG(
            layers=np.array([3, 4, 4]), nrns=np.array([2318, 1294, 1357]),
            lags=np.array([0., 5., 9.]), times=np.array([93., 6356., 19153.])
        )
    ]


@pytest.fixture(scope='module')
def polygrps_dup() -> list[png.PNG]:
    return [
        png.PNG(
            layers=np.array([3, 4, 4]), nrns=np.array([1419, 1294, 1292]),
            lags=np.array([0., 4., 11.]), times=np.array([849., 7317., 7777.])
        ),
        png.PNG(
            layers=np.array([3, 4, 4]), nrns=np.array([1419, 1294, 1292]),
            lags=np.array([0., 4., 11.]), times=np.array([93., 6356., 19153.])
        )
    ]


@pytest.fixture
def png_db(tmp_path) -> polydb.PNGDatabase:
    db = polydb.PNGDatabase(tmp_path / 'polydb.db')
    db.create()
    return db


def _get_equal_png(query: png.PNG, polygrps: list[png.PNG]) -> Optional[png.PNG]:
    for polygrp in polygrps:
        if polygrp == query:
            return polygrp
    return None


def test_get_pngs(png_db: polydb.PNGDatabase, polygrps: list[png.PNG]):
    # Bulk records
    png_models = png_db.get_records(polydb.PNGModel)
    assert len(png_models) == 0
    png_db.insert_pngs(polygrps)
    png_models = png_db.get_records(polydb.PNGModel)
    assert len(png_models) == len(polygrps)
    # Select specific IDs
    polygrps_db = png_db.get_pngs(4, [1294], 1)
    assert len(polygrps_db) == 2
    # Select non-existent IDs
    polygrps_db = png_db.get_pngs(4, [42], 1)
    assert len(polygrps_db) == 0


def test_png_integrity(png_db: polydb.PNGDatabase, polygrps: list[png.PNG]):
    # Assert no duplicates
    png_db.insert_pngs(polygrps)
    assert len(png_db.get_records(polydb.PNGModel)) == len(polygrps)
    png_db.insert_pngs(polygrps)
    assert len(png_db.get_records(polydb.PNGModel)) == len(polygrps)
    # Compare reconstructed PNGs against original data
    polygrps_db = png_db.get_all_pngs()
    assert len(polygrps_db) == len(polygrps)
    for polygrp_db in polygrps_db:
        assert polygrp_db in polygrps
        polygrp = _get_equal_png(polygrp_db, polygrps)
        assert polygrp is not None
        np.testing.assert_allclose(polygrp_db.layers, polygrp.layers)
        np.testing.assert_allclose(polygrp_db.nrns, polygrp.nrns)
        np.testing.assert_allclose(polygrp_db.lags, polygrp.lags)
        np.testing.assert_allclose(polygrp_db.times, polygrp.times)


def test_png_duplicates(png_db: polydb.PNGDatabase, polygrps_dup: list[png.PNG]):
    # Rollback inserting a sequence containing duplicate PNGs (by hash value).
    with pytest.raises(IntegrityError):
        png_db.insert_pngs(polygrps_dup)
    assert len(png_db.get_all_pngs()) == 0
    # Prevent second duplicate insertion
    png_db.insert_pngs(polygrps_dup[:1])
    assert len(png_db.get_all_pngs()) == 1
    png_db.insert_pngs(polygrps_dup[1:])
    assert len(png_db.get_all_pngs()) == 1


def test_runs(png_db: polydb.PNGDatabase, polygrps: list[png.PNG]):
    # Positional index of inserted neuronIDs
    index = 1
    # Split groups to test insertion
    polygrps_0 = polygrps[:1]
    polygrps_1 = polygrps[1:3]
    polygrps_2 = polygrps[3:]
    nrn_ids_0 = [polygrp.nrns[index] for polygrp in polygrps_0]
    nrn_ids_1 = [polygrp.nrns[index] for polygrp in polygrps_1]
    nrn_ids_2 = [polygrp.nrns[index] for polygrp in polygrps_2]
    # Insert group 0 (skip duplicate)
    png_db.insert_runs(nrn_ids_0, 3, index)
    assert 1996 in png_db.get_run_nrns(3, index)
    png_db.insert_runs(nrn_ids_0, 3, index)
    assert len(png_db.get_run_nrns(3, index)) == 1
    # Insert group 1
    png_db.insert_runs(nrn_ids_1, 4, index)
    assert len(png_db.get_run_nrns(4, index)) == 2
    # Insert group 2 (skip duplicate)
    png_db.insert_runs(nrn_ids_2, 4, index)
    assert len(png_db.get_run_nrns(4, index)) == 3


def test_insertion_integrity(png_db: polydb.PNGDatabase, polygrps: list[png.PNG]):
    # Insert overlapping PNGs
    polygrps_0 = polygrps[:3]
    polygrps_1 = polygrps[3:]
    png_db.insert_pngs(polygrps_0)
    assert len(png_db.get_all_pngs()) == 3
    png_db.insert_pngs(polygrps_1)
    assert len(png_db.get_all_pngs()) == 5
    # Insert overlapping run IDs
    index = 1
    nrn_ids_0 = [polygrp.nrns[index] for polygrp in polygrps_0]
    nrn_ids_1 = [polygrp.nrns[index] for polygrp in polygrps_1]

    # png_db.insert_runs()
