from typing import Iterable

from hsnn.analysis.png.db import PNGDatabase
from hsnn.analysis import ResultsDatabase
from hsnn.cluster import tasks
from hsnn.core.logger import get_logger

__all__ = [
    'detect_pngs',
    'detect_unconstrained_pngs',
    'detect_significant_pngs'
]

logger = get_logger(__name__)


def detect_pngs(
    db: PNGDatabase, rdb: ResultsDatabase, position: int = 1,
    layers: Iterable[int] | None = None, num_workers: int | None = None,
    spade_kwargs: dict | None = None,
) -> None:
    """Detect triplet PNGs (HFBs) with synaptic constraints.

    Args:
        db: Database to store detected PNGs.
        rdb: Spike recordings database.
        position: Firing order index for DB retrieval (default 1 = second-firing).
        layers: List of target layers to process.
        num_workers: Number of parallel workers.
        spade_kwargs: Additional SPADE parameters.
    """
    layer_ids = _get_layer_ids(rdb, layers)
    for layer in layer_ids:
        nrn_ids = sorted(rdb.syn_params.xs(layer, level='layer').index.unique('post'))
        polygrps = tasks.get_or_detect(
            nrn_ids, layer, position, db, rdb, method='structural',
            num_workers=num_workers, spade_kwargs=spade_kwargs,
        )
        logger.info(f"Layer {layer}: # HFBs={len(polygrps)}\n")


def detect_unconstrained_pngs(
    db: PNGDatabase, rdb: ResultsDatabase, position: int = 1,
    layers: Iterable[int] | None = None, num_workers: int | None = None,
    spade_kwargs: dict | None = None,
) -> None:
    """Detect triplet PNGs with [L-1, L, L] structure WITHOUT synaptic constraints.

    This detects all three-neuron PNGs where:
    - First neuron is in layer L-1
    - Second and third neurons are in layer L
    - No weight or delay constraints are enforced

    Args:
        db: Database to store detected PNGs.
        rdb: Spike recordings database.
        position: Firing order index for DB retrieval (default 1 = second-firing).
        layers: List of target layers to process.
        num_workers: Number of parallel workers.
        spade_kwargs: Additional SPADE parameters.
    """
    layer_ids = _get_layer_ids(rdb, layers)
    for layer in layer_ids:
        nrn_ids = sorted(rdb.syn_params.xs(layer, level='layer').index.unique('post'))
        polygrps = tasks.get_or_detect(
            nrn_ids, layer, position, db, rdb, method='unconstrained',
            num_workers=num_workers, spade_kwargs=spade_kwargs,
        )
        logger.info(f"Layer {layer}: # PNGs (unconstrained)={len(polygrps)}\n")


def detect_significant_pngs(
    db_sgnf: PNGDatabase, db: PNGDatabase, rdb: ResultsDatabase, position: int = 1,
    layers: Iterable[int] | None = None, num_workers: int | None = None,
    spade_kwargs: dict | None = None,
) -> None:
    """Identify significant PNGs corresponding to three-neuron HFB circuits
    from previously detected ones.

    Args:
        db_sgnf (PNGDatabase): Database to store significant PNGs.
        db (PNGDatabase): Database with previously detected PNGs.
        rdb (ResultsDatabase): Spike recordings database.
        position (int, optional): Firing order of the high-level feature neuron. Defaults to 1.
        layers (Iterable[int], optional): List of layers to process. Defaults to None.
        num_workers (int, optional): Number of Ray workers. Defaults to None.
        spade_kwargs (dict, optional): Additional arguments for SPADE. Defaults to None.
    """
    layer_ids = _get_layer_ids(rdb, layers)
    for layer in layer_ids:
        nrn_ids_expected = sorted(rdb.syn_params.xs(layer, level='layer').index.unique('post'))
        # Get prior runs and detections
        nrn_ids = db.get_run_nrns(layer, position)
        assert set(nrn_ids) == set(nrn_ids_expected), "Detections missing for some nrns"
        polygrps = db.get_pngs(layer, nrn_ids, position)
        # Get significant PNGs from prior detections
        sgnf_tests = tasks.test_significance(
            polygrps, rdb, num_workers=num_workers, spade_kwargs=spade_kwargs
        )
        # Insert significant PNGs and associated run logs
        polygrps_sgnf = [polygrps[i] for i in range(len(sgnf_tests)) if sgnf_tests[i]]
        db_sgnf.insert_pngs(polygrps_sgnf)
        db_sgnf.insert_runs(nrn_ids, layer, position)
        logger.info(f"Layer {layer}: # HFBs (sgnf)={len(polygrps_sgnf)}\n")


def _get_layer_ids(
    rdb: ResultsDatabase, layers: Iterable[int] | None = None
) -> list[int]:
    _layers = rdb.get_coord_values('layer')[1:] if layers is None else layers
    return [int(layer) for layer in _layers]
