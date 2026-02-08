from collections import defaultdict
from typing import Iterable, Sequence

from hsnn.analysis import png
from hsnn.core.logger import logging

logger = logging.getLogger()


def chunks(lst, k):
    """Yield k nearly-equal slices of lst.
    """
    n = len(lst)
    step = (n + k - 1) // k
    for i in range(0, n, step):
        yield i, lst[i : i + step]


def log_png_hash_collisions(
    layer: int,
    nrn_ids: Iterable[int],
    detections: Sequence[Sequence[png.PNG]],
) -> None:
    """Log PNG hash collisions across different nrn_ids."""
    nrn_ids_list = list(nrn_ids)[:len(detections)]
    hash_to_nrns: dict[int, set[int]] = defaultdict(set)
    hash_to_pngs: dict[int, list[png.PNG]] = defaultdict(list)

    for nrn_id, pngs_for_nrn in zip(nrn_ids_list, detections):
        for polygrp in pngs_for_nrn:
            h = hash(polygrp)
            hash_to_nrns[h].add(nrn_id)
            hash_to_pngs[h].append(polygrp)

    for h, ids in hash_to_nrns.items():
        if len(ids) > 1:
            logger.warning(
                "PNG hash collision across nrn_ids at layer %d: hash=%d, nrn_ids=%s, count=%d",
                layer, h, sorted(ids), len(hash_to_pngs[h]),
            )
            for i, polygrp in enumerate(hash_to_pngs[h]):
                logger.warning(
                    "  PNG[%d] hash=%d layers=%s nrns=%s lags=%s n_times=%d",
                    i, h, polygrp.layers, polygrp.nrns, polygrp.lags, len(polygrp.times)
                )
