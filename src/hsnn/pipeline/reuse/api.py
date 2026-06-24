"""Builds the canonical HFB annotation tables that are reused.

This module is the shared substrate for the analyses. It loads the significance-tested
three-neuron HFB detections for a single representative, post-trained trial of a given
``(experiment, model_type)`` combination, and joins:

* each PNG's structure (constituent neuron ids, layers, relative spike lags, span
  and occurrence count);
* each PNG's per-``(side, conformation)`` selectivity (precision, recall, F1) and
  preferred feature label;
* each constituent neuron's stimulus-specific information and preferred label.

It produces three tidy tables (returned as pandas DataFrames):

``hfb_annotations``
    One row per significant PNG, with provenance, structure, PNG selectivity,
    per-role (L/H/B) constituent selectivity and derived alignment flags.
``png_label_metrics``
    One row per ``(png_id, side, conformation)`` with precision, recall and F1.
``neuron_information``
    One row per ``(layer, neuron)`` with the preferred side/conformation,
    information (bits) and an ``informative`` flag.

The tables are persisted once (parquet) under a ``reuse`` sub-directory of the
representative trial's checkpoint, so later tasks read them from disk rather than
re-querying the detections.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import xarray as xr

from hsnn.analysis import measures
from hsnn.analysis.png import PNG, stats
from hsnn.analysis.png.db import PNGDatabase
from hsnn.core.logger import get_logger
from hsnn.utils import handler, io

from .constants import DEFAULT_DURATION, DEFAULT_OFFSET, REUSE_SUBDIR, TABLE_NAMES
from ._utils import (
    add_alignment_flags,
    join_neuron_information,
    join_png_preferred_label,
    png_record,
)

__all__ = [
    "ReuseInputs",
    "select_representative_trial",
    "resolve_trial",
    "open_store",
    "load_reuse_inputs",
    "load_polygrps",
    "infer_num_reps",
    "build_png_label_metrics",
    "build_neuron_information",
    "build_annotation_table",
    "build_tables",
    "persist_tables",
    "load_tables",
    "audit_tables",
]

logger = get_logger(__name__)

DEFAULT_NRN_IDS = range(4096)
HIGH_INDEX = 1
HFB_SIZE = 3
INPUT_LAYER = 0  # Poisson stimulus layer; excluded from the spiking layers (L1-L4).


@dataclass
class ReuseInputs:
    """Loaded inputs for the annotation-table builder.

    Attributes:
        experiment: Experiment directory (relative to the experiments root).
        model_type: Architecture key (e.g. ``'ALL'`` or ``'SEMI'``).
        trial: The representative trial.
        store: Artifact store positioned at ``checkpoint``.
        checkpoint: Checkpoint index (e.g. ``-1`` for the last/post state).
        cfg: The trial configuration.
        db: The significance-tested HFB database (post state).
        labels: Dataset annotations (one binary column per side plus ``image_id``).
        results: Post-state spike recordings.
        num_recording_reps: Number of stimulus repetitions present in the
            recordings. Note this is **not** the number of repetitions the
            detections were run over (see :func:`infer_num_reps`).
        num_imgs: Number of distinct images.
        spiking_layers: Spiking layers present in the recordings (e.g. ``[1, 2, 3, 4]``).
    """

    experiment: str
    model_type: str
    trial: handler.TrialView
    store: handler.ArtifactStore
    checkpoint: int
    cfg: dict
    db: PNGDatabase
    labels: pd.DataFrame
    results: xr.DataArray
    num_recording_reps: int
    num_imgs: int
    spiking_layers: list[int]


def select_representative_trial(
    expt: handler.ExperimentHandler, model_type: str, analysis_type: str = "detection"
) -> handler.TrialView:
    """Selects the single most-representative trial for a combination.

    The representative is the trial whose final-iteration loss is closest to the
    mean across the combination's replicate trials (the
    :func:`handler.get_closest_samples` rule), restricted to the trials registered
    for ``analysis_type`` under ``model_type`` in the experiment metadata.

    Args:
        expt: The experiment handler.
        model_type: Architecture key (e.g. ``'ALL'``).
        analysis_type: Metadata analysis key. Defaults to ``'detection'``.

    Returns:
        The representative trial.

    Raises:
        KeyError: If ``analysis_type`` is not registered for ``model_type``.
        ValueError: If a unique representative cannot be resolved.
    """
    trials_dict = expt.metadata.get_trials_dict(model_type)
    if analysis_type not in trials_dict:
        raise KeyError(
            f"'{analysis_type}' not in metadata for model_type '{model_type}'; "
            f"available: {list(trials_dict)}"
        )
    trial_names = set(trials_dict[analysis_type])
    closest_dirs = expt.index_to_dir[handler.get_closest_samples(expt.get_summary(-1))] # pyright: ignore[reportArgumentType]
    matches = [name for name in closest_dirs.to_numpy() if name in trial_names]
    if len(matches) != 1:
        raise ValueError(
            f"Expected a unique representative trial for {model_type}/{analysis_type}, "
            f"found {matches} (candidates: {sorted(trial_names)})"
        )
    trial = expt[matches[0]]
    logger.info(f"Representative trial for {model_type}/{analysis_type}: {trial.name}")
    return trial


def resolve_trial(
    expt: handler.ExperimentHandler,
    model_type: str,
    trial: str | int | None = None,
    analysis_type: str = "detection",
) -> handler.TrialView:
    """Resolves the trial to analyse: an explicit one, else the representative.

    Args:
        expt: The experiment handler.
        model_type: Architecture key (e.g. ``'ALL'``).
        trial: An explicit trial name (str) or positional index (int). If
            ``None`` (default), the representative trial is selected via
            :func:`select_representative_trial`.
        analysis_type: Metadata analysis key. Defaults to ``'detection'``.

    Returns:
        The resolved trial.
    """
    if trial is None:
        return select_representative_trial(expt, model_type, analysis_type)
    resolved = expt[trial]
    logger.info(f"Using explicitly-specified trial: {resolved.name}")
    return resolved


def open_store(
    experiment: str,
    model_type: str,
    *,
    trial: str | int | None = None,
    checkpoint: int = -1,
    analysis_type: str = "detection",
) -> handler.ArtifactStore:
    """Opens the artifact store for a combination's checkpoint.

    Lightweight resolver for reading the persisted reuse tables: it resolves the
    trial (the representative one by default) and returns its
    :class:`~hsnn.utils.handler.ArtifactStore` without loading detections or
    recordings. Use with :func:`load_tables`.

    Args:
        experiment: Experiment directory (relative to the experiments root).
        model_type: Architecture key (e.g. ``'ALL'``).
        trial: Explicit trial name/index, or ``None`` for the representative trial.
        checkpoint: Checkpoint index. Defaults to ``-1`` (post/last).
        analysis_type: Metadata analysis key. Defaults to ``'detection'``.

    Returns:
        The artifact store positioned at the resolved checkpoint.
    """
    expt = handler.ExperimentHandler(experiment) # pyright: ignore[reportArgumentType]
    resolved = resolve_trial(expt, model_type, trial, analysis_type)
    return handler.ArtifactStore(resolved, ckpt_idx=checkpoint)


def load_reuse_inputs(
    experiment: str,
    model_type: str,
    *,
    trial: str | int | None = None,
    checkpoint: int = -1,
    analysis_type: str = "detection",
) -> ReuseInputs:
    """Resolves and loads everything the builder needs for one combination.

    Args:
        experiment: Experiment directory (relative to the experiments root).
        model_type: Architecture key (e.g. ``'ALL'``).
        trial: Explicit trial name/index, or ``None`` for the representative trial.
        checkpoint: Checkpoint index. Defaults to ``-1`` (post/last).
        analysis_type: Metadata analysis key. Defaults to ``'detection'``.

    Returns:
        The bundled inputs.
    """
    expt = handler.ExperimentHandler(experiment) # pyright: ignore[reportArgumentType]
    resolved = resolve_trial(expt, model_type, trial, analysis_type)
    store = handler.ArtifactStore(resolved, ckpt_idx=checkpoint)
    cfg = resolved.config
    db = handler.load_detections(resolved, state="post", sgnf=True)["post"]
    imageset, labels = io.get_dataset(cfg["training"]["data"], return_annotations=True)
    results = store.load_results()
    spiking_layers = [
        int(x) for x in np.unique(results["layer"].values) if int(x) != INPUT_LAYER
    ]
    return ReuseInputs(
        experiment=experiment,
        model_type=model_type,
        trial=resolved,
        store=store,
        checkpoint=checkpoint,
        cfg=cfg,
        db=db,
        labels=labels,
        results=results,
        num_recording_reps=int(results.sizes["rep"]),
        num_imgs=len(imageset),
        spiking_layers=spiking_layers,
    )


def load_polygrps(
    db: PNGDatabase,
    layers: Iterable[int],
    index: int = HIGH_INDEX,
    nrn_ids: Iterable[int] = DEFAULT_NRN_IDS,
) -> dict[int, list[PNG]]:
    """Loads the significant PNGs grouped by their high-level (index-1) neuron's layer.

    Args:
        db: The HFB database.
        layers: High-level (index-1) neuron layers to query.
        index: Lag index of the high-level neuron used to group PNGs. Defaults to 1.
        nrn_ids: Candidate high-level neuron ids. Defaults to ``range(4096)``.

    Returns:
        Mapping of high-level (index-1) layer to its (size-3) HFB PNGs.
    """
    polygrps_by_layer: dict[int, list[PNG]] = {}
    for layer in layers:
        polygrps = [
            png for png in db.get_pngs(int(layer), nrn_ids, index)
            if len(png.nrns) == HFB_SIZE
        ]
        if polygrps:
            polygrps_by_layer[int(layer)] = polygrps
        logger.info(f"Layer {layer}: {len(polygrps)} significant HFBs")
    return polygrps_by_layer


def infer_num_reps(polygrps_by_layer: Mapping[int, list[PNG]]) -> int:
    """Infers the number of repetitions the detections were run over.

    The PNG occurrence metrics must be computed over the same set of stimulus
    repetitions used for detection (which is typically fewer than the number of
    repetitions in the recordings). This is recovered from the stored PNGs as the
    largest occurrence repetition index across all PNGs, plus one. Taking the
    global maximum (rather than a per-PNG value, which varies with each PNG's
    activations) yields the detection repetition count.

    Args:
        polygrps_by_layer: PNGs grouped by high-level (index-1) layer (from :func:`load_polygrps`).

    Returns:
        The number of detection repetitions.

    Raises:
        ValueError: If no PNG carries occurrence repetition indices.
    """
    max_rep = -1
    for polygrps in polygrps_by_layer.values():
        for png in polygrps:
            if png.reps is not None and len(png.reps):
                max_rep = max(max_rep, int(np.max(png.reps)))
    if max_rep < 0:
        raise ValueError(
            "Cannot infer num_reps: no PNG occurrences with repetition indices."
        )
    return max_rep + 1


def build_png_label_metrics(
    polygrps_by_layer: Mapping[int, list[PNG]],
    labels: pd.DataFrame,
    num_reps: int,
    num_imgs: int,
    duration: float = DEFAULT_DURATION,
    offset: float = DEFAULT_OFFSET,
    index: int = HIGH_INDEX,
) -> pd.DataFrame:
    """Builds the ``png_label_metrics`` table (one row per PNG and feature label).

    Args:
        polygrps_by_layer: PNGs grouped by high-level (index-1) layer (from :func:`load_polygrps`).
        labels: Dataset annotations.
        num_reps: Number of stimulus repetitions.
        num_imgs: Number of distinct images.
        duration: Observation window (ms). Defaults to 200.0.
        offset: Observation offset (ms). Defaults to 50.0.
        index: Lag index of the high-level neuron. Defaults to 1.

    Returns:
        Long-form metrics with columns ``png_id``, ``side``, ``conformation``,
        ``precision``, ``recall`` and ``f1``.
    """
    frames = []
    for layer, polygrps in polygrps_by_layer.items():
        occ_array = stats.get_occurrences_array(
            polygrps, num_reps, num_imgs, index=index, duration=duration, offset=offset
        )
        occ_array = occ_array.assign_coords(png=[hash(png) for png in polygrps])
        frames.append(stats.get_label_metrics(occ_array, labels))
    if not frames:
        return pd.DataFrame(
            columns=["png_id", "side", "conformation", "precision", "recall", "f1"]
        )
    return pd.concat(frames, ignore_index=True)


def build_neuron_information(
    results: xr.DataArray,
    labels: pd.DataFrame,
    layers: Iterable[int],
    duration: float = DEFAULT_DURATION,
    offset: float = DEFAULT_OFFSET,
    nrn_cls: str = "EXC",
) -> pd.DataFrame:
    """Builds the ``neuron_information`` table for all relevant layers.

    Args:
        results: Post-state spike recordings.
        labels: Dataset annotations.
        layers: Spiking layers to summarise (expected 1-4).
        duration: Observation window (ms). Defaults to 200.0.
        offset: Observation offset (ms). Defaults to 50.0.
        nrn_cls: Neuron class. Defaults to ``'EXC'``.

    Returns:
        Table with columns ``layer``, ``neuron``, ``pref_side``,
        ``pref_conformation``, ``info_bits`` and ``informative``.
    """
    frames = []
    for layer in layers:
        summary = measures.summarise_neuron_information(
            results, labels, duration, offset, int(layer), nrn_cls
        )
        summary = summary.reset_index().rename(columns={"nrn": "neuron"})
        summary.insert(0, "layer", int(layer))
        frames.append(summary)
    return pd.concat(frames, ignore_index=True)


def build_annotation_table(
    polygrps_by_layer: Mapping[int, list[PNG]],
    png_label_metrics: pd.DataFrame,
    neuron_information: pd.DataFrame,
    *,
    experiment: str,
    model_type: str,
    trial_id: str,
    checkpoint: int,
) -> pd.DataFrame:
    """Builds the canonical ``hfb_annotations`` table (one row per PNG).

    Joins each PNG's structure with its preferred feature label (highest-F1 label
    among those with precision > 0.5) and its constituent neurons' preferred labels
    and information, then derives the L/H, L/H/PNG and L/H/B/PNG alignment flags.

    Args:
        polygrps_by_layer: PNGs grouped by high-level (index-1) layer.
        png_label_metrics: The ``png_label_metrics`` table.
        neuron_information: The ``neuron_information`` table.
        experiment: Experiment directory.
        model_type: Architecture key.
        trial_id: Representative trial name.
        checkpoint: Checkpoint index.

    Returns:
        The annotation table.
    """
    ann = pd.DataFrame([
        png_record(png, experiment, model_type, trial_id, checkpoint)
        for polygrps in polygrps_by_layer.values()
        for png in polygrps
    ])
    ann = join_png_preferred_label(ann, png_label_metrics)
    ann = join_neuron_information(ann, neuron_information)
    ann = add_alignment_flags(ann)
    return ann


def build_tables(
    experiment: str,
    model_type: str,
    *,
    trial: str | int | None = None,
    checkpoint: int = -1,
    duration: float = DEFAULT_DURATION,
    offset: float = DEFAULT_OFFSET,
    index: int = HIGH_INDEX,
    nrn_ids: Iterable[int] = DEFAULT_NRN_IDS,
    num_reps: int | None = None,
    analysis_type: str = "detection",
) -> tuple[dict[str, pd.DataFrame], ReuseInputs]:
    """Loads inputs and builds the three annotation tables for one combination.

    Args:
        experiment: Experiment directory (relative to the experiments root).
        model_type: Architecture key (e.g. ``'ALL'``).
        trial: Explicit trial name/index, or ``None`` for the representative trial.
            Specifying a trial supports quantifying the spread of metrics across
            the combination's replicate trials.
        checkpoint: Checkpoint index. Defaults to ``-1``.
        duration: Observation window (ms). Defaults to 200.0.
        offset: Observation offset (ms). Defaults to 50.0.
        index: Lag index of the high-level neuron. Defaults to 1.
        nrn_ids: Candidate high-level neuron ids. Defaults to ``range(4096)``.
        num_reps: Number of stimulus repetitions used for the occurrence/F1
            metrics. If ``None`` (default), it is inferred from the stored PNGs
            via :func:`infer_num_reps` (the detection repetition count, which is
            typically fewer than the recording repetition count).
        analysis_type: Metadata analysis key. Defaults to ``'detection'``.

    Returns:
        A ``{name: DataFrame}`` mapping (keys = :data:`TABLE_NAMES`) and the loaded
        :class:`ReuseInputs`.
    """
    inputs = load_reuse_inputs(
        experiment, model_type, trial=trial, checkpoint=checkpoint,
        analysis_type=analysis_type,
    )
    polygrps_by_layer = load_polygrps(inputs.db, inputs.spiking_layers, index, nrn_ids)
    resolved_num_reps = (
        num_reps if num_reps is not None else infer_num_reps(polygrps_by_layer)
    )
    logger.info(
        f"Occurrence metrics use num_reps={resolved_num_reps} "
        f"(recordings have {inputs.num_recording_reps} reps)"
    )
    png_label_metrics = build_png_label_metrics(
        polygrps_by_layer, inputs.labels, resolved_num_reps, inputs.num_imgs,
        duration, offset, index,
    )
    neuron_information = build_neuron_information(
        inputs.results, inputs.labels, inputs.spiking_layers, duration, offset
    )
    hfb_annotations = build_annotation_table(
        polygrps_by_layer, png_label_metrics, neuron_information,
        experiment=experiment, model_type=model_type,
        trial_id=inputs.trial.name, checkpoint=checkpoint,
    )
    tables = {
        "hfb_annotations": hfb_annotations,
        "png_label_metrics": png_label_metrics,
        "neuron_information": neuron_information,
    }
    return tables, inputs


def persist_tables(
    store: handler.ArtifactStore,
    tables: Mapping[str, pd.DataFrame],
    subdir: str = REUSE_SUBDIR,
    overwrite: bool = False,
) -> None:
    """Persists the annotation tables as parquet under the reuse sub-directory.

    Args:
        store: Artifact store positioned at the representative checkpoint.
        tables: A ``{name: DataFrame}`` mapping (keys = :data:`TABLE_NAMES`).
        subdir: Sub-directory beneath the checkpoint. Defaults to ``'reuse'``.
        overwrite: Overwrite existing artifacts. Defaults to ``False``.
    """
    for name in TABLE_NAMES:
        store.save_table(
            tables[name], subdir=subdir, base_name=name, overwrite=overwrite
        )
        logger.info(f"Persisted '{name}' ({len(tables[name])} rows)")


def load_tables(
    store: handler.ArtifactStore, subdir: str = REUSE_SUBDIR
) -> dict[str, pd.DataFrame]:
    """Loads the persisted annotation tables from the reuse sub-directory.

    Args:
        store: Artifact store positioned at the representative checkpoint.
        subdir: Sub-directory beneath the checkpoint. Defaults to ``'reuse'``.

    Returns:
        A ``{name: DataFrame}`` mapping (keys = :data:`TABLE_NAMES`).
    """
    return {
        name: store.load_table(subdir=subdir, base_name=name)
        for name in TABLE_NAMES
    }


def audit_tables(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarises per-layer structure for the acceptance checks.

    Args:
        tables: A ``{name: DataFrame}`` mapping (keys = :data:`TABLE_NAMES`).

    Returns:
        A per-layer audit table with the significant-PNG count, the number of
        distinct high-level (index-1) neurons, the mean HFBs per high-level
        neuron, and whether ``neuron_information`` covers that layer.
    """
    ann = tables["hfb_annotations"]
    neuron_information = tables["neuron_information"]
    info_layers = set(neuron_information["layer"].unique())
    grouped = ann.groupby("layer")
    n_pngs = grouped.size()
    n_high = grouped["h_id"].nunique()
    audit = pd.DataFrame({
        "n_pngs": n_pngs,
        "n_high_neurons": n_high,
        "mean_hfbs_per_high": (n_pngs / n_high),
        "neuron_info_present": [layer in info_layers for layer in n_pngs.index],
    })
    audit.index.name = "layer"
    return audit
