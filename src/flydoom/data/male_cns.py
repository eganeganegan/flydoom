"""Official HHMI Janelia Male CNS v1.0 bulk and neuPrint adapters."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from flydoom.data.loaders import read_table
from flydoom.data.schema import ConnectomeGraph

NEUPRINT_SERVER = "https://neuprint.janelia.org"
NEUPRINT_DATASET = "male-cns:v1.0"
BULK_ROOT = "gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/"
SKELETON_ROOT = "gs://flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-swc/"
REQUIRED_FILES = (
    "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "body-neurotransmitters-male-cns-v1.0.feather",
    "body-stats-male-cns-v1.0-minconf-0.5.feather",
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
)
OPTIONAL_FILES = (
    "syn-points-male-cns-v1.0-minconf-0.5.feather",
    "syn-partners-male-cns-v1.0-minconf-0.5.feather",
    "tbar-neurotransmitters-male-cns-v1.0.feather",
)


def _first(frame: pd.DataFrame, names: Iterable[str], *, required: bool = False) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    if required:
        raise ValueError(f"None of the required source columns were found: {list(names)}")
    return None


def _canonical_neurons(
    annotations: pd.DataFrame,
    neurotransmitters: pd.DataFrame | None = None,
    stats: pd.DataFrame | None = None,
) -> pd.DataFrame:
    body = _first(annotations, ("body_id", "bodyId", "body"), required=True)
    assert body is not None
    candidates = {
        "type": ("type", "cell_type", "systematicType"),
        "class": ("class", "superclass", "super_class"),
        "side": ("side", "somaSide", "rootSide", "hemisphere"),
        "region": ("region", "somaNeuromere", "brain_region"),
    }
    mapping = {body: "body_id"}
    for target, names in candidates.items():
        if (source := _first(annotations, names)) is not None:
            mapping[source] = target
    neurons = annotations.rename(columns=mapping)[list(dict.fromkeys(mapping.values()))].copy()
    if (location := _first(annotations, ("somaLocation", "soma_location", "position"))) is not None:

        def coordinate(value, index: int) -> float:
            if isinstance(value, dict):
                return float(value.get(("x", "y", "z")[index], np.nan))
            if isinstance(value, (list, tuple, np.ndarray)) and len(value) >= 3:
                return float(value[index])
            return np.nan

        for index, axis in enumerate(("x", "y", "z")):
            neurons[axis] = (
                annotations[location].map(lambda value, i=index: coordinate(value, i)).to_numpy()
            )
    additions = (
        (
            neurotransmitters,
            ("predictedNt", "consensusNt", "celltypePredictedNt", "neurotransmitter"),
            "neurotransmitter",
        ),
        (stats, ("pre", "PreSyn", "pre_synapses"), "pre_synapses"),
        (stats, ("post", "PostSyn", "post_synapses"), "post_synapses"),
    )
    for extra, names, target in additions:
        if extra is None or (value := _first(extra, names)) is None:
            continue
        extra_body = _first(extra, ("body_id", "bodyId", "body"), required=True)
        addition = (
            extra[[extra_body, value]]
            .rename(columns={extra_body: "body_id", value: target})
            .drop_duplicates("body_id")
        )
        neurons = neurons.merge(addition, on="body_id", how="left")
    labels = (
        neurons.get("class", pd.Series("", index=neurons.index)).fillna("").astype(str)
        + " "
        + neurons.get("type", pd.Series("", index=neurons.index)).fillna("").astype(str)
    ).str.lower()
    neurons["is_sensory"] = labels.str.contains(r"sensory|visual|optic|photoreceptor")
    neurons["is_descending"] = labels.str.contains(r"descending|\bdn")
    neurons["is_motor"] = labels.str.contains(r"motor|motoneuron")
    return neurons


def _canonical_edges(edges: pd.DataFrame, neurons: pd.DataFrame) -> pd.DataFrame:
    pre = _first(edges, ("pre_body_id", "bodyId_pre", "body_pre", "pre"), required=True)
    post = _first(edges, ("post_body_id", "bodyId_post", "body_post", "post"), required=True)
    weight = _first(edges, ("synapse_count", "weight", "count"), required=True)
    region = _first(edges, ("region", "roi", "primary_post", "neuropil"))
    assert pre is not None and post is not None and weight is not None
    columns = [pre, post, weight] + ([region] if region else [])
    rename = {pre: "pre_body_id", post: "post_body_id", weight: "synapse_count"}
    if region:
        rename[region] = "region"
    result = edges[columns].rename(columns=rename).copy()
    transmitter = neurons.set_index("body_id").get("neurotransmitter")
    if transmitter is not None:
        result["neurotransmitter"] = result["pre_body_id"].map(transmitter).fillna("unknown")
    return result


def load_male_cns_bulk(directory: str | Path) -> ConnectomeGraph:
    """Load the four official v1.0 Feather tables into the canonical schema."""
    directory = Path(directory)
    missing = [name for name in REQUIRED_FILES if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Male CNS v1.0 files in {directory}: {missing}")
    annotations, nt, stats, weights = (read_table(directory / name) for name in REQUIRED_FILES)
    neurons = _canonical_neurons(annotations, nt, stats)
    edges = _canonical_edges(weights, neurons)
    annotated = set(neurons["body_id"])
    edges = edges.loc[
        edges["pre_body_id"].isin(annotated) & edges["post_body_id"].isin(annotated)
    ].reset_index(drop=True)
    connected = set(edges["pre_body_id"]) | set(edges["post_body_id"])
    neurons = neurons.loc[neurons["body_id"].isin(connected)].reset_index(drop=True)
    return ConnectomeGraph(neurons, edges)


def load_male_cns_neuprint(
    *,
    types: list[str] | None = None,
    classes: list[str] | None = None,
    rois: list[str] | None = None,
    body_ids: list[int] | None = None,
    min_weight: int = 3,
    max_neurons: int = 5_000,
    num_hops: int = 2,
    token: str | None = None,
) -> ConnectomeGraph:
    """Fetch a bounded targeted k-hop graph from the official neuPrint dataset."""
    try:
        from neuprint import Client, NeuronCriteria, fetch_adjacencies, fetch_neurons
    except ImportError as exc:
        raise RuntimeError("Install the data extra: pip install -e '.[data]'") from exc
    token = token or os.environ.get("NEUPRINT_TOKEN")
    if not token:
        raise RuntimeError("Set NEUPRINT_TOKEN to a token from neuprint.janelia.org")
    client = Client(NEUPRINT_SERVER, dataset=NEUPRINT_DATASET, token=token)
    criteria = NeuronCriteria(bodyId=body_ids, type=types, class_=classes, rois=rois, client=client)
    properties = [
        "bodyId",
        "type",
        "class",
        "somaSide",
        "somaNeuromere",
        "predictedNt",
        "pre",
        "post",
        "somaLocation",
    ]
    initial = fetch_neurons(criteria, omit_rois=True, returned_columns=properties, client=client)
    if initial.empty:
        requested = {
            "types": types,
            "classes": classes,
            "rois": rois,
            "body_ids": body_ids,
        }
        raise ValueError(f"The neuPrint criteria matched no Male CNS neurons: {requested}")
    initial = initial.head(max_neurons)
    source_ids = set(initial["bodyId"])
    seen = set(source_ids)
    frontier = set(source_ids)
    neuron_parts = [initial]
    connection_parts: list[pd.DataFrame] = []
    for _ in range(num_hops):
        if not frontier or len(seen) >= max_neurons:
            break
        neuron_batch, connections = fetch_adjacencies(
            list(frontier),
            None,
            min_total_weight=min_weight,
            omit_rois=True,
            properties=properties,
            client=client,
        )
        new_strength = (
            connections.loc[~connections["bodyId_post"].isin(seen)]
            .groupby("bodyId_post")["weight"]
            .sum()
            .sort_values(ascending=False)
        )
        new_ids = set(new_strength.head(max_neurons - len(seen)).index)
        accepted = seen | new_ids
        connection_parts.append(
            connections.loc[
                connections["bodyId_pre"].isin(seen) & connections["bodyId_post"].isin(accepted)
            ]
        )
        neuron_parts.append(neuron_batch.loc[neuron_batch["bodyId"].isin(accepted)])
        seen.update(new_ids)
        frontier = new_ids
    neuron_info = pd.concat(neuron_parts, ignore_index=True).drop_duplicates("bodyId")
    neuron_info = neuron_info.loc[neuron_info["bodyId"].isin(seen)].reset_index(drop=True)
    if connection_parts:
        connections = pd.concat(connection_parts, ignore_index=True)
        connections = connections.groupby(["bodyId_pre", "bodyId_post"], as_index=False)[
            "weight"
        ].sum()
    else:
        connections = pd.DataFrame(columns=["bodyId_pre", "bodyId_post", "weight"])
    neurons = _canonical_neurons(neuron_info)
    neurons.loc[neurons["body_id"].isin(source_ids), "is_sensory"] = True
    if "somaLocation" in neuron_info:
        coordinates = neuron_info["somaLocation"].apply(
            lambda value: value if isinstance(value, (list, tuple, np.ndarray)) else [np.nan] * 3
        )
        for index, axis in enumerate(("x", "y", "z")):
            neurons[axis] = coordinates.map(lambda value, i=index: value[i]).to_numpy()
    edges = _canonical_edges(connections, neurons)
    return ConnectomeGraph(neurons.reset_index(drop=True), edges.reset_index(drop=True))
