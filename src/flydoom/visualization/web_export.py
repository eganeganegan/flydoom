"""Export compact binary sessions for the FlyDoom WebGL viewer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from flydoom.data.schema import ConnectomeGraph
from flydoom.visualization.layout import load_swc
from flydoom.visualization.trace import EpisodeTrace


def _write_array(directory: Path, name: str, values: np.ndarray, dtype: str) -> str:
    filename = f"{name}.bin"
    directory.joinpath(filename).write_bytes(np.asarray(values, dtype=dtype).tobytes(order="C"))
    return filename


def _aligned_neurons(trace: EpisodeTrace, graph: ConnectomeGraph):
    neurons = graph.neurons.copy()
    neurons["trace_body_id"] = neurons["body_id"].astype(str)
    neurons = neurons.set_index("trace_body_id")
    body_ids = trace.body_ids.astype(str)
    missing = sorted(set(body_ids).difference(neurons.index))
    if missing:
        raise ValueError(f"Trace contains body IDs absent from the graph: {missing[:5]}")
    return neurons.loc[body_ids]


def _graph_edges(trace: EpisodeTrace, graph: ConnectomeGraph) -> tuple[np.ndarray, np.ndarray]:
    lookup = {body_id: index for index, body_id in enumerate(trace.body_ids.astype(str))}
    pre = graph.edges["pre_body_id"].astype(str).map(lookup)
    post = graph.edges["post_body_id"].astype(str).map(lookup)
    keep = pre.notna() & post.notna()
    indices = np.column_stack(
        (pre.loc[keep].to_numpy(dtype=np.uint32), post.loc[keep].to_numpy(dtype=np.uint32))
    )
    weights = np.log1p(graph.edges.loc[keep, "synapse_count"].to_numpy(dtype=np.float32))
    scale = max(float(np.percentile(weights, 99)), 1e-6) if len(weights) else 1.0
    return indices, np.clip(weights / scale, 0.0, 1.0)


def _skeleton_geometry(
    body_ids: np.ndarray,
    positions: np.ndarray,
    skeleton_dir: Path | None,
    max_segments: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if skeleton_dir is None:
        return positions, np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.uint32)
    segment_parts: list[np.ndarray] = []
    owner_parts: list[np.ndarray] = []
    centroids: dict[int, np.ndarray] = {}
    remaining = max_segments
    for owner, body_id in enumerate(body_ids.astype(str)):
        path = skeleton_dir / f"{body_id}.swc"
        if remaining <= 0:
            break
        if not path.is_file():
            continue
        points, lines = load_swc(path)
        if not len(lines):
            continue
        lines = lines[:remaining]
        segments = points[lines].reshape(-1, 3)
        segment_parts.append(segments)
        owner_parts.append(np.full(len(segments), owner, dtype=np.uint32))
        centroids[owner] = points.mean(axis=0)
        remaining -= len(lines)
    if not segment_parts:
        return positions, np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.uint32)
    segments = np.concatenate(segment_parts).astype(np.float32)
    center = np.median(segments, axis=0, keepdims=True)
    centered = segments - center
    scale = max(float(np.percentile(np.linalg.norm(centered, axis=1), 98)), 1e-6)
    segments = centered / scale
    display_positions = positions.copy()
    for owner, centroid in centroids.items():
        display_positions[owner] = (centroid - center[0]) / scale
    return display_positions, segments, np.concatenate(owner_parts)


def export_web_session(
    trace: EpisodeTrace,
    graph: ConnectomeGraph,
    output: str | Path,
    *,
    skeleton_dir: str | Path | None = None,
    max_skeleton_segments: int = 2_000_000,
    title: str = "FlyDoom neural activity",
) -> Path:
    """Write a browser-streamable session directory and return its manifest path."""
    if max_skeleton_segments < 0:
        raise ValueError("max_skeleton_segments must be non-negative")
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    aligned = _aligned_neurons(trace, graph)
    positions, skeleton_segments, skeleton_owners = _skeleton_geometry(
        trace.body_ids,
        np.asarray(trace.positions, dtype=np.float32),
        Path(skeleton_dir) if skeleton_dir else None,
        max_skeleton_segments,
    )
    edges, edge_strength = _graph_edges(trace, graph)
    activity = np.asarray(trace.activity, dtype=np.float32)
    activity_clip = max(float(np.percentile(np.abs(activity), 99.5)), 1e-6)
    node_kind = np.zeros(len(aligned), dtype=np.uint8)
    node_kind[aligned["is_sensory"].fillna(False).to_numpy(dtype=bool)] = 1
    node_kind[aligned["is_descending"].fillna(False).to_numpy(dtype=bool)] = 2
    files = {
        "positions": _write_array(directory, "positions", positions, "<f4"),
        "activity": _write_array(directory, "activity", activity, "<f4"),
        "frames": _write_array(directory, "frames", trace.frames, "u1"),
        "node_kind": _write_array(directory, "node-kind", node_kind, "u1"),
        "edges": _write_array(directory, "edges", edges, "<u4"),
        "edge_strength": _write_array(directory, "edge-strength", edge_strength, "<f4"),
    }
    if len(skeleton_segments):
        files["skeleton_segments"] = _write_array(
            directory, "skeleton-segments", skeleton_segments, "<f4"
        )
        files["skeleton_owners"] = _write_array(
            directory, "skeleton-owners", skeleton_owners, "<u4"
        )
    frame_shape = list(trace.frames.shape[1:])
    action_names = [] if trace.action_names is None else trace.action_names.astype(str).tolist()
    manifest = {
        "version": 1,
        "title": title,
        "fps": trace.fps,
        "frame_count": len(trace.frames),
        "frame_shape": frame_shape,
        "node_count": len(trace.body_ids),
        "edge_count": len(edges),
        "skeleton_segment_count": len(skeleton_segments) // 2,
        "total_synapses": float(graph.edges["synapse_count"].sum()),
        "activity_clip": activity_clip,
        "files": files,
        "actions": trace.actions.astype(int).tolist(),
        "rewards": trace.rewards.astype(float).tolist(),
        "action_names": action_names,
        "body_ids": trace.body_ids.astype(str).tolist(),
        "types": trace.types.astype(str).tolist(),
        "regions": trace.regions.astype(str).tolist(),
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, separators=(",", ":")))
    return manifest_path
