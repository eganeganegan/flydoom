from __future__ import annotations

import json

import numpy as np
import pandas as pd

from flydoom.data.schema import ConnectomeGraph
from flydoom.visualization.trace import EpisodeTrace
from flydoom.visualization.web_export import export_web_session


def test_web_session_exports_compact_aligned_binary_data(tmp_path) -> None:
    graph = ConnectomeGraph(
        pd.DataFrame(
            {
                "body_id": [10, 20, 30],
                "type": ["LC4", "PVLP024", "DNp04"],
                "is_sensory": [True, False, False],
                "is_descending": [False, False, True],
            }
        ),
        pd.DataFrame(
            {
                "pre_body_id": [10, 20],
                "post_body_id": [20, 30],
                "synapse_count": [12, 8],
            }
        ),
    )
    trace = EpisodeTrace(
        frames=np.zeros((2, 4, 5, 3), dtype=np.uint8),
        activity=np.asarray([[0.0, 0.5, 1.0], [0.2, 0.7, 0.1]], dtype=np.float32),
        actions=np.asarray([0, 3]),
        rewards=np.asarray([-1.0, 101.0]),
        positions=np.zeros((3, 3), dtype=np.float32),
        body_ids=np.asarray(["10", "20", "30"]),
        regions=np.asarray(["LO", "PVLP", "GNG"]),
        types=np.asarray(["LC4", "PVLP024", "DNp04"]),
        action_names=np.asarray(["no_op", "move_left", "move_right", "shoot"]),
    )

    manifest_path = export_web_session(trace, graph, tmp_path / "session")
    manifest = json.loads(manifest_path.read_text())

    assert manifest["node_count"] == 3
    assert manifest["edge_count"] == 2
    assert manifest["action_names"][-1] == "shoot"
    assert manifest_path.with_name(manifest["files"]["activity"]).stat().st_size == 2 * 3 * 4
    kinds = np.fromfile(manifest_path.with_name(manifest["files"]["node_kind"]), dtype=np.uint8)
    assert kinds.tolist() == [1, 0, 2]
