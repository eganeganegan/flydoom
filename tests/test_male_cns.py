from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pandas as pd

from flydoom.data.male_cns import REQUIRED_FILES, load_male_cns_bulk, load_male_cns_neuprint


def test_official_bulk_aliases_are_canonicalized(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "body": [10, 20, 30],
            "type": ["visual sensory", "interneuron", "DNp01"],
            "class": ["sensory", "central", "descending neuron"],
            "side": ["L", "R", "R"],
        }
    ).to_feather(tmp_path / REQUIRED_FILES[0])
    pd.DataFrame({"body": [10, 20, 30], "predictedNt": ["ACh", "GABA", "ACh"]}).to_feather(
        tmp_path / REQUIRED_FILES[1]
    )
    pd.DataFrame({"body": [10, 20, 30], "pre": [5, 6, 7], "post": [2, 3, 4]}).to_feather(
        tmp_path / REQUIRED_FILES[2]
    )
    pd.DataFrame({"body_pre": [10, 20], "body_post": [20, 30], "weight": [5, 4]}).to_feather(
        tmp_path / REQUIRED_FILES[3]
    )
    graph = load_male_cns_bulk(tmp_path)
    assert graph.neurons.columns.isin(["body_id", "type", "class"]).sum() == 3
    assert graph.edges[["pre_body_id", "post_body_id"]].iloc[0].tolist() == [10, 20]
    assert graph.neurons["is_sensory"].tolist() == [True, False, False]
    assert graph.neurons["is_descending"].tolist() == [False, False, True]


def test_neuprint_mode_performs_bounded_k_hop_expansion(monkeypatch) -> None:
    module = ModuleType("neuprint")
    module.Client = lambda *args, **kwargs: object()
    module.NeuronCriteria = lambda **kwargs: kwargs
    module.fetch_neurons = lambda *args, **kwargs: pd.DataFrame(
        {"bodyId": [1], "type": ["R1-6"], "class": ["sensory"], "predictedNt": ["ACh"]}
    )

    def adjacency(sources, *_args, **_kwargs):
        source = list(sources)[0]
        target = source + 1
        neurons = pd.DataFrame(
            {
                "bodyId": [source, target],
                "type": [f"n{source}", f"n{target}"],
                "class": ["central", "descending" if target == 3 else "central"],
                "predictedNt": ["ACh", "GABA"],
            }
        )
        edges = pd.DataFrame({"bodyId_pre": [source], "bodyId_post": [target], "weight": [5]})
        return neurons, edges

    module.fetch_adjacencies = adjacency
    monkeypatch.setitem(sys.modules, "neuprint", module)
    graph = load_male_cns_neuprint(types=["R1-6"], num_hops=2, max_neurons=3, token="test")
    assert graph.neurons["body_id"].tolist() == [1, 2, 3]
    assert graph.edges[["pre_body_id", "post_body_id"]].values.tolist() == [[1, 2], [2, 3]]
    assert graph.neurons["is_sensory"].tolist() == [True, False, False]
