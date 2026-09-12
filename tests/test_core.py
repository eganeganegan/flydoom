from __future__ import annotations

import pandas as pd
import torch

from flydoom.data.loaders import generate_mock_connectome
from flydoom.data.schema import ConnectomeGraph
from flydoom.data.subgraph import SubgraphSpec, extract_subgraph
from flydoom.models.connectome_network import ConnectomeRateNetwork


def tiny_graph() -> ConnectomeGraph:
    neurons = pd.DataFrame(
        {
            "neuron_id": [90, 12, 44, 3],
            "super_class": ["visual", "central", "central", "descending"],
            "brain_region": ["visual", "central", "central", "descending"],
            "is_sensory": [True, False, False, False],
            "is_descending": [False, False, False, True],
        }
    )
    edges = pd.DataFrame(
        {
            "pre_neuron_id": [90, 12, 44],
            "post_neuron_id": [12, 44, 3],
            "synapse_count": [5, 4, 3],
        }
    )
    return ConnectomeGraph(neurons, edges)


def test_ids_map_to_contiguous_sparse_indices() -> None:
    graph = tiny_graph()
    edge_index, weights = graph.sparse_tensors()
    assert graph.id_to_index == {90: 0, 12: 1, 44: 2, 3: 3}
    assert edge_index.tolist() == [[0, 1, 2], [1, 2, 3]]
    assert weights.tolist() == [5.0, 4.0, 3.0]


def test_subgraph_extracts_source_to_target_paths() -> None:
    selected = extract_subgraph(
        tiny_graph(),
        SubgraphSpec(source_class="visual", target_class="descending", num_hops=4),
    )
    assert selected.node_count == 4
    assert selected.edge_count == 3


def test_mock_connectome_uses_canonical_schema() -> None:
    graph = generate_mock_connectome(100, seed=7)
    assert graph.node_count == 100
    assert graph.neurons["is_sensory"].sum() == 20
    assert graph.edges["synapse_count"].min() >= 1


def test_sparse_rate_network_forward_and_gradients() -> None:
    graph = tiny_graph()
    edge_index, weights = graph.sparse_tensors()
    model = ConnectomeRateNetwork(graph.node_count, edge_index, weights)
    external = torch.zeros(2, graph.node_count)
    external[:, 0] = 1.0
    output = model(external, steps=4)
    assert output.shape == (2, 4)
    assert torch.all(output[:, -1] > 0)
    output[:, -1].sum().backward()
    assert model.edge_weight.grad is not None
    assert torch.isfinite(model.edge_weight.grad).all()
