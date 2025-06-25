import pytest
import torch
from torch_geometric.data import Data

from rings.utils import Shuffle, is_connected


class TestShuffleTransform:
    def setup_method(self):
        # Create a simple graph for testing
        self.edge_index = torch.tensor(
            [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long
        )
        self.x = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float)
        self.data = Data(x=self.x, edge_index=self.edge_index, num_nodes=3)

    def test_initialization(self):
        # Test basic initialization
        transform = Shuffle(shuffle_edges=True, shuffle_features=True)
        assert transform.shuffle_edges is True
        assert transform.shuffle_features is True

        transform = Shuffle(shuffle_edges=False, shuffle_features=False)
        assert transform.shuffle_edges is False
        assert transform.shuffle_features is False

    def test_shuffle_features(self):
        # Test feature shuffling functionality
        transform = Shuffle(
            shuffle_edges=False,
            shuffle_features=True,
            generator=torch.Generator().manual_seed(42),
        )
        transformed_data = transform(self.data.clone())

        # Verify number of nodes and features remains the same
        assert transformed_data.num_nodes == self.data.num_nodes
        assert transformed_data.x.shape == self.data.x.shape

        # Verify that features were actually shuffled (different order)
        assert not torch.equal(transformed_data.x, self.data.x)

        # Verify that all original features are present (just in different order)
        original_features = set(tuple(x.tolist()) for x in self.data.x)
        shuffled_features = set(tuple(x.tolist()) for x in transformed_data.x)
        assert original_features == shuffled_features

    def test_shuffle_edges(self):
        # Test edge shuffling functionality
        transform = Shuffle(
            shuffle_edges=True,
            shuffle_features=False,
            generator=torch.Generator().manual_seed(42),
        )
        transformed_data = transform(self.data.clone())

        # Verify number of edges remains the same
        assert transformed_data.edge_index.shape == self.data.edge_index.shape

        # Verify that edges were actually shuffled
        assert not torch.equal(
            transformed_data.edge_index[1], self.data.edge_index[1]
        )

        # Source nodes should remain the same
        assert torch.equal(
            transformed_data.edge_index[0], self.data.edge_index[0]
        )

    def test_no_self_loops(self):
        # Test that no self-loops are created during edge shuffling
        transform = Shuffle(shuffle_edges=True, shuffle_features=False)
        transformed_data = transform(self.data.clone())

        # Check for self-loops
        source_nodes = transformed_data.edge_index[0]
        target_nodes = transformed_data.edge_index[1]

        # Ensure no self-loops exist
        assert not (source_nodes == target_nodes).any()

    def test_combined_transform(self):
        # Test combining both edge and feature shuffling
        transform = Shuffle(
            shuffle_edges=True,
            shuffle_features=True,
            generator=torch.Generator().manual_seed(42),
        )
        transformed_data = transform(self.data.clone())

        # Verify both edges and features were modified
        assert not torch.equal(
            transformed_data.edge_index, self.data.edge_index
        )
        assert not torch.equal(transformed_data.x, self.data.x)

    def test_no_transformation(self):
        # Test when no transformation is requested
        transform = Shuffle(shuffle_edges=False, shuffle_features=False)
        transformed_data = transform(self.data.clone())

        # Verify data remains unchanged
        assert torch.equal(transformed_data.edge_index, self.data.edge_index)
        assert torch.equal(transformed_data.x, self.data.x)

    def test_with_no_features(self):
        # Test behavior when data has no features
        edge_index = torch.tensor(
            [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long
        )
        data_no_features = Data(edge_index=edge_index, num_nodes=3)

        transform = Shuffle(shuffle_edges=True, shuffle_features=True)
        transformed_data = transform(data_no_features.clone())

        # Only edges should be shuffled, no error should be raised
        assert not torch.equal(
            transformed_data.edge_index, data_no_features.edge_index
        )


class TestIsConnected:
    def test_connected_graph(self):
        # Create a connected graph
        edge_index = torch.tensor(
            [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long
        )
        data = Data(edge_index=edge_index, num_nodes=3)

        assert is_connected(data) is True

    def test_disconnected_graph(self):
        # Create a disconnected graph (two separate components)
        edge_index = torch.tensor(
            [[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long
        )
        data = Data(edge_index=edge_index, num_nodes=4)

        assert is_connected(data) is False

    def test_single_node_graph(self):
        # Create a graph with a single node (no edges)
        data = Data(
            edge_index=torch.zeros((2, 0), dtype=torch.long), num_nodes=1
        )

        # A single node graph with no edges is connected by definition
        assert is_connected(data) is True

    def test_empty_graph(self):
        # Create an empty graph (no nodes)
        data = Data(
            edge_index=torch.zeros((2, 0), dtype=torch.long), num_nodes=0
        )

        # An empty graph is connected by convention
        # (NetworkX would raise NetworkXPointlessConcept, but we handle this case)
        assert is_connected(data) is True
