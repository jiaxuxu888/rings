import pytest
import numpy as np
import torch
import networkx as nx
from unittest.mock import MagicMock, patch

from torch_geometric.data import Data
from rings.complementarity.functor import ComplementarityFunctor


class TestComplementarityFunctor:
    @pytest.fixture
    def mock_feature_metric(self):
        return MagicMock(return_value=np.array([[0, 1], [1, 0]]))

    @pytest.fixture
    def mock_graph_metric(self):
        return MagicMock(return_value=np.array([[0, 2], [2, 0]]))

    @pytest.fixture
    def mock_comparator(self):
        # Create a class with the required methods to mock the comparator
        class MockComparatorClass:
            def __init__(self, n_jobs=None, **kwargs):
                self.n_jobs = n_jobs
                self.kwargs = kwargs

            def __call__(self, D_X, D_G):
                return {"complementarity": 0.5}

            @property
            def invalid_data(self):
                return {"complementarity": np.nan}

        # Return the mock class instead of trying to mock a callable
        return MagicMock(side_effect=MockComparatorClass)

    @pytest.fixture
    def functor(self, mock_feature_metric, mock_graph_metric, mock_comparator):
        return ComplementarityFunctor(
            feature_metric=mock_feature_metric,
            graph_metric=mock_graph_metric,
            comparator=mock_comparator,
            n_jobs=1,
        )

    def test_init(
        self, mock_feature_metric, mock_graph_metric, mock_comparator
    ):
        """Test initialization of ComplementarityFunctor."""
        functor = ComplementarityFunctor(
            feature_metric=mock_feature_metric,
            graph_metric=mock_graph_metric,
            comparator=mock_comparator,
            n_jobs=2,
            custom_param="test",
        )

        assert functor.feature_metric == mock_feature_metric
        assert functor.graph_metric == mock_graph_metric
        assert functor.n_jobs == 2
        assert functor.use_edge_information is False
        assert functor.kwargs == {"custom_param": "test"}
        mock_comparator.assert_called_once_with(n_jobs=2, custom_param="test")

    def test_init_with_edge_information(
        self, mock_feature_metric, mock_graph_metric, mock_comparator
    ):
        """Test initialization with edge information enabled."""
        functor = ComplementarityFunctor(
            feature_metric=mock_feature_metric,
            graph_metric=mock_graph_metric,
            comparator=mock_comparator,
            n_jobs=1,
        )
        functor.use_edge_information = True

        assert functor.use_edge_information is True
        assert "weight" in functor.kwargs
        assert functor.kwargs["weight"] == "edge_attr"

    @patch("rings.complementarity.functor.to_networkx")
    @patch("rings.complementarity.functor.lift_attributes")
    @patch("rings.complementarity.functor.lift_graph")
    @patch("rings.complementarity.functor.maybe_normalize_diameter")
    def test_forward_single_graph(
        self,
        mock_normalize,
        mock_lift_graph,
        mock_lift_attrs,
        mock_to_networkx,
        functor,
        mock_comparator,
    ):
        """Test forward method with a single graph."""
        # Setup mocks
        mock_graph = MagicMock()
        mock_to_networkx.return_value = mock_graph
        mock_graph.number_of_edges.return_value = 1
        mock_graph.number_of_nodes.return_value = 2
        nx.is_connected = MagicMock(return_value=True)
        nx.get_node_attributes = MagicMock(return_value={0: [1, 2], 1: [3, 4]})

        # Mock lift functions
        mock_lift_graph.return_value = np.array([[0, 1], [1, 0]])
        mock_lift_attrs.return_value = np.array([[0, 2], [2, 0]])
        mock_normalize.side_effect = lambda x: x

        # Create test data
        x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        test_data = Data(x=x, edge_index=edge_index)

        # Run forward
        result = functor.forward([test_data])

        # Check results
        assert "complementarity" in result
        assert torch.is_tensor(result["complementarity"])
        assert result["complementarity"].shape == (1,)
        assert float(result["complementarity"][0]) == 0.5

        # Verify the workflow
        mock_to_networkx.assert_called_once()
        mock_lift_graph.assert_called_once()
        mock_lift_attrs.assert_called_once()
        mock_normalize.call_count == 2

    @patch("rings.complementarity.functor.to_networkx")
    @patch("rings.complementarity.functor.lift_attributes")
    @patch("rings.complementarity.functor.lift_graph")
    @patch("rings.complementarity.functor.maybe_normalize_diameter")
    def test_forward_batch(
        self,
        mock_normalize,
        mock_lift_graph,
        mock_lift_attrs,
        mock_to_networkx,
        functor,
    ):
        """Test forward method with a batch of graphs."""
        # Setup mocks
        mock_graph = MagicMock()
        mock_to_networkx.return_value = mock_graph
        mock_graph.number_of_edges.return_value = 1
        mock_graph.number_of_nodes.return_value = 2
        nx.is_connected = MagicMock(return_value=True)
        nx.get_node_attributes = MagicMock(return_value={0: [1, 2], 1: [3, 4]})

        # Mock lift functions
        mock_lift_graph.return_value = np.array([[0, 1], [1, 0]])
        mock_lift_attrs.return_value = np.array([[0, 2], [2, 0]])
        mock_normalize.side_effect = lambda x: x

        # Create test data
        x1 = torch.tensor([[1, 2], [3, 4]], dtype=torch.float)
        edge_index1 = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        test_data1 = Data(x=x1, edge_index=edge_index1)

        x2 = torch.tensor([[5, 6], [7, 8]], dtype=torch.float)
        edge_index2 = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        test_data2 = Data(x=x2, edge_index=edge_index2)

        # Run forward
        result = functor.forward([test_data1, test_data2])

        # Check results
        assert "complementarity" in result
        assert torch.is_tensor(result["complementarity"])
        assert result["complementarity"].shape == (2,)
        assert float(result["complementarity"][0]) == 0.5
        assert float(result["complementarity"][1]) == 0.5

    @patch("rings.complementarity.functor.to_networkx")
    def test_forward_with_edge_attr(self, mock_to_networkx, functor):
        """Test forward with edge attributes."""
        # Setup mocks
        mock_graph = MagicMock()
        mock_to_networkx.return_value = mock_graph
        mock_graph.number_of_edges.return_value = 1
        mock_graph.number_of_nodes.return_value = 2
        nx.is_connected = MagicMock(return_value=True)
        nx.get_node_attributes = MagicMock(return_value={0: [1, 2], 1: [3, 4]})
        nx.get_edge_attributes = MagicMock(return_value={(0, 1): [0.5, 0.5]})
        nx.set_edge_attributes = MagicMock()

        # Enable edge information
        functor.use_edge_information = True

        # Create test data with edge attributes
        x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        edge_attr = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float)
        test_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        # Apply patching for internal methods
        with patch.object(
            functor, "_complementarity", return_value={"complementarity": 0.5}
        ):
            # Run forward
            result = functor.forward([test_data])

            # Check edge attribute processing
            mock_to_networkx.assert_called_with(
                test_data,
                to_undirected=True,
                node_attrs=["x"],
                edge_attrs=["edge_attr"],
            )

            # When use_edge_information is True and edge_attr is present
            # nx.get_edge_attributes and nx.set_edge_attributes should be called
            assert nx.get_edge_attributes.called
            assert nx.set_edge_attributes.called

    @patch("networkx.is_connected")
    def test_complementarity_connected_graph(self, mock_is_connected, functor):
        """Test complementarity calculation with a connected graph."""
        # Setup test graph
        G = nx.Graph()
        G.add_nodes_from([0, 1])
        G.add_edge(0, 1)
        nx.set_node_attributes(G, {0: [1, 2], 1: [3, 4]}, "x")

        mock_is_connected.return_value = True

        # Apply patching
        with (
            patch(
                "rings.complementarity.functor.lift_graph",
                return_value=np.array([[0, 1], [1, 0]]),
            ) as mock_lift_graph,
            patch(
                "rings.complementarity.functor.lift_attributes",
                return_value=np.array([[0, 2], [2, 0]]),
            ) as mock_lift_attrs,
            patch(
                "rings.complementarity.functor.maybe_normalize_diameter",
                side_effect=lambda x: x,
            ) as mock_normalize,
        ):

            result = functor._complementarity(G)

            # Verify correct methods were called
            mock_is_connected.assert_called_once_with(G)
            assert mock_lift_graph.called
            assert mock_lift_attrs.called
            assert mock_normalize.call_count == 2

            # Check result
            assert "complementarity" in result
            assert result["complementarity"] == 0.5

    @patch("warnings.warn")
    def test_complementarity_empty_feature_matrix(self, mock_warn, functor):
        """Test complementarity calculation with empty feature matrix."""
        # Setup test graph with no node attributes
        G = nx.Graph()
        G.add_nodes_from([0, 1])
        G.add_edge(0, 1)
        nx.set_node_attributes(G, {}, "x")  # Empty node attributes

        with patch("networkx.get_node_attributes", return_value={}):
            result = functor._complementarity(G)

            # Check warning and result
            mock_warn.assert_called_once_with(
                "Feature matrix X is empty, skipping graph."
            )
            assert "complementarity" in result
            assert np.isnan(result["complementarity"])

    @patch("warnings.warn")
    def test_complementarity_empty_graph(self, mock_warn, functor):
        """Test complementarity calculation with empty graph."""
        # Setup empty graph
        G = nx.Graph()

        result = functor._complementarity(G)

        # Check warning and result
        mock_warn.assert_called_once_with("Graph G has no nodes, skipping it.")
        assert "complementarity" in result
        assert np.isnan(result["complementarity"])

    def test_complementarity_return_metric_spaces(self, functor):
        """Test complementarity calculation with return_metric_spaces=True."""
        # Setup test graph
        G = nx.Graph()
        G.add_nodes_from([0, 1])
        G.add_edge(0, 1)
        nx.set_node_attributes(G, {0: [1, 2], 1: [3, 4]}, "x")

        # Apply patching
        with (
            patch("networkx.is_connected", return_value=True),
            patch(
                "rings.complementarity.functor.lift_graph",
                return_value=np.array([[0, 1], [1, 0]]),
            ) as mock_lift_graph,
            patch(
                "rings.complementarity.functor.lift_attributes",
                return_value=np.array([[0, 2], [2, 0]]),
            ) as mock_lift_attrs,
            patch(
                "rings.complementarity.functor.maybe_normalize_diameter",
                side_effect=lambda x: x,
            ) as mock_normalize,
        ):

            result = functor._complementarity(G, return_metric_spaces=True)

            # Check result contains metric spaces
            assert "complementarity" in result
            assert "D_X" in result
            assert "D_G" in result
            assert len(result["D_X"]) == 1
            assert len(result["D_G"]) == 1
