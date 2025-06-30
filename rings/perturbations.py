"""This file defines the Mode Perturbations, i.e Graph Transformations, of the RINGS Framework."""

import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import dense_to_sparse

from rings.utils import Shuffle, is_connected


class Original(BaseTransform):
    """
    A placeholder transform that returns the input node features & graph data without modifications.
    """

    def __call__(self, data):
        """
        Return the original, unmodified data object.

        Args:
            data (torch_geometric.data.Data): Input graph data object.

        Returns:
            torch_geometric.data.Data: Unmodified graph data object.
        """
        return data


#  ╭──────────────────────────────────────────────────────────╮
#  │ Perturb Node Features                                    |
#  ╰──────────────────────────────────────────────────────────╯


class EmptyFeatures(BaseTransform):
    """
    A transform that assigns identical features (zero vector) to each node.
    """

    def __call__(self, data):
        """
        Assign zero vectors as features to each node.

        Args:
            data (torch_geometric.data.Data): Input graph data object.

        Returns:
            torch_geometric.data.Data: Transformed graph data object with zero node features.
        """
        num_nodes = data.num_nodes
        data.x = torch.zeros((num_nodes, 1), dtype=torch.float)
        return data


class CompleteFeatures(BaseTransform):
    """
    A transform that assigns unique node IDs as features to each node.
    Each node is represented by a padded one-hot encoded vector.
    """

    def __init__(self, max_nodes):
        """
        Initialize the CompleteFeatures transform.

        Args:
            max_nodes (int): Maximum number of nodes for one-hot feature encoding.
        """
        self.max_nodes = max_nodes

    def __call__(self, data):
        """
        Apply the transform to assign one-hot encoded node IDs as features.

        Args:
            data (torch_geometric.data.Data): Input graph data object.

        Returns:
            torch_geometric.data.Data: Transformed graph data object with one-hot node features.
        """
        num_nodes = data.num_nodes  # Total number of nodes in the graph

        # Create a one-hot encoding for each node
        one_hot = torch.eye(self.max_nodes, dtype=torch.float)

        # Select the rows corresponding to the actual nodes
        data.x = one_hot[:num_nodes]

        return data


class RandomFeatures(BaseTransform):
    """
    A transform that randomizes existing node features either by sampling new features from a standard normal distribution, or shuffling node features between nodes.
    """

    def __init__(self, shuffle=False, fixed_dimension=None):
        """
        Initialize the RandomFeatures transform.

        Args:
            shuffle (bool): If True, shuffle existing node features. If False (default), sample new features from a standard normal distribution.

            fixed_dimension (int, optional): Fixed dimension for new random features. If None, use the original feature dimension.
        """
        self.dimension = fixed_dimension
        self.shuffle = shuffle

        current_seed = torch.initial_seed()
        self.generator = torch.Generator().manual_seed(current_seed)

    def __call__(self, data):
        """
        Apply the transform to assign or shuffle node features.

        Args:
            data (torch_geometric.data.Data): Input graph data object.

        Returns:
            torch_geometric.data.Data: Transformed graph data object with modified features.
        """
        if self.shuffle:
            return self._shuffle(data)
        else:
            return self._randomize_features(data, dimension=self.dimension)

    def _shuffle(self, data):
        """
        Shuffle node features among nodes in the graph.

        Args:
            data: Graph data object.
            generator: Random generator for reproducibility.

        Returns:
            data: Graph data object with shuffled features.
        """
        return Shuffle(shuffle_features=True, generator=self.generator)(data)

    def _randomize_features(self, data, dimension=None):
        """
        Sample random node features from a standard normal distribution using `torch.randn`.

        Args:
            data: Graph data object.
            dimension: Fixed feature dimension (if None, use original dimension).
            generator: Random generator for reproducibility.

        Returns:
            data: Graph data object with randomized features.
        """
        if hasattr(data, "x") and data.x is not None:
            num_nodes = data.x.size(0)
            dim = data.x.size(1) if dimension is None else dimension
            data.x = torch.randn(num_nodes, dim, generator=self.generator)
        return data


#  ╭──────────────────────────────────────────────────────────╮
#  │ Perturb Graph Structure                                  |
#  ╰──────────────────────────────────────────────────────────╯


class EmptyGraph(BaseTransform):
    """
    A transform that removes all edges from the graph, creating an empty graph.
    """

    def __call__(self, data):
        """
        Remove all edges from the graph.

        Args:
            data (torch_geometric.data.Data): Input graph data object.

        Returns:
            torch_geometric.data.Data: Transformed graph data object with no edges.
        """
        data.edge_index = torch.tensor(
            [[], []], dtype=torch.long
        )  # Remove edges by setting edge_index to an empty tensor
        data.edge_attr = None  # Optionally remove edge attributes if they exist

        return data


class CompleteGraph(BaseTransform):
    """
    A transform that replaces the existing graph structure with a complete graph.
    """

    def __call__(self, data):
        """
        Convert the graph into a complete graph.

        Args:
            data (torch_geometric.data.Data): Input graph data object.

        Returns:
            torch_geometric.data.Data: Transformed graph data object as a complete graph.
        """
        num_nodes = data.num_nodes  # Number of nodes in the graph

        # Create a complete graph adjacency matrix (all nodes connected to each other)
        adj = torch.ones((num_nodes, num_nodes), dtype=torch.long) - torch.eye(
            num_nodes, dtype=torch.long
        )

        # Convert dense adjacency matrix to edge_index format
        edge_index, _ = dense_to_sparse(adj)

        # Assign the complete graph to data.edge_index
        data.edge_index = edge_index

        # Remove any existing edge attributes
        data.edge_attr = None

        return data


class RandomGraph(BaseTransform):
    """
    A transform that replaces the existing graph structure with a random graph. The graph is generated either using an Erdos-Renyi perturbation or by randomly shuffling the current edges.
    """

    def __init__(self, p=None, shuffle=False):
        """
        Initialize the RandomGraph transform.

        Args:
            p (float): Probability of an edge existing between any two nodes. Only used if shuffle=False (default).
            shuffle (bool): If True, shuffle the existing graph structure. If False (default), assign edges with probability p (i.e. impose Erdos-Renyi graph structure).
        """
        self.p = p
        self.shuffle = shuffle
        self.generator = torch.Generator().manual_seed(torch.initial_seed())

    def __call__(self, data):
        """
        Replace the graph structure with a random graph.

        Args:
            data (torch_geometric.data.Data): Input graph data object.

        Returns:
            torch_geometric.data.Data: Transformed graph data object with random graph structure.
        """
        if self.shuffle:
            return self._shuffle(data)

        else:
            return self._randomize_graph(data)

    def _shuffle(self, data):
        """
        Shuffle edges in the graph.

        Args:
            data (torch_geometric.data.Data): Input graph data object.

        Returns:
            torch_geometric.data.Data: Transformed graph with shuffled edges.
        """
        return Shuffle(shuffle_edges=True, generator=self.generator)(data)

    def _randomize_graph(self, data):
        """
        Generate a random Erdos-Renyi graph structure.

        Args:
            data (torch_geometric.data.Data): Input graph data object.

        Returns:
            torch_geometric.data.Data: Transformed graph with random structure.
        """
        num_nodes = data.num_nodes
        if num_nodes is None:
            raise ValueError("Data object must have `num_nodes` defined.")

        num_edges = self._set_num_edges(data, num_nodes, self.p)

        row = torch.randint(
            0, num_nodes, (num_edges,), device="cpu", generator=self.generator
        )
        col = torch.randint(
            0, num_nodes, (num_edges,), device="cpu", generator=self.generator
        )

        # Remove self-loops
        mask = row != col
        row, col = row[mask], col[mask]

        # Create sparse adjacency matrix
        edge_index = torch.stack([row, col], dim=0).to(data.edge_index.device)
        data.edge_index = edge_index
        data.edge_attr = None

        return data

    @staticmethod
    def _set_num_edges(data, N, p):
        """
        Compute the number of edges for the random graph.

        Args:
            data (torch_geometric.data.Data): Input graph data object.
            N (int): Number of nodes in the graph.
            p (float): Edge probability.

        Returns:
            int: Number of edges.
        """
        num_edges = data.edge_index.size(1) if p is None else int(p * N * (N - 1) / 2)
        return num_edges


class RandomConnectedGraph(BaseTransform):
    """
    A transform that replaces the existing graph structure with a random graph. The graph is generated either using an Erdos-Renyi perturbation or by randomly shuffling the current edges. The resulting graph is guaranteed to be connected.
    """

    def __init__(self, p=None, shuffle=False):
        """
        Initialize the RandomConnectedGraph transform.

        Args:
            p (float): Edge probability for the random connected graph.
            shuffle (bool): If True, shuffle the existing graph structure. If False (default), assign edges with probability p (i.e. impose Erdos-Renyi graph structure).
        """
        self.p = p
        self.shuffle = shuffle
        self.generator = torch.Generator().manual_seed(torch.initial_seed())

    def __call__(self, data):
        """
        Generate a random connected graph structure.

        Args:
            data (torch_geometric.data.Data): Input graph data object.

        Returns:
            torch_geometric.data.Data: Transformed graph with connected structure.
        """
        transform = self._shuffle if self.shuffle else self._randomize_connected_graph
        data = transform(data)
        while not is_connected(data):
            data = transform(data)
        return data

    def _shuffle(self, data):
        """
        Shuffle edges in the graph.

        Args:
            data (torch_geometric.data.Data): Input graph data object.

        Returns:
            torch_geometric.data.Data: Graph with shuffled edges.
        """
        return Shuffle(shuffle_edges=True, generator=self.generator)(data)

    def _randomize_connected_graph(self, data):
        """
        Generate a random connected graph structure.

        Args:
            data (torch_geometric.data.Data): Input graph data object.

        Returns:
            torch_geometric.data.Data: Transformed graph with connected structure.
        """
        num_nodes = data.num_nodes
        if num_nodes is None:
            raise ValueError("Data object must have `num_nodes` defined.")

        # Add additional random edges based on probability p
        edges = self._random_spanning_tree(num_nodes)
        edge_set = set(edges)

        num_additional_edges = self._set_num_additional_edges(
            data, num_nodes, self.p, len(edges)
        )

        while len(edge_set) < len(edges) + num_additional_edges:
            # Randomly sample a new edge
            u = torch.randint(
                0, num_nodes, (1,), device="cpu", generator=self.generator
            ).item()
            v = torch.randint(
                0, num_nodes, (1,), device="cpu", generator=self.generator
            ).item()

            # Avoid self-loops and duplicate edges
            if u != v:
                edge_set.add((min(u, v), max(u, v)))  # Use sorted edges for consistency

        # Convert edge_set to a PyTorch edge_index on CPU
        edge_index = torch.tensor(list(edge_set), dtype=torch.long, device="cpu").t()

        # Transfer the edge_index to the same device as the input data
        data.edge_index = edge_index.to(data.edge_index.device)
        data.edge_attr = None  # Clear edge attributes if they exist
        return data

    def _random_spanning_tree(self, N):
        """
        Generate a random spanning tree.

        Args:
            N (int): Number of nodes in the graph.

        Returns:
            list: List of edges in the spanning tree.
        """
        # Generate a random spanning tree
        spanning_tree = []
        for i in range(1, N):
            # Connect each node to a random previous node
            spanning_tree.append(
                (i, torch.randint(0, i, (1,), generator=self.generator).item())
            )
        return spanning_tree

    @staticmethod
    def _set_num_additional_edges(data, N, p, l_tree):
        """
        Calculate the number of additional edges to add to a spanning tree.

        Args:
            data (torch_geometric.data.Data): Input graph data object.
            N (int): Number of nodes in the graph.
            p (float): Edge probability.
            l_tree (int): Number of edges in the spanning tree.

        Returns:
            int: Number of additional edges to add.
        """
        num_edges = data.edge_index.size(1) if p is None else int(p * N * (N - 1) / 2)
        num_new_edges = num_edges - l_tree
        return num_new_edges if num_new_edges > 0 else 0
