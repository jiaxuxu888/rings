"""Utility Functions and Helper Transforms"""

import networkx as nx
import torch
from torch_geometric.utils import to_networkx
from torch_geometric.transforms import BaseTransform
from torch_geometric.transforms import OneHotDegree


#  ╭──────────────────────────────────────────────────────────╮
#  │ Helper Transforms                                        |
#  ╰──────────────────────────────────────────────────────────╯


class Shuffle(BaseTransform):
    def __init__(
        self, shuffle_edges=False, shuffle_features=False, generator=None
    ):
        """
        Initialize the Shuffle transform.

        Args:
            shuffle_edges (bool): If True, shuffle the order of edges and reassign them to random nodes.
            shuffle_features (bool): If True, shuffle the order of node features among nodes.
        """
        self.shuffle_edges = shuffle_edges
        self.shuffle_features = shuffle_features
        self.generator = generator

    def __call__(self, data):
        """
        Apply the shuffle transformation to the graph.

        Args:
            data (torch_geometric.data.Data): Input graph data object.

        Returns:
            torch_geometric.data.Data: Transformed data object.
        """
        if self.shuffle_features:
            data = self._shuffle_features(data)
        if self.shuffle_edges:
            data = self._shuffle_edges(data)
        return data

    def _shuffle_features(self, data):
        if hasattr(data, "x") and data.x is not None:
            # Shuffle node features among nodes
            num_nodes = data.num_nodes
            perm = torch.randperm(
                num_nodes, device=data.x.device, generator=self.generator
            )
            data.x = data.x[perm]
        return data

    def _shuffle_edges(self, data):
        if hasattr(data, "edge_index") and data.edge_index is not None:
            # Retrieve the source and target nodes from edge_index
            source_nodes = data.edge_index[0]
            target_nodes = data.edge_index[1]
            num_nodes = data.num_nodes
            if num_nodes is None:
                raise ValueError("Data object must have `num_nodes` defined.")

            # Create a random permutation of node indices for the target nodes
            perm = torch.randperm(
                num_nodes,
                device=data.edge_index.device,
                generator=self.generator,
            )

            # Shuffle target nodes using the permutation
            shuffled_target_nodes = perm[target_nodes]

            # Ensure the original number of edges is preserved
            source_nodes, shuffled_target_nodes = self._ensure_no_self_loops(
                source_nodes, shuffled_target_nodes, num_nodes
            )
            assert shuffled_target_nodes.size() == target_nodes.size()
            # Update edge_index with the shuffled edges
            data.edge_index = torch.stack(
                [source_nodes, shuffled_target_nodes], dim=0
            )

        return data

    def _ensure_no_self_loops(self, source_nodes, target_nodes, num_nodes):
        """
        Ensure no self-loops exist in the shuffled edges. If any self-loops are detected, replace them with random edges (by assigning a random target) to maintain the original number of edges.

        Args:
            source_nodes (Tensor): Source nodes of the edges.
            target_nodes (Tensor): Target nodes of the edges.
            num_nodes (int): Number of nodes in the graph.

        Returns:
            Tuple[Tensor, Tensor]: Updated source and target nodes.
        """
        # Identify self-loops
        self_loop_mask = source_nodes == target_nodes
        num_self_loops = self_loop_mask.sum().item()
        if num_self_loops > 0:
            random_target_nodes = torch.randint(
                0,
                num_nodes,
                (num_self_loops,),
                device=target_nodes.device,
                generator=self.generator,
            )

            # Generate new target nodes ensuring they're not equal to the source nodes to avoid creating self-loops
            valid_targets_mask = (
                random_target_nodes != source_nodes[self_loop_mask]
            )
            while not valid_targets_mask.all():
                invalid_indices = ~valid_targets_mask
                random_target_nodes[invalid_indices] = torch.randint(
                    0,
                    num_nodes,
                    (invalid_indices.sum().item(),),
                    device=target_nodes.device,
                    generator=self.generator,
                )
                valid_targets_mask = (
                    random_target_nodes != source_nodes[self_loop_mask]
                )

            # Replace only target nodes to keep the source nodes intact
            target_nodes[self_loop_mask] = random_target_nodes

        return source_nodes, target_nodes


#  ╭──────────────────────────────────────────────────────────╮
#  │ Utility Functions                                        |
#  ╰──────────────────────────────────────────────────────────╯


def is_connected(data):
    """
    Helper function to check if a graph is connected.

    Args:
        data: Graph data object.

    Returns:
        bool: True if the graph is connected, False otherwise.
            Empty graphs (0 nodes) are (vacuously) considered connected by convention.
    """
    if data.num_nodes == 0:
        # Empty graphs are connected by convention
        return True

    G = to_networkx(data, to_undirected=True)

    return nx.is_connected(G)
