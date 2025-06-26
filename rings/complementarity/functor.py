"""Core Functor for facilitating Mode Complementarity computation."""

from rings.complementarity.metrics import lift_attributes
from rings.complementarity.metrics import lift_graph
from rings.complementarity.utils import maybe_normalize_diameter

from torch_geometric.utils import to_networkx

import numpy as np
import networkx as nx

import joblib
import operator
import torch
import warnings


class ComplementarityFunctor(torch.nn.Module):
    def __init__(
        self,
        feature_metric,
        graph_metric,
        comparator,
        n_jobs,
        **kwargs,
    ):
        torch.nn.Module.__init__(self)

        self.n_jobs = n_jobs
        self.feature_metric = feature_metric
        self.graph_metric = graph_metric

        self._use_edge_information = False
        self.kwargs = kwargs

        # Set up the comparator
        self.comparator = comparator(n_jobs=n_jobs, **self.kwargs)

    @property
    def use_edge_information(self):
        return self._use_edge_information

    @use_edge_information.setter
    def use_edge_information(self, value):
        self._use_edge_information = value
        if value:
            self.kwargs["weight"] = "edge_attr"
        elif "weight" in self.kwargs and self.kwargs["weight"] == "edge_attr":
            del self.kwargs["weight"]

    def forward(self, batch):
        """Handle a batch of graphs in `pytorch-geometric` format."""
        batch_size = len(batch)

        outputs = joblib.Parallel(n_jobs=self.n_jobs)(
            joblib.delayed(self._forward)(batch[i]) for i in range(batch_size)
        )

        # This attribute is guaranteed to be available in all functors,
        # and we can always convert it to a tensor.
        complementarity_scores = torch.as_tensor(
            list(map(operator.itemgetter("complementarity"), outputs)),
            dtype=torch.float,
        )

        result = {"complementarity": complementarity_scores}

        if len(outputs) > 0:
            other_keys = list(outputs[0].keys())
            other_keys.remove("complementarity")

            for key in other_keys:
                data = list(map(operator.itemgetter(key), outputs))
                result[key] = data

        return result

    def _forward(self, data):
        """Handle single graph in `pytorch-geometric` format.

        Parameters
        ----------
        data : torch_geometric.Data
            The graph, represented as an input tensor with optional node
            attributes and edge attributes.

        Returns
        -------
        dict
            The complementarity score of the graph, calculated using the
            global parameters of the class, followed by optional info
            that depends on the selected comparator function.
        """
        G = to_networkx(
            data,
            to_undirected=True,
            node_attrs=["x"],
            edge_attrs=["edge_attr"] if "edge_attr" in data else None,
        )

        # Check if edge attributes need additional pre-processing. We
        # generally reduce them via a norm to turn them into scalars.
        if "edge_attr" in data and self.use_edge_information:
            attributes = nx.get_edge_attributes(G, "edge_attr")

            attributes.update(
                {
                    edge: np.linalg.norm(attribute)
                    for edge, attribute in attributes.items()
                }
            )

            nx.set_edge_attributes(G, attributes, "edge_attr")

        output = self._complementarity(G)
        return output

    def _complementarity(self, G, return_metric_spaces: bool = False):
        """Calculate complementarity of a single graph.

        This is the core function that calculates the complementarity of
        a single graph, subject to shared parameters.

        Parameters
        ----------
        G : networkx.Graph
            The graph, represented as `networkx.Graph` with optional
            node attributes and edge attributes. Node attributes are
            assumed to be stored in an attribute called `x`, whereas
            edge attributes are optionally stored under `edge_attr`.

            Node attributes may be arbitrary `array_like` (they need
            to be compatible with the selected metric, though) while
            edge attributes need to be scalars.

        return_metric_spaces (optional, default False):
            Return the metric spaces being compared.

        Returns
        -------
        dict
            The complementarity score of the graph, calculated using the
            global parameters of the class, followed by optional info
            that depends on the selected comparator function.
        """
        X = np.asarray(list(nx.get_node_attributes(G, "x").values()))

        empty_graph = False if G.number_of_edges() > 0 else True

        if len(X) == 0:
            warnings.warn("Feature matrix X is empty, skipping graph.")
            return self.comparator.invalid_data

        if G.number_of_nodes() == 0:
            warnings.warn("Graph G has no nodes, skipping it.")
            return self.comparator.invalid_data

        # Check if the graph is connected
        if not nx.is_connected(G) and not empty_graph:
            # Process each connected component and compute their complementarity score
            components = list(nx.connected_components(G))

            # Lift graphs for each component
            D_G = [
                lift_graph(
                    G.subgraph(C), metric=self.graph_metric, **self.kwargs
                )
                for C in components
            ]

            # Lift attributes (node features) for each component
            D_X = [
                lift_attributes(
                    X[list(C)],
                    metric=self.feature_metric,
                    n_jobs=self.n_jobs,
                    **self.kwargs,
                )
                for C in components
            ]

            sizes = [len(C) for C in components]
        else:
            # Lift Graph into Metric Space
            D_G = [lift_graph(G, metric=self.graph_metric, **self.kwargs)]

            # Lift Features into Metric Space
            D_X = [
                lift_attributes(
                    X,
                    metric=self.feature_metric,
                    n_jobs=self.n_jobs,
                    **self.kwargs,
                )
            ]

            sizes = [len(G.nodes())]

        # Normalize diameters for both D_X and D_G
        D_X = [maybe_normalize_diameter(d_x) for d_x in D_X]
        D_G = [maybe_normalize_diameter(d_g) for d_g in D_G]

        # Compute complementarity scores for each component
        scores = [
            self.comparator(d_x, d_g)["complementarity"]
            for d_x, d_g in zip(D_X, D_G)
        ]

        # Compute the weighted average score
        if sum(sizes) > 0:
            weighted_average_score = np.average(scores, weights=sizes)
        else:
            # If sizes sum to zero, use simple average or return NaN
            warnings.warn("Weights sum to zero, using simple average instead")
            if scores:
                weighted_average_score = np.mean(scores)
            else:
                weighted_average_score = np.nan

        return_dict = {
            "complementarity": weighted_average_score,
        }
        if return_metric_spaces:
            return_dict["D_X"] = D_X
            return_dict["D_G"] = D_G

        return return_dict
