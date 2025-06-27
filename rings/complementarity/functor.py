"""Core Functor for facilitating Mode Complementarity computation."""

from rings.complementarity.metrics import lift_attributes
from rings.complementarity.metrics import lift_graph
from rings.complementarity.utils import maybe_normalize_diameter

from torch_geometric.utils import to_networkx

import numpy as np
import networkx as nx
import pandas as pd

import joblib
import operator
import torch
import warnings
from typing import Dict, List, Union, Optional, Any


class ComplementarityFunctor(torch.nn.Module):
    """
    A functor for computing complementarity between graph structure and node features.

    This class computes complementarity by comparing the metric spaces derived from
    graph structure and node features using a specified comparator.

    Parameters
    ----------
    feature_metric : callable
        The metric function used to compute distances between node features.
    graph_metric : callable
        The metric function used to compute distances in the graph structure.
    comparator : class
        A comparator class that implements a __call__ method to compare metric spaces.
    n_jobs : int, default=1
        Number of jobs to run in parallel. If 1, no parallelism is used.
    use_edge_information : bool, default=False
        Whether to use edge attributes in graph metric computation.
    normalize_diameters : bool, default=True
        Whether to normalize the diameters of metric spaces before comparison.
    **kwargs : dict
        Additional arguments passed to the comparator and metric functions.
    """

    def __init__(
        self,
        feature_metric,
        graph_metric,
        comparator,
        n_jobs: int = 1,
        use_edge_information: bool = False,
        normalize_diameters: bool = True,
        **kwargs,
    ):
        torch.nn.Module.__init__(self)

        self.n_jobs = n_jobs
        self.feature_metric = feature_metric
        self.graph_metric = graph_metric
        self.normalize_diameters = normalize_diameters
        self.use_edge_information = use_edge_information

        # Build kwargs for metrics and comparator
        self.kwargs = kwargs.copy()

        # Set edge weight parameter if using edge information
        if self.use_edge_information:
            self.kwargs["weight"] = "edge_attr"

        # Set up the comparator
        self.comparator = comparator(n_jobs=n_jobs, **self.kwargs)

    def forward(
        self,
        batch,
        as_dataframe: bool = True,
    ) -> Union[Dict[str, Any], pd.DataFrame, List[Dict[str, Any]]]:
        """
        Compute complementarity for a batch of graphs in PyTorch Geometric format.

        Parameters
        ----------
        batch : list
            A batch of graphs in PyTorch Geometric format.
        as_dataframe : bool, default=True
            If True, return results as a pandas DataFrame, otherwise as a dictionary
            with tensor values or a list of dictionaries.

        Returns
        -------
        Union[Dict[str, Any], pd.DataFrame, List[Dict[str, Any]]]
            Results of complementarity computation for each graph in the batch.
            If as_dataframe=True, returns a pandas DataFrame.
            Otherwise, returns either a dictionary with tensor values or a list of dictionaries.

        Examples
        --------
        >>> from rings.complementarity.comparator import MatrixNormComparator
        >>> import torch_geometric.datasets as datasets
        >>> dataset = datasets.TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
        >>> functor = ComplementarityFunctor(
        ...     feature_metric=lambda x, y: np.linalg.norm(x - y),
        ...     graph_metric=lambda x, y: abs(x - y),
        ...     comparator=MatrixNormComparator,
        ...     n_jobs=4,
        ...     normalize_diameters=True
        ... )
        >>> # Get first 5 graphs as a batch
        >>> batch = [dataset[i] for i in range(5)]
        >>> results = functor.forward(batch)
        >>> print(results)
        """
        batch_size = len(batch)

        # Process batch in parallel or sequentially
        if self.n_jobs > 1 and batch_size > 1:
            outputs = joblib.Parallel(n_jobs=self.n_jobs)(
                joblib.delayed(self._process_single)(batch[i])
                for i in range(batch_size)
            )
        else:
            outputs = [
                self._process_single(batch[i]) for i in range(batch_size)
            ]

        # Convert to DataFrame if requested
        if as_dataframe:
            return pd.DataFrame(outputs)

        # Otherwise, return as tensor dictionary or list
        complementarity_scores = torch.as_tensor(
            list(map(operator.itemgetter("complementarity"), outputs)),
            dtype=torch.float,
        )

        result = {"complementarity": complementarity_scores}

        if len(outputs) > 0:
            other_keys = list(outputs[0].keys())
            if "complementarity" in other_keys:
                other_keys.remove("complementarity")

            for key in other_keys:
                result[key] = [output[key] for output in outputs]

        # If DataFrame format wasn't requested but there's only one result,
        # return the list of individual results
        if len(outputs) > 1:
            return result
        else:
            return outputs

    def _process_single(self, data) -> Dict[str, Any]:
        """
        Process a single graph for complementarity calculation.

        Parameters
        ----------
        data : torch_geometric.Data
            The graph in PyTorch Geometric format.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the complementarity score and any additional metrics.
        """
        # Step 1: Preprocess graph
        G = self._preprocess_graph(data)

        # Step 2: Compute complementarity
        return self._compute_complementarity(G)

    def _preprocess_graph(self, data):
        """
        Preprocess a graph from PyTorch Geometric format to NetworkX.

        Parameters
        ----------
        data : torch_geometric.Data
            The graph in PyTorch Geometric format.

        Returns
        -------
        networkx.Graph
            The preprocessed graph in NetworkX format.
        """
        # Convert to NetworkX
        G = to_networkx(
            data,
            to_undirected=True,
            node_attrs=["x"],
            edge_attrs=["edge_attr"] if "edge_attr" in data else None,
        )

        # Process edge attributes if needed
        if "edge_attr" in data and self.use_edge_information:
            attributes = nx.get_edge_attributes(G, "edge_attr")

            # Convert multi-dimensional edge attributes to scalars using norm
            attributes.update(
                {
                    edge: np.linalg.norm(attribute)
                    for edge, attribute in attributes.items()
                }
            )

            nx.set_edge_attributes(G, attributes, "edge_attr")

        return G

    def _compute_complementarity(
        self, G, return_metric_spaces: bool = False
    ) -> Dict[str, Any]:
        """
        Calculate complementarity of a single graph.

        This method breaks down the complementarity calculation into clear steps:
        1. Extract node features
        2. Lift graph structure and node features to metric spaces
        3. Normalize metric spaces if required
        4. Compute complementarity scores
        5. Aggregate results from multiple components if needed

        Parameters
        ----------
        G : networkx.Graph
            The graph, represented as `networkx.Graph` with optional
            node attributes and edge attributes. Node attributes are
            assumed to be stored in an attribute called `x`, whereas
            edge attributes are optionally stored under `edge_attr`.

        return_metric_spaces : bool, default=False
            Return the metric spaces being compared.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the complementarity score and any additional metrics.
        """
        # Extract node features
        X = np.asarray(list(nx.get_node_attributes(G, "x").values()))

        # Validate input
        empty_graph = G.number_of_edges() == 0

        if len(X) == 0:
            warnings.warn("Feature matrix X is empty, skipping graph.")
            return self.comparator.invalid_data

        if G.number_of_nodes() == 0:
            warnings.warn("Graph G has no nodes, skipping it.")
            return self.comparator.invalid_data

        # Handle connected components
        metric_spaces = self._lift_metrics(G, X, empty_graph)
        if metric_spaces is None:
            return self.comparator.invalid_data

        D_X, D_G, sizes = metric_spaces

        # Normalize metric spaces if required
        if self.normalize_diameters:
            D_X, D_G = self._normalize_metrics(D_X, D_G)

        # Compute scores for each component
        scores = self._compute_scores(D_X, D_G)

        # Aggregate results
        weighted_average_score = self._aggregate(scores, sizes)

        # Prepare return dictionary
        result = {"complementarity": weighted_average_score}

        if return_metric_spaces:
            result["D_X"] = D_X
            result["D_G"] = D_G

        return result

    def _lift_metrics(self, G, X, empty_graph: bool):
        """
        Lift graph structure and node features to metric spaces.

        Parameters
        ----------
        G : networkx.Graph
            The input graph.
        X : numpy.ndarray
            Node feature matrix.
        empty_graph : bool
            Whether the graph has no edges.

        Returns
        -------
        tuple or None
            A tuple of (D_X, D_G, sizes) containing feature metric spaces,
            graph metric spaces, and component sizes. Returns None if operation fails.
        """
        # Check if the graph is connected
        if not nx.is_connected(G) and not empty_graph:
            # Process each connected component separately
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

        return D_X, D_G, sizes

    def _normalize_metrics(self, D_X, D_G):
        """
        Normalize the diameters of metric spaces.

        Parameters
        ----------
        D_X : list of numpy.ndarray
            List of feature metric spaces.
        D_G : list of numpy.ndarray
            List of graph metric spaces.

        Returns
        -------
        tuple
            A tuple of normalized (D_X, D_G).
        """
        # Normalize diameters for both D_X and D_G
        D_X_normalized = [maybe_normalize_diameter(d_x) for d_x in D_X]
        D_G_normalized = [maybe_normalize_diameter(d_g) for d_g in D_G]

        return D_X_normalized, D_G_normalized

    def _compute_scores(self, D_X, D_G):
        """
        Compute complementarity scores for each component.

        Parameters
        ----------
        D_X : list of numpy.ndarray
            List of feature metric spaces.
        D_G : list of numpy.ndarray
            List of graph metric spaces.

        Returns
        -------
        list
            List of complementarity scores for each component.
        """
        # Compute complementarity scores for each component
        return [
            self.comparator(d_x, d_g)["complementarity"]
            for d_x, d_g in zip(D_X, D_G)
        ]

    def _aggregate(self, scores, sizes):
        """
        Aggregate scores from multiple components using weighted average.

        Parameters
        ----------
        scores : list
            List of complementarity scores.
        sizes : list
            List of component sizes (weights).

        Returns
        -------
        float
            Weighted average complementarity score.
        """
        # Compute the weighted average score
        if sum(sizes) > 0:
            return np.average(scores, weights=sizes)
        else:
            # If sizes sum to zero, use simple average or return NaN
            warnings.warn("Weights sum to zero, using simple average instead")
            if scores:
                return np.mean(scores)
            else:
                return np.nan
