"Methods for lifting attributes and graphs to metric spaces."

import sklearn
import warnings

from kneed import KneeLocator

from sklearn.metrics.pairwise import pairwise_distances

from sklearn.neighbors import kneighbors_graph

import numpy as np
import networkx as nx

from src.utils import calculate_diffusion_operator
from src.utils import von_neumann_entropy

import src.metrics as supported_metrics

#  ╭──────────────────────────────────────────────────────────╮
#  │ Lifters                                                  │
#  ╰──────────────────────────────────────────────────────────╯


def lift_attributes(X, metric, n_jobs, **kwargs):
    """Lift attributes to a metric space.

    Lift attributes of a graph to a metric space by calculating their
    pairwise distances either directly or via delegation to`sklearn`.
    """
    operator = getattr(supported_metrics, metric, None)

    if metric in sklearn.metrics.pairwise._VALID_METRICS:
        operator = standard_feature_metrics
    elif operator is None:
        raise RuntimeError(f"Unsupported metric: {metric}")

    return operator(X, metric=metric, n_jobs=n_jobs, **kwargs)


def lift_graph(G, metric, **kwargs):
    """Lift graph to a metric space.

    Lift graph to a metric space by calculating distances between its
    nodes using a specific graph metric.
    """

    # Immediately return zero metric space if graph is empty.
    if G.number_of_edges() == 0:
        n = G.number_of_nodes()
        return np.zeros((n, n))

    operator = getattr(supported_metrics, metric, None)

    if operator is None:
        raise RuntimeError(f"Unsupported metric: {metric}")

    return operator(G, **kwargs)


#  ╭──────────────────────────────────────────────────────────╮
#  │ Attribute Metrics                                        │
#  ╰──────────────────────────────────────────────────────────╯


def standard_feature_metrics(X, **kwargs):
    """
    Calculate pairwise distances between features.

    This method calculates pairwise distances between features in the
    attribute space of a graph. The default metric is the Euclidean metric.

    Parameters
    ----------
    X : array_like
        Graph feature space, with rows indicating nodes and columns
        corresponding to feature dimensions.

    Returns
    -------
    np.array
        Matrix of distance values
    """
    return pairwise_distances(
        X,
        metric=kwargs.get("metric", "euclidean"),
        n_jobs=kwargs.get("n_jobs", None),
    )


def potential_distance(X, **kwargs):
    """Calculate the potential distance based on an RBF kernel.

    This implements the potential distance introduced by Coifman and
    Lafon [1]_, using an RBF kernel. The kernel bandwidth $epsilon$
    is estimated based on the largest nearest-neighbour distance.

    Parameters
    ----------
    X : array_like
        Graph feature space, with rows indicating nodes and columns
        corresponding to feature dimensions.

    t : int or `None`
        Number of steps for the diffusion operator that is used for
        the potential distance. If set to `None`, a suitable $t$ is
        selected based on the von Neumann entropy.

    epsilon : float or `None`
        Neighbourhood smoothing parameter. If set to `None`, a scale
        will be estimated based on nearest-neighbour distances.

    n_jobs : int or None
        Number of jobs for a `joblib.Parallel` context.

    Notes
    -----
    This implementation is currently restricted to Euclidean distances;
    while picking a different metric is technically possible, there are
    some issues with ensuring the definiteness of the resulting kernel.
    The easiest way is to side-step this issue for now.

    References
    ----------
    .. [1] R. R. Coifman, S. Lafon, "Diffusion maps," Applied and Computational
       Harmonic Analysis, Volume 21, Issue 1, pp. 5–30, 2006.
    """
    # Unless the client specified a default parameter, we have to
    # estimate it.
    t = kwargs.get("num_steps", None)
    epsilon = kwargs.get("epsilon", None)
    n_jobs = kwargs.get("n_jobs", None)

    if epsilon is None:
        # Use only the unique elements of the array. This ensures
        # that we get a nonzero estimate (unless all features are
        # identical).
        X_ = np.unique(X, axis=0)

        # Required for some data sets because `kneighbors_graph` may be
        # unable to process them.
        if len(X_) <= 1:
            epsilon = 1.0
        else:
            knn_distances = kneighbors_graph(
                X_, 1, mode="distance", n_jobs=n_jobs
            )
            epsilon = np.max(knn_distances)

    P = calculate_diffusion_operator(X, epsilon)

    if t is None:
        entropy = von_neumann_entropy(P, 100)

        kneedle = KneeLocator(
            x=np.arange(start=1, stop=len(entropy) + 1, dtype=int),
            y=entropy,
            curve="convex",
            direction="decreasing",
        )

        t = kneedle.elbow

        # Unable to get a proper elbow; let's default to not going
        # deeper into the features. This condition will only occur
        # for degenerate data (e.g. if no features are present, or
        # the ones that are present are not unique).
        if not isinstance(t, int):
            t = 1

    # Powered diffusion operator with a semi-hacky variant for handling
    # values that are too small.
    Pt = np.linalg.matrix_power(P, t)
    Pt += 1e-8

    diffusion_potential = -np.log(Pt)
    distances = pairwise_distances(diffusion_potential, n_jobs=n_jobs)

    return distances


#  ╭──────────────────────────────────────────────────────────╮
#  │ Graph Metrics                                            │
#  ╰──────────────────────────────────────────────────────────╯


def diffusion_distance(G, **kwargs):
    """Calculate diffusion distance between vertices of a graph.

    Parameters
    ----------
    G : nx.Graph
        Input graph. All attributes of the graph will be ignored in
        the subsequent calculations.

    num_steps : int
        Number of steps for the diffusion operator that is used for
        the distance calculations.

    norm: bool (default True)
        Normalize the Laplacian.

    symmetric : bool (default True)
        Normalize the Laplacian _symmetrically_.

    Returns
    -------
    np.array
        Matrix of distance values
    """
    t = kwargs.get("num_steps", 1)
    n_jobs = kwargs.get("n_jobs", None)
    symmetric = kwargs.get("symmetric", True)
    norm = kwargs.get("norm", True)

    if not norm and not symmetric:
        warnings.warn(
            "Assuming default Laplacian, which is symmetric.",  # noqa: E501
            UserWarning,
        )
        symmetric = True

    A = nx.adjacency_matrix(G).todense()
    D = np.diag(A.sum(axis=1))

    # Bail out early on if there are isolated nodes.
    if (np.diagonal(D) == 0).any():
        return np.nan

    # NEW VERSION
    L = _compute_laplacian(A, D, norm=norm, symmetric=symmetric)
    psi = _compute_psi(L, t, symmetric=symmetric)

    if np.iscomplexobj(psi):
        warnings.warn(
            "Input data contains complex numbers. The imaginary part will be discarded.",  # noqa: E501
            UserWarning,
        )
        psi = np.real(psi)
    return pairwise_distances(psi, metric="euclidean", n_jobs=n_jobs)


def heat_kernel_distance(G, **kwargs):
    """Calculate heat kernel distance between vertices of a graph.

    Parameters
    ----------
    G : nx.Graph
        Input graph. All attributes of the graph will be ignored in
        the subsequent calculations.

    t : int or `None`
        Number of steps for the diffusion operator that is used for
        the potential distance. If set to `None`, a suitable $t$ is
        selected based on the von Neumann entropy.

    Returns
    -------
    np.array
        Matrix of distance values
    """
    t = kwargs.get("num_steps", 1)
    n_jobs = kwargs.get("n_jobs", None)
    symmetric = kwargs.get("symmetric", True)
    norm = kwargs.get("norm", True)

    A = nx.adjacency_matrix(G).todense()
    D = np.diag(A.sum(axis=1))

    L = _compute_laplacian(A, D, norm=norm, symmetric=symmetric)

    eigenvalues, eigenvectors = _compute_spectrum(L, symmetric=symmetric)
    scales = np.exp(-t * eigenvalues)
    X = scales * eigenvectors

    if np.iscomplexobj(X):
        warnings.warn(
            "Input data contains complex numbers. The imaginary part will be discarded.",  # noqa: E501
            UserWarning,
        )
        X = np.real(X)

    return pairwise_distances(X, metric="euclidean", n_jobs=n_jobs)


def resistance_distance(G, weight=None, **kwargs):
    """
    Calculate resistance distance between vertices of a graph.

    Parameters
    ----------
    G : nx.Graph
        Input graph. All attributes of the graph will be ignored in
        the subsequent calculations.

    weight : str or None
        The edge attribute that holds the numerical value used as a
        weight. If set to `None`, the graph is treated as unweighted.

    kwargs
        Additional keyword arguments. Only required for compatibility
        reasons.

    Returns
    -------
    np.array
        Matrix of distance values
    """
    try:
        distances = nx.resistance_distance(G, weight=weight)
        distances = nx.utils.dict_to_numpy_array(distances)
    except nx.NetworkXError:
        distances = [np.nan]

    return distances


def shortest_path_distance(G, weight=None, **kwargs):
    """Calculate shortest-path distance between vertices.

    Calculate shortest-path distance between vertices of a graph using
    the Floyd--Warshall algorithm.

    Parameters
    ----------
    G : nx.Graph
        Input graph. All attributes of the graph will be ignored in
        the subsequent calculations.

    weight : str or None
        The edge attribute that holds the numerical value used as a
        weight. If set to `None`, the graph is treated as unweighted.

    kwargs
        Additional keyword arguments. Only required for compatibility
        reasons.

    Returns
    -------
    np.array
        Matrix of distance values
    """
    return nx.floyd_warshall_numpy(G, weight=weight)


#  ╭──────────────────────────────────────────────────────────╮
#  │ Spectral Helper Functions                                │
#  ╰──────────────────────────────────────────────────────────╯


def _compute_laplacian(A, D, norm=True, symmetric=True):
    """
    Compute the Laplacian matrix of a graph.

    Parameters
    ----------
    A : numpy.ndarray
        Adjacency matrix of the graph.
    D : numpy.ndarray
        Degree matrix of the graph.
    norm : bool, optional
        If True, compute the normalized Laplacian. Default is True.
    symmetric : bool, optional
        If True, compute the symmetric normalized Laplacian. Default is True.

    Returns
    -------
    L : numpy.ndarray
        The Laplacian matrix of the graph.

    Notes
    -----
    The Laplacian matrix is defined as L = D - A, where D is the degree matrix
    and A is the adjacency matrix.

    If `norm` is True and `symmetric` is True, the symmetric normalized Laplacian is computed as L = I - D^(-1/2) * A * D^(-1/2).

    If `norm` is True and `symmetric` is False, the random walk normalized Laplacian is computed as L = I - D^(-1) * A.
    """
    if norm and symmetric:
        D_inv_sqrt = np.diag(1.0 / np.sqrt(D.diagonal()))
        L = np.eye(*A.shape) - D_inv_sqrt @ A @ D_inv_sqrt
    elif norm and not symmetric:
        L = np.eye(*A.shape) - np.diag(1.0 / D.diagonal()) @ A
    else:
        L = D - A
    return L


def _compute_psi(L, t, symmetric=True):
    """
    Compute Psi, a matrix made up of powered eigenvalues^t * eigenvectors of the Laplacian matrix.

    Parameters
    ----------
    L : array_like
        The Laplacian matrix.
    t : float
        The exponent to which the eigenvalues are raised.
    symmetric : bool, optional
        If True, assumes the Laplacian matrix is symmetric and uses `np.linalg.eigh`.
        If False, uses `np.linalg.eig`. Default is True.

    Returns
    -------
    psi : ndarray
        The Psi matrix computed from the eigenvalues and eigenvectors of the Laplacian matrix.

    Notes
    -----
    If `symmetric` is True, the function uses `np.linalg.eigh` which is more efficient for symmetric matrices.
    Otherwise, it uses `np.linalg.eig`.
    """
    eigenvalues, eigenvectors = _compute_spectrum(L, symmetric=symmetric)
    eigenvalues = np.power(eigenvalues, t)
    psi = eigenvalues * eigenvectors
    return psi


def _compute_spectrum(L, symmetric=True):
    """
    Compute the eigenvalues and eigenvectors of the Laplacian matrix.

    Parameters
    ----------
    L : array_like
        The Laplacian matrix.
    symmetric : bool, optional
        If True, assumes the Laplacian matrix is symmetric and uses `np.linalg.eigh`.
        If False, uses `np.linalg.eig`. Default is True.

    Returns
    -------
    eigenvalues : ndarray
        The eigenvalues of the Laplacian matrix.
    eigenvectors : ndarray
        The eigenvectors of the Laplacian matrix.

    Notes
    -----
    If `symmetric` is True, the function uses `np.linalg.eigh` which is more efficient for symmetric matrices.
    Otherwise, it uses `np.linalg.eig`.
    """
    if symmetric:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
    else:
        eigenvalues, eigenvectors = np.linalg.eig(L)
    return eigenvalues, eigenvectors
