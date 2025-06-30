"""Utility Functions for Mode Complementarity"""

import numpy as np


def maybe_normalize_diameter(D):
    """
    Normalize a distance matrix by its diameter if possible.

    If the distance matrix has a non-zero maximum value (diameter), this function
    normalizes all distances by dividing by that maximum value. This produces a
    distance matrix with values in [0,1]. If the matrix describes a trivial metric
    space (e.g., a single point or multiple identical points), it is returned unchanged.

    Parameters
    ----------
    D : array_like
        Square distance matrix, assumed to describe pairwise distances
        of a finite metric space.

    Returns
    -------
    array_like
        The distance matrix, normalized by its diameter (maximum value).

    Examples
    --------
    >>> import numpy as np
    >>> from rings.complementarity.utils import maybe_normalize_diameter
    >>>
    >>> # Example 1: Non-trivial distance matrix
    >>> D1 = np.array([[0, 2, 4], [2, 0, 6], [4, 6, 0]])
    >>> D1_norm = maybe_normalize_diameter(D1)
    >>> print(D1_norm)
    [[0.         0.33333333 0.66666667]
     [0.33333333 0.         1.        ]
     [0.66666667 1.         0.        ]]
    >>>
    >>> # Example 2: Zero matrix (single point space)
    >>> D2 = np.zeros((3, 3))
    >>> D2_norm = maybe_normalize_diameter(D2)
    >>> print(D2_norm)  # Returns unchanged
    [[0. 0. 0.]
     [0. 0. 0.]
     [0. 0. 0.]]
    """
    if (diam := np.max(D)) > 0:
        D /= diam

    return D
