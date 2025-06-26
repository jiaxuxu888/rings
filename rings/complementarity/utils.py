"""Utility Functions for Mode Complementarity"""

import numpy as np


def maybe_normalize_diameter(D):
    """Normalises a distance matrix if possible.

    If the distance matrix is non-zero, i.e. it describes a space with
    more than a single point, we normalise its diameter. Else, we just
    return the matrix unchanged.

    Parameters
    ----------
    D : array_like
        Square distance matrix, assumed to describe pairwise distances
        of a finite metric space.

    Returns
    -------
    array_like
        The distance matrix, normalised by its diameter, i.e. its
        largest non-zero value.
    """
    if (diam := np.max(D)) > 0:
        D /= diam

    return D
