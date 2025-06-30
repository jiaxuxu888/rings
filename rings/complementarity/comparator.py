"""This module contains classes that compare metric spaces (i.e. pairwise distance matrices)."""

import numpy as np
from typing import Dict, Optional


class MatrixNormComparator:
    """
    A class to compare matrices based on specified norms and calculate complementarity.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments to specify the norm to be used. Supported norms are:
        - "L11" (default)
        - "frobenius"

    Attributes
    ----------
    norm : str
        The norm to be used for comparison.

    Methods
    -------
    __call__(x, y, **kwargs)
        Compare two matrices using the specified norm.

    L11_norm(M)
        Calculate the L1,1 norm of a matrix.

    frobenius_norm(M)
        Calculate the Frobenius norm of a matrix.

    invalid_data
        Return a dictionary with NaN score and other standardized fields.
    """

    def __init__(self, **kwargs):
        """
        Initialize the MatrixNormComparator with the specified norm.
        Parameters
        ----------
        **kwargs : dict
            Keyword arguments to specify the norm to be used. Supported norms are:
            - "L11" (default)
            - "frobenius"
        """
        self.norm = kwargs["norm"] if "norm" in kwargs else "L11"

    def __call__(self, x, y, **kwargs):
        """
        Compare two matrices using the specified norm.

        Parameters
        ----------
        x : numpy.ndarray
            The first matrix for comparison.
        y : numpy.ndarray
            The second matrix for comparison.
        **kwargs : dict, optional
            Additional keyword arguments (not used).

        Returns
        -------
        dict
            Dictionary with keys:
            - score: The computed norm-based complementarity score
            - pvalue: None (p-values not computed for norm-based comparisons)
            - pvalue_adjusted: None
            - significant: None
            - method: The norm used (e.g., "L11" or "frobenius")
        """
        n = len(x)
        divisor = n**2 - n if n > 1 else 1

        if self.norm == "L11":
            complementarity_score = self.L11_norm(x - y)
        elif self.norm == "frobenius":
            complementarity_score = self.frobenius_norm(x - y)
            divisor = np.sqrt(divisor)
        else:
            raise RuntimeError(f"Unexpected norm '{self.norm}'")

        complementarity_score /= divisor

        return {
            "score": complementarity_score,
            "pvalue": None,
            "pvalue_adjusted": None,
            "significant": None,
            "method": self.norm,
        }

    @staticmethod
    def L11_norm(M):
        """
        Calculate the L1,1 norm of a matrix.
        Parameters
        ----------
        M : numpy.ndarray
            The matrix for which to calculate the L1,1 norm.
        Returns
        -------
        float
            The L1,1 norm of the matrix.
        """
        return abs(M).sum()

    @staticmethod
    def frobenius_norm(M):
        """
        Calculate the Frobenius norm of a matrix.
        Parameters
        ----------
        M : numpy.ndarray
            The matrix for which to calculate the Frobenius norm.
        Returns
        -------
        float
            The Frobenius norm of the matrix.
        """
        return np.linalg.norm(M, ord="fro")

    @property
    def invalid_data(self):
        """
        Return a dictionary with NaN score and other standardized fields.

        Returns
        -------
        dict
            Dictionary with keys:
            - score: NaN (invalid data)
            - pvalue: None
            - pvalue_adjusted: None
            - significant: None
            - method: The norm used
        """
        return {
            "score": np.nan,
            "pvalue": None,
            "pvalue_adjusted": None,
            "significant": None,
            "method": self.norm,
        }


#  ╭──────────────────────────────────────────────────────────╮
#  │ Factory Functions                                        |
#  ╰──────────────────────────────────────────────────────────╯


def L11MatrixNormComparator(**kwargs):
    """
    Factory function that returns a MatrixNormComparator with L11 norm.

    Returns
    -------
    MatrixNormComparator
        A comparator configured to use the L11 norm
    """
    return MatrixNormComparator(norm="L11", **kwargs)


def FrobeniusMatrixNormComparator(**kwargs):
    """
    Factory function that returns a MatrixNormComparator with Frobenius norm.

    Returns
    -------
    MatrixNormComparator
        A comparator configured to use the Frobenius norm
    """
    return MatrixNormComparator(norm="frobenius", **kwargs)
