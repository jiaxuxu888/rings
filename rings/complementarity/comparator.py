"""This module contains classes that compare metric spaces (i.e. pairwise distance matrices)."""

import numpy as np


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
    __call__(D_X, D_G)
        Calculate complementarity based on the specified matrix norm.
    L11_norm(M)
        Calculate the L1,1 norm of a matrix.
    frobenius_norm(M)
        Calculate the Frobenius norm of a matrix.
    invalid_data
        Return a dictionary with NaN complementarity score.
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

    def __call__(self, D_X, D_G):
        """
        Calculate complementarity based on a matrix norm.
        Parameters
        ----------
        D_X : numpy.ndarray
            The first matrix for comparison.
        D_G : numpy.ndarray
            The second matrix for comparison.
        Returns
        -------
        dict
            A dictionary with the complementarity score.
        """
        n = len(D_X)
        divisor = n**2 - n if n > 1 else 1

        if self.norm == "L11":
            complementarity_score = self.L11_norm(D_X - D_G)
        elif self.norm == "frobenius":
            complementarity_score = self.frobenius_norm(D_X - D_G)
            divisor = np.sqrt(divisor)
        else:
            raise RuntimeError(f"Unexpected norm '{self.norm}'")

        complementarity_score /= divisor

        return {"complementarity": complementarity_score}

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
        Return a dictionary with NaN complementarity score.
        Returns
        -------
        dict
            A dictionary with NaN complementarity score.
        """
        return {"complementarity": np.nan}
