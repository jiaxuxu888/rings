import numpy as np
import itertools
import pandas as pd
from typing import Dict, List, Union, Optional, Any

# For parallelism
try:
    import joblib
except ImportError:
    joblib = None


class SeparabilityFunctor:
    """
    A functor for computing separability between multiple distributions.

    This class computes all pairwise comparisons between distributions,
    applies Bonferroni correction, and returns structured results.

    Parameters
    ----------
    comparator : object
        An instance of a comparator class (e.g., KSComparator, WilcoxonComparator)
        that implements a compare() method.
    n_jobs : int, default=1
        Number of jobs to run in parallel. If 1, no parallelism is used.
    alpha : float, default=0.01
        Family-wise significance level for hypothesis testing.
    """

    def __init__(self, comparator, n_jobs: int = 1, alpha: float = 0.01):
        """
        Initialize the SeparabilityFunctor.

        Parameters
        ----------
        comparator : object
            An instance of a comparator class (e.g., KSComparator, WilcoxonComparator)
            that implements a compare() method.
        n_jobs : int, default=1
            Number of jobs to run in parallel. If 1, no parallelism is used.
        alpha : float, default=0.01
            Family-wise significance level for hypothesis testing.
        """
        self.comparator = comparator
        self.n_jobs = n_jobs
        self.alpha = alpha

        # Check if joblib is available when n_jobs > 1
        if self.n_jobs > 1 and joblib is None:
            raise ImportError(
                "joblib is required for parallel execution. "
                "Install it with 'pip install joblib'."
            )

    def _run_comparison(
        self,
        mode1: str,
        mode2: str,
        s1: np.ndarray,
        s2: np.ndarray,
        n_tests: int,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run a single comparison between two distributions.

        Parameters
        ----------
        mode1 : str
            Name of the first distribution.
        mode2 : str
            Name of the second distribution.
        s1 : np.ndarray
            First sample of measurements.
        s2 : np.ndarray
            Second sample of measurements.
        n_tests : int
            Total number of tests for Bonferroni correction.
        **kwargs
            Additional arguments to pass to the comparator.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the results of the comparison.
        """
        # Run the comparison
        result = self.comparator.compare(
            s1, s2, alpha=self.alpha, n_hypotheses=n_tests, **kwargs
        )

        # Add the mode names to the result
        result["mode1"] = mode1
        result["mode2"] = mode2

        return result

    def forward(
        self,
        distributions: Dict[str, Union[List[float], np.ndarray]],
        n_permutations: int = 10_000,
        random_state: Optional[int] = 42,
        as_dataframe: bool = True,
    ) -> Union[List[Dict[str, Any]], pd.DataFrame]:
        """
        Compute all pairwise separability tests between distributions.

        Parameters
        ----------
        distributions : Dict[str, Union[List[float], np.ndarray]]
            Dictionary mapping mode names to performance distributions.
            Example: {"Original": np.array([0.8, 0.7, ...]), "EmptyGraph": np.array([0.5, 0.6, ...])}
        n_permutations : int, default=10_000
            Number of permutations for the statistical tests.
        random_state : Optional[int], default=42
            Random seed for reproducibility.
        as_dataframe : bool, default=True
            If True, return results as a pandas DataFrame, otherwise as a list of dictionaries.

        Returns
        -------
        Union[List[Dict[str, Any]], pd.DataFrame]
            Results of all pairwise comparisons, either as a list of dictionaries or a pandas DataFrame.
        """
        # Get all unique pairs of modes
        pairs = list(itertools.combinations(distributions.keys(), 2))
        n_tests = len(pairs)

        # If there are no pairs, return early
        if n_tests == 0:
            if as_dataframe:
                return pd.DataFrame()
            return []

        # Define the function to run in parallel or sequentially
        def _run_comparison_wrapper(pair):
            mode1, mode2 = pair
            s1 = np.array(distributions[mode1])
            s2 = np.array(distributions[mode2])
            return self._run_comparison(
                mode1,
                mode2,
                s1,
                s2,
                n_tests,
                n_permutations=n_permutations,
                random_state=random_state,
            )

        # Run comparisons in parallel or sequentially
        if self.n_jobs > 1 and len(pairs) > 1:
            results = joblib.Parallel(n_jobs=self.n_jobs)(
                joblib.delayed(_run_comparison_wrapper)(pair) for pair in pairs
            )
        else:
            results = [_run_comparison_wrapper(pair) for pair in pairs]

        # Convert to DataFrame if requested
        if as_dataframe:
            return pd.DataFrame(results)

        return results
