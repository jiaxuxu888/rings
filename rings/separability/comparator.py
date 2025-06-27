import numpy as np
from scipy.stats import ks_2samp, wilcoxon
from typing import Dict, List, Union, Optional
from tqdm import tqdm


class KSComparator:
    """
    Comparator that uses the Kolmogorov-Smirnov test to compare two distributions.

    Uses permutation testing with KS statistic to determine if two samples
    come from the same distribution.

    Methods
    -------
    __call__(x, y, **kwargs)
        Compare two samples using Kolmogorov-Smirnov test with permutation testing.
    """

    def __call__(
        self,
        x: Union[List[float], np.ndarray],
        y: Union[List[float], np.ndarray],
        n_permutations: int = 10_000,
        alpha: float = 0.01,
        n_hypotheses: int = 1,
        random_state: Optional[int] = 42,
        **kwargs,
    ) -> Dict:
        """
        Compare two samples using Kolmogorov-Smirnov test with permutation testing.

        Parameters
        ----------
        x : Union[List[float], np.ndarray]
            First sample of measurements
        y : Union[List[float], np.ndarray]
            Second sample of measurements
        n_permutations : int, optional
            Number of permutations for the test (default: 10,000)
        alpha : float, optional
            Significance level (default: 0.01)
        n_hypotheses : int, optional
            Number of hypotheses being tested (for Bonferroni correction)
        random_state : Optional[int], optional
            Seed for reproducibility
        **kwargs : dict, optional
            Additional keyword arguments (not used)

        Returns
        -------
        dict
            Dictionary with keys:
            - score: The KS statistic observed in the original data
            - pvalue: The p-value from the permutation test
            - pvalue_adjusted: The Bonferroni-corrected p-value
            - significant: Whether the adjusted p-value is less than alpha
            - method: "KS" (identifier for this test)
        """
        # Convert inputs to numpy arrays
        x = np.array(x)
        y = np.array(y)

        # Initialize RNG
        rng = np.random.RandomState(random_state)

        # Calculate observed statistic
        observed_result = ks_2samp(x, y)
        observed_statistic = observed_result.statistic

        # Prepare for permutation test
        combined = np.concatenate((x, y))
        n1 = len(x)
        n2 = len(y)

        # Run permutation test
        permuted_statistics = np.zeros(n_permutations)

        for i in tqdm(range(n_permutations), desc="KS Permutation Test"):
            # Shuffle the combined data
            shuffled = rng.permutation(combined)

            # Split into two groups of original sizes
            perm_s1 = shuffled[:n1]
            perm_s2 = shuffled[n1:]

            # Compute statistic on permuted data
            perm_result = ks_2samp(perm_s1, perm_s2)
            permuted_statistics[i] = perm_result.statistic

        # Calculate p-value (two-sided)
        pvalue = np.mean(permuted_statistics >= observed_statistic)

        # Apply Bonferroni correction
        pvalue_adjusted = min(1.0, pvalue * n_hypotheses)

        return {
            "score": observed_statistic,
            "pvalue": pvalue,
            "pvalue_adjusted": pvalue_adjusted,
            "significant": pvalue_adjusted < alpha,
            "method": "KS",
        }


class WilcoxonComparator:
    """
    Comparator that uses the Wilcoxon signed-rank test to compare two distributions.

    Uses permutation testing with Wilcoxon statistic to determine if
    two samples come from the same distribution.

    Methods
    -------
    __call__(x, y, **kwargs)
        Compare two samples using Wilcoxon signed-rank test with permutation testing.
    """

    def __call__(
        self,
        x: Union[List[float], np.ndarray],
        y: Union[List[float], np.ndarray],
        n_permutations: int = 10_000,
        alpha: float = 0.01,
        n_hypotheses: int = 1,
        random_state: Optional[int] = 42,
        **kwargs,
    ) -> Dict:
        """
        Compare two samples using Wilcoxon signed-rank test with permutation testing.

        Parameters
        ----------
        x : Union[List[float], np.ndarray]
            First sample of measurements
        y : Union[List[float], np.ndarray]
            Second sample of measurements
        n_permutations : int, optional
            Number of permutations for the test (default: 10,000)
        alpha : float, optional
            Significance level (default: 0.01)
        n_hypotheses : int, optional
            Number of hypotheses being tested (for Bonferroni correction)
        random_state : Optional[int], optional
            Seed for reproducibility
        **kwargs : dict, optional
            Additional keyword arguments (not used)

        Returns
        -------
        dict
            Dictionary with keys:
            - score: The Wilcoxon statistic observed in the original data
            - pvalue: The p-value from the permutation test
            - pvalue_adjusted: The Bonferroni-corrected p-value
            - significant: Whether the adjusted p-value is less than alpha
            - method: "Wilcoxon" (identifier for this test)
        """
        # Convert inputs to numpy arrays
        x = np.array(x)
        y = np.array(y)

        # Initialize RNG
        rng = np.random.RandomState(random_state)

        # Verify arrays have the same length for Wilcoxon paired test
        if len(x) != len(y):
            raise ValueError(
                "Wilcoxon signed-rank test requires samples of equal length"
            )

        # Calculate observed statistic
        observed_result = wilcoxon(x, y, alternative="two-sided")
        observed_statistic = observed_result.statistic

        # Prepare for permutation test
        combined = np.vstack((x, y)).T  # Paired data
        n_pairs = len(combined)

        # Run permutation test
        permuted_statistics = np.zeros(n_permutations)

        for i in tqdm(range(n_permutations), desc="Wilcoxon Permutation Test"):
            # For each pair, randomly swap or don't swap the elements
            perm_combined = combined.copy()
            swap_indices = rng.choice([True, False], size=n_pairs)

            if swap_indices.any():
                (
                    perm_combined[swap_indices, 0],
                    perm_combined[swap_indices, 1],
                ) = (
                    perm_combined[swap_indices, 1],
                    perm_combined[swap_indices, 0],
                )

            # Split into two groups
            perm_s1 = perm_combined[:, 0]
            perm_s2 = perm_combined[:, 1]

            # Compute statistic on permuted data
            perm_result = wilcoxon(perm_s1, perm_s2, alternative="two-sided")
            permuted_statistics[i] = perm_result.statistic

        # Calculate p-value (two-sided)
        # For Wilcoxon, the direction matters, so we compare absolute deviations
        # from the mean of the permutation distribution
        mean_perm_stat = np.mean(permuted_statistics)
        observed_dev = np.abs(observed_statistic - mean_perm_stat)
        perm_devs = np.abs(permuted_statistics - mean_perm_stat)

        pvalue = np.mean(perm_devs >= observed_dev)

        # Apply Bonferroni correction
        pvalue_adjusted = min(1.0, pvalue * n_hypotheses)

        return {
            "score": observed_statistic,
            "pvalue": pvalue,
            "pvalue_adjusted": pvalue_adjusted,
            "significant": pvalue_adjusted < alpha,
            "method": "Wilcoxon",
        }
