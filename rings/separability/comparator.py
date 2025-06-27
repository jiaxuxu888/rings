import numpy as np
from scipy.stats import ks_2samp, mannwhitneyu
from typing import Dict, List, Union, Optional
from tqdm import tqdm


class KSComparator:
    """
    Comparator that uses the Kolmogorov-Smirnov test to compare two distributions.

    Uses permutation testing with KS statistic to determine if two samples
    come from the same distribution.
    """

    def compare(
        self,
        s1: Union[List[float], np.ndarray],
        s2: Union[List[float], np.ndarray],
        n_permutations: int = 10_000,
        alpha: float = 0.01,
        n_hypotheses: int = 1,
        random_state: Optional[int] = 42,
    ) -> Dict:
        """
        Compare two samples using Kolmogorov-Smirnov test with permutation testing.

        Args:
            s1: First sample of measurements
            s2: Second sample of measurements
            n_permutations: Number of permutations for the test (default: 10,000)
            alpha: Significance level (default: 0.01)
            n_hypotheses: Number of hypotheses being tested (for Bonferroni correction)
            random_state: Seed for reproducibility

        Returns:
            Dictionary containing:
                - observed_statistic: The KS statistic observed in the original data
                - pvalue: The p-value from the permutation test
                - pvalue_adjusted: The Bonferroni-corrected p-value
                - significant: Whether the adjusted p-value is less than alpha
        """
        # Convert inputs to numpy arrays
        s1 = np.array(s1)
        s2 = np.array(s2)

        # Initialize RNG
        rng = np.random.RandomState(random_state)

        # Calculate observed statistic
        observed_result = ks_2samp(s1, s2)
        observed_statistic = observed_result.statistic

        # Prepare for permutation test
        combined = np.concatenate((s1, s2))
        n1 = len(s1)
        n2 = len(s2)

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
            "observed_statistic": observed_statistic,
            "pvalue": pvalue,
            "pvalue_adjusted": pvalue_adjusted,
            "significant": pvalue_adjusted < alpha,
        }


class WilcoxonComparator:
    """
    Comparator that uses the Mann-Whitney U test (equivalent to Wilcoxon rank-sum)
    to compare two distributions.

    Uses permutation testing with Mann-Whitney U statistic to determine if
    two samples come from the same distribution.
    """

    def compare(
        self,
        s1: Union[List[float], np.ndarray],
        s2: Union[List[float], np.ndarray],
        n_permutations: int = 10_000,
        alpha: float = 0.01,
        n_hypotheses: int = 1,
        random_state: Optional[int] = 42,
    ) -> Dict:
        """
        Compare two samples using Mann-Whitney U test with permutation testing.

        Args:
            s1: First sample of measurements
            s2: Second sample of measurements
            n_permutations: Number of permutations for the test (default: 10,000)
            alpha: Significance level (default: 0.01)
            n_hypotheses: Number of hypotheses being tested (for Bonferroni correction)
            random_state: Seed for reproducibility

        Returns:
            Dictionary containing:
                - observed_statistic: The U statistic observed in the original data
                - pvalue: The p-value from the permutation test
                - pvalue_adjusted: The Bonferroni-corrected p-value
                - significant: Whether the adjusted p-value is less than alpha
        """
        # Convert inputs to numpy arrays
        s1 = np.array(s1)
        s2 = np.array(s2)

        # Initialize RNG
        rng = np.random.RandomState(random_state)

        # Calculate observed statistic
        observed_result = mannwhitneyu(s1, s2, alternative="two-sided")
        observed_statistic = observed_result.statistic

        # Prepare for permutation test
        combined = np.concatenate((s1, s2))
        n1 = len(s1)
        n2 = len(s2)

        # Run permutation test
        permuted_statistics = np.zeros(n_permutations)

        for i in tqdm(range(n_permutations), desc="Wilcoxon Permutation Test"):
            # Shuffle the combined data
            shuffled = rng.permutation(combined)

            # Split into two groups of original sizes
            perm_s1 = shuffled[:n1]
            perm_s2 = shuffled[n1:]

            # Compute statistic on permuted data
            perm_result = mannwhitneyu(
                perm_s1, perm_s2, alternative="two-sided"
            )
            permuted_statistics[i] = perm_result.statistic

        # Calculate p-value (two-sided)
        # For Mann-Whitney U, the direction matters, so we compare absolute deviations
        # from the mean of the permutation distribution
        mean_perm_stat = np.mean(permuted_statistics)
        observed_dev = np.abs(observed_statistic - mean_perm_stat)
        perm_devs = np.abs(permuted_statistics - mean_perm_stat)

        pvalue = np.mean(perm_devs >= observed_dev)

        # Apply Bonferroni correction
        pvalue_adjusted = min(1.0, pvalue * n_hypotheses)

        return {
            "observed_statistic": observed_statistic,
            "pvalue": pvalue,
            "pvalue_adjusted": pvalue_adjusted,
            "significant": pvalue_adjusted < alpha,
        }
