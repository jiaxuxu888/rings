import pytest
import numpy as np
from rings.complementarity.comparator import MatrixNormComparator


class TestMatrixNormComparator:
    def test_init_default(self):
        """Test initialization with default parameters."""
        comparator = MatrixNormComparator()
        assert comparator.norm == "L11"

    def test_init_custom(self):
        """Test initialization with custom norm parameter."""
        comparator = MatrixNormComparator(norm="frobenius")
        assert comparator.norm == "frobenius"

    def test_invalid_norm(self):
        """Test that using an invalid norm raises RuntimeError."""
        comparator = MatrixNormComparator(norm="invalid")
        D_X = np.array([[0, 1], [1, 0]])
        D_G = np.array([[0, 2], [2, 0]])

        with pytest.raises(RuntimeError) as excinfo:
            comparator(D_X, D_G)
        assert "Unexpected norm 'invalid'" in str(excinfo.value)

    def test_l11_norm(self):
        """Test the L11 norm calculation."""
        M = np.array([[-1, 2], [3, -4]])
        result = MatrixNormComparator.L11_norm(M)
        # Expected: sum(abs([-1, 2, 3, -4])) = 1 + 2 + 3 + 4 = 10
        assert result == 10.0

    def test_frobenius_norm(self):
        """Test the Frobenius norm calculation."""
        M = np.array([[1, 2], [3, 4]])
        result = MatrixNormComparator.frobenius_norm(M)
        # sqrt(1^2 + 2^2 + 3^2 + 4^2) = sqrt(1 + 4 + 9 + 16) = sqrt(30)
        assert np.isclose(result, np.sqrt(30))

    def test_invalid_data_property(self):
        """Test the invalid_data property."""
        comparator = MatrixNormComparator()
        result = comparator.invalid_data
        assert "complementarity" in result
        assert np.isnan(result["complementarity"])

    def test_call_with_l11_norm(self):
        """Test the __call__ method with L11 norm."""
        comparator = MatrixNormComparator(norm="L11")
        D_X = np.array([[0, 1], [1, 0]])
        D_G = np.array([[0, 2], [2, 0]])
        result = comparator(D_X, D_G)

        # Expected: sum(abs([0-0, 1-2], [1-2, 0-0])) / (2²-2) = sum(abs([0, -1], [-1, 0])) / 2 = 2/2 = 1.0
        assert "complementarity" in result
        assert np.isclose(result["complementarity"], 1.0)

    def test_call_with_frobenius_norm(self):
        """Test the __call__ method with Frobenius norm."""
        comparator = MatrixNormComparator(norm="frobenius")
        D_X = np.array([[0, 1], [1, 0]])
        D_G = np.array([[0, 2], [2, 0]])
        result = comparator(D_X, D_G)

        # Expected: ||[0-0, 1-2], [1-2, 0-0]||_F / sqrt(2²-2) = ||[0, -1], [-1, 0]||_F / sqrt(2) = sqrt(2) / sqrt(2) = 1.0
        assert "complementarity" in result
        assert np.isclose(result["complementarity"], 1.0)

    def test_single_element_matrices(self):
        """Test calculation with single element matrices."""
        comparator = MatrixNormComparator()
        D_X = np.array([[5]])
        D_G = np.array([[2]])
        result = comparator(D_X, D_G)

        # Expected: |5-2| / 1 = 3
        assert np.isclose(result["complementarity"], 3.0)

    def test_larger_matrices(self):
        """Test calculation with larger matrices."""
        comparator = MatrixNormComparator()
        D_X = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
        D_G = np.array([[0, 2, 4], [2, 0, 6], [4, 6, 0]])
        result = comparator(D_X, D_G)

        # The sum of absolute differences is 12, and divisor is 3²-3 = 6
        expected = 12 / 6
        assert np.isclose(result["complementarity"], expected)
