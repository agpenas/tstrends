"""Unit tests for the smoothing implementations in the label_tuning module."""

import numpy as np
import pytest
from scipy import signal

from tstrends.label_tuning.smoothing import SimpleMovingAverage, LinearWeightedAverage
from tstrends.label_tuning.smoothing_direction import Direction


class TestSimpleMovingAverage:
    """Tests for the SimpleMovingAverage class."""

    @pytest.fixture
    def valid_values(self):
        """Fixture providing valid values for smoothing."""
        return [0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]

    def test_left_smoothing(self, valid_values):
        """Test left-sided smoothing."""
        smoother = SimpleMovingAverage(window_size=3, direction="left")
        result = smoother.smooth(valid_values)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(valid_values)

        expected = [
            0.333,
            0.667,
            1.0,
            1.333,
            1.667,
            2.0,
            2.0,
            1.333,
            0.667,
        ]
        assert np.allclose(result, expected, atol=0.001)

    def test_centered_smoothing(self, valid_values):
        """Test centered smoothing."""
        smoother = SimpleMovingAverage(window_size=3, direction="centered")
        result = smoother.smooth(valid_values)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(valid_values)

        expected = [0.0, 0.333, 0.667, 1.0, 1.333, 1.667, 2.0, 2.0, 1.333]
        assert np.allclose(result, expected, atol=0.001)


class TestLinearWeightedAverage:
    """Tests for the LinearWeightedAverage class."""

    @pytest.fixture
    def valid_values(self):
        """Fixture providing valid values for smoothing."""
        return [0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]

    def test_left_smoothing(self, valid_values):
        """Test left-sided smoothing with linear weights."""
        smoother = LinearWeightedAverage(window_size=3, direction="left")
        result = smoother.smooth(valid_values)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(valid_values)

        expected = [0.5, 0.833, 1.0, 1.5, 1.833, 2.0, 2.0, 2.0, 2.0]
        assert np.allclose(result, expected, atol=0.001)

    def test_centered_smoothing(self, valid_values):
        """Test centered smoothing with linear weights."""
        smoother = LinearWeightedAverage(window_size=3, direction="centered")
        result = smoother.smooth(valid_values)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(valid_values)

        expected = [0.0, 0.25, 0.75, 1.0, 1.25, 1.75, 2.0, 2.0, 2.0]
        assert np.allclose(result, expected, atol=0.001)

    @pytest.mark.parametrize(
        "test_values",
        [
            [0.0] * 5,  # All zeros
            [-1.0, -2.0, -3.0, -2.0, -1.0],  # Negative values
            [1e6, 2e6, 3e6, 4e6, 5e6],  # Large values
        ],
    )
    def test_special_value_cases(self, test_values):
        """Test smoothing with special value cases."""
        smoother = LinearWeightedAverage(window_size=3)
        result = smoother.smooth(test_values)
        assert len(result) == len(test_values)
        assert not np.any(np.isnan(result))  # No NaN values
        assert not np.any(np.isinf(result))  # No infinite values
