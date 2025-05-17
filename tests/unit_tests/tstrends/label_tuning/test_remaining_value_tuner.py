"""Unit tests for the remaining value tuner."""

import numpy as np
import pytest

from tstrends.label_tuning.remaining_value_tuner import RemainingValueTuner
from tstrends.label_tuning.smoothing import SimpleMovingAverage


class TestRemainingValueTuner:
    """Tests for the RemainingValueTuner class."""

    @pytest.fixture
    def tuner(self):
        """Fixture providing a RemainingValueTuner instance."""
        return RemainingValueTuner()

    @pytest.fixture
    def uptrend_series(self):
        """Fixture providing an uptrend time series and labels."""
        return {
            "prices": [100.0, 101.0, 102.0, 103.0, 104.0],
            "labels": [1, 1, 1, 1, 1],
        }

    @pytest.fixture
    def downtrend_series(self):
        """Fixture providing a downtrend time series and labels."""
        return {
            "prices": [100.0, 98.0, 96.0, 94.0, 92.0],
            "labels": [-1, -1, -1, -1, -1],
        }

    @pytest.fixture
    def non_monotonic_uptrend_series(self):
        """Fixture providing a non-monotonic uptrend time series and labels."""
        return {
            "prices": [100.0, 101.0, 102.0, 103.0, 104.0, 103.0, 107.0, 108.0],
            "labels": [1, 1, 1, 1, 1, 1, 1, 1],
        }

    @pytest.fixture
    def non_monotonic_downtrend_series(self):
        """Fixture providing a non-monotonic downtrend time series and labels."""
        return {
            "prices": [100.0, 98.0, 96.0, 94.0, 95.0, 97.0, 89.0, 88.0],
            "labels": [-1, -1, -1, -1, -1, -1, -1, -1],
        }

    @pytest.fixture
    def mixed_trend_series(self):
        """Fixture providing a mixed trend time series and labels."""
        return {
            "prices": [100.0, 102.0, 101.0, 100.0, 99.0, 101.0, 102.0, 104.0],
            "labels": [1, 1, -1, -1, -1, 1, 1, 1],
        }

    @pytest.mark.parametrize(
        "test_series,expected",
        [
            ("uptrend_series", [4.0, 3.0, 2.0, 1.0, 0.0]),
            ("downtrend_series", [-8.0, -6.0, -4.0, -2.0, 0.0]),
            ("mixed_trend_series", [2.0, 0.0, -2.0, -1.0, 0.0, 3.0, 2.0, 0.0]),
        ],
    )
    def test_basic_trends(self, tuner, test_series, expected, request):
        """Test tuning with different trend patterns."""
        series = request.getfixturevalue(test_series)
        result = tuner.tune(series["prices"], series["labels"])
        assert np.allclose(result, expected)

    @pytest.mark.parametrize(
        "test_series,expected",
        [
            (
                "non_monotonic_uptrend_series",
                [8.0, 7.0, 6.0, 5.0, 4.0, 4.0, 1.0, 0.0],
            ),
            (
                "non_monotonic_downtrend_series",
                [-12.0, -10.0, -8.0, -6.0, -6.0, -6.0, -1.0, 0.0],
            ),
        ],
    )
    def test_enforce_monotonicity(self, tuner, test_series, expected, request):
        """Test tuning with monotonicity enforcement for up and down trends."""
        series = request.getfixturevalue(test_series)
        result = tuner.tune(
            series["prices"],
            series["labels"],
            enforce_monotonicity=True,
        )
        assert np.allclose(result, expected)

    @pytest.mark.parametrize(
        "test_series,expected",
        [
            (
                "uptrend_series",
                [1.0, 0.75, 0.5, 0.25, 0.0],
            ),
            (
                "downtrend_series",
                [-1.0, -0.75, -0.5, -0.25, 0.0],
            ),
            (
                "mixed_trend_series",
                [1.0, 0.0, -1.0, -0.5, 0.0, 1.0, 0.667, 0.0],
            ),
        ],
    )
    def test_normalize_over_interval(self, tuner, test_series, expected, request):
        """Test tuning with interval normalization."""
        series = request.getfixturevalue(test_series)
        result = tuner.tune(
            series["prices"],
            series["labels"],
            normalize_over_interval=True,
        )
        assert np.allclose(result, expected, rtol=1e-3)

    @pytest.mark.parametrize(
        "test_series,shift,expected",
        [
            (
                "uptrend_series",
                2,
                [0.0, 0.0, 4.0, 3.0, 2.0],
            ),
            (
                "downtrend_series",
                1,
                [0.0, -8.0, -6.0, -4.0, -2.0],
            ),
            (
                "mixed_trend_series",
                3,
                [0.0, 0.0, 0.0, 2.0, 0.0, -2.0, -1.0, 0.0],
            ),
        ],
    )
    def test_shift_periods(self, tuner, test_series, shift, expected, request):
        """Test tuning with period shifting."""
        series = request.getfixturevalue(test_series)
        result = tuner.tune(series["prices"], series["labels"], shift_periods=shift)
        assert np.allclose(result, expected)

    @pytest.mark.parametrize(
        "test_series,window_size,expected",
        [
            (
                "uptrend_series",
                3,
                [3.0, 2.0, 1.0, 0.333333, 0.0],
            ),
            (
                "downtrend_series",
                3,
                [-6.0, -4.0, -2.0, -0.67, 0.0],
            ),
        ],
    )
    def test_with_smoother(self, tuner, test_series, window_size, expected, request):
        """Test tuning with smoothing applied."""
        series = request.getfixturevalue(test_series)
        smoother = SimpleMovingAverage(window_size=window_size)
        result = tuner.tune(series["prices"], series["labels"], smoother=smoother)
        # Use atol to handle small floating-point differences
        assert np.allclose(result, expected, rtol=1e-2, atol=1e-10)

    @pytest.mark.parametrize(
        "prices,labels,error_type",
        [
            ([], [1, 2, 3], ValueError),  # Empty time series
            ([1, 2, 3], [], ValueError),  # Empty labels
            ([1, 2], [1, 2, 3], ValueError),  # Mismatched lengths
            ([1, "2", 3], [1, 1, 1], TypeError),  # Invalid time series type
            ([1, 2, 3], [1, 2, 3], ValueError),  # Invalid label values (not -1, 0, 1)
        ],
    )
    def test_invalid_inputs(self, tuner, prices, labels, error_type):
        """Test that invalid inputs raise appropriate exceptions."""
        with pytest.raises(error_type):
            tuner.tune(prices, labels)

    @pytest.mark.parametrize(
        "labels,expected_intervals",
        [
            ([1, 1, 1], [0, 2]),  # Single trend
            ([1, 1, -1, -1, 1, 1, 0, 0], [0, 2, 4, 6, 7]),  # Multiple trends
            ([0, 0, 0], [0, 2]),  # Single flat trend
            (
                [1, -1, 1, -1],
                [0, 1, 2, 3, 3],
            ),  # Alternating trends - includes end index
        ],
    )
    def test_find_trend_intervals(self, tuner, labels, expected_intervals):
        """Test the trend interval finding helper method."""
        intervals = tuner._find_trend_intervals(labels)
        assert intervals == expected_intervals

    @pytest.mark.parametrize(
        "values,expected_normalized",
        [
            (
                np.array([2.0, 4.0, -6.0, 3.0]),
                np.array([1 / 3, 2 / 3, -1.0, 0.5]),
            ),  # Regular case
            (
                np.array([0.0, 0.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 0.0, 0.0]),
            ),  # All zeros
            (
                np.array([5.0]),
                np.array([1.0]),
            ),  # Single value
            (
                np.array([-3.0, -6.0, -1.5]),
                np.array([-0.5, -1.0, -0.25]),
            ),  # All negative
        ],
    )
    def test_normalize_values(self, tuner, values, expected_normalized):
        """Test the value normalization helper method."""
        normalized = tuner._normalize_values(values)
        assert np.allclose(normalized, expected_normalized)
