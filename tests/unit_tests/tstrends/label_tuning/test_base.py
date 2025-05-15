"""Unit tests for the base classes in the label_tuning module."""

import numpy as np
import pytest

from tstrends.label_tuning.base import BaseLabelTuner, BaseSmoother
from tstrends.label_tuning.smoothing_direction import Direction


class MockLabelTuner(BaseLabelTuner):
    """Mock implementation of BaseLabelTuner for testing."""

    def tune(
        self, time_series: list[float], labels: list[int], **kwargs
    ) -> list[float]:
        self._verify_inputs(time_series, labels)
        # Simply return the labels as floats
        return [float(label) for label in labels]


class MockSmoother(BaseSmoother):
    """Mock implementation of BaseSmoother for testing."""

    def smooth(self, values: list[float]) -> np.ndarray:
        array = np.asarray(values)
        return array  # Simple mock that returns input unchanged


class TestBaseLabelTuner:
    """Tests for the BaseLabelTuner class."""

    @pytest.fixture
    def mock_tuner(self):
        """Fixture providing a mock label tuner instance."""
        return MockLabelTuner()

    @pytest.fixture
    def valid_time_series(self):
        """Fixture providing a valid time series."""
        return [100.0, 101.0, 99.0, 102.0, 103.0]

    @pytest.fixture
    def valid_labels(self):
        """Fixture providing valid labels."""
        return [1, 1, -1, 1, 1]

    def test_verify_inputs_valid(self, mock_tuner, valid_time_series, valid_labels):
        """Test input verification with valid inputs."""
        # Should not raise any exceptions
        mock_tuner._verify_inputs(valid_time_series, valid_labels)

        # Test with numpy arrays
        mock_tuner._verify_inputs(np.array(valid_time_series), np.array(valid_labels))

    @pytest.mark.parametrize(
        "invalid_time_series",
        [
            None,  # Not a list
            [],  # Empty list
            [1, "2", 3],  # Invalid type in list
            [1.0, 2.0],  # Wrong length
        ],
    )
    def test_verify_inputs_invalid_time_series(
        self, mock_tuner, invalid_time_series, valid_labels
    ):
        """Test input verification with invalid time series."""
        with pytest.raises((TypeError, ValueError)):
            mock_tuner._verify_inputs(invalid_time_series, valid_labels)

    @pytest.mark.parametrize(
        "invalid_labels",
        [
            None,  # Not a list
            [],  # Empty list
            [1, 2, 3],  # Invalid label values
            [-1, 1],  # Wrong length
        ],
    )
    def test_verify_inputs_invalid_labels(
        self, mock_tuner, valid_time_series, invalid_labels
    ):
        """Test input verification with invalid labels."""
        with pytest.raises((TypeError, ValueError)):
            mock_tuner._verify_inputs(valid_time_series, invalid_labels)


class TestBaseSmoother:
    """Tests for the BaseSmoother class."""

    @pytest.fixture
    def valid_values(self):
        """Fixture providing valid values for smoothing."""
        return [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_init_valid(self):
        """Test smoother initialization with valid parameters."""
        # Test with default parameters
        smoother = MockSmoother()
        assert smoother.window_size == 3
        assert smoother.direction == Direction.LEFT

        # Test with custom parameters
        smoother = MockSmoother(window_size=5, direction="centered")
        assert smoother.window_size == 5
        assert smoother.direction == Direction.CENTERED

        # Test with Direction enum
        smoother = MockSmoother(window_size=4, direction="left")
        assert smoother.window_size == 4
        assert smoother.direction == Direction.LEFT

    @pytest.mark.parametrize(
        "invalid_window_size",
        [
            1,  # Too small
            0,  # Zero
            -1,  # Negative
            2.5,  # Float
            "3",  # String
        ],
    )
    def test_init_invalid_window_size(self, invalid_window_size):
        """Test smoother initialization with invalid window sizes."""
        with pytest.raises(ValueError):
            MockSmoother(window_size=invalid_window_size)

    @pytest.mark.parametrize(
        "invalid_direction,expected_error",
        [
            ("right", ValueError),  # Invalid direction string
            ("invalid", ValueError),  # Invalid direction string
            (123, TypeError),  # Invalid type
            (None, TypeError),  # Invalid type
            (1.5, TypeError),  # Invalid type
        ],
    )
    def test_init_invalid_direction(self, invalid_direction, expected_error):
        """Test smoother initialization with invalid directions."""
        with pytest.raises(expected_error):
            MockSmoother(direction=invalid_direction)
