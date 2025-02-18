import numpy as np
import pytest

from time_series_trends.trend_labelling.binary_CTL import BinaryCTL
from time_series_trends.trend_labelling.label_scaling import Labels


@pytest.fixture
def labeller():
    return BinaryCTL(omega=0.1)


class TestBinaryCTL:
    def test_initialization_valid(self):
        """Test valid initialization of BinaryCTL."""
        labeller = BinaryCTL(omega=0.1)
        assert labeller.omega == 0.1

    @pytest.mark.parametrize(
        "invalid_omega,error_type,error_match",
        [
            ("0.1", TypeError, "omega must be a float."),
            ([0.1], TypeError, "omega must be a float."),
            ({"omega": 0.1}, TypeError, "omega must be a float."),
        ],
    )
    def test_initialization_invalid_type(self, invalid_omega, error_type, error_match):
        """Test initialization with invalid omega types."""
        with pytest.raises(error_type, match=error_match):
            BinaryCTL(omega=invalid_omega)

    @pytest.mark.parametrize(
        "time_series,expected_label,test_name",
        [
            ([1.0, 1.05, 1.15, 1.2], Labels.UP.value, "uptrend"),
            ([1.0, 0.95, 0.85, 0.8], Labels.DOWN.value, "downtrend"),
        ],
        ids=lambda x: x[2] if isinstance(x, tuple) else str(x),
    )
    def test_simple_trends(self, labeller, time_series, expected_label, test_name):
        """Test labelling of simple trends."""
        labels = labeller.get_labels(time_series)
        assert all(label == expected_label for label in labels)

    @pytest.mark.parametrize(
        "time_series,expected_labels",
        [
            (
                [1.0, 1.15, 1.2, 1.0],
                [Labels.UP, Labels.UP, Labels.UP, Labels.DOWN],
            ),
            (
                [1.0, 1.15, 1.2, 1.0, 0.85, 0.95, 1.1],
                [
                    Labels.UP,
                    Labels.UP,
                    Labels.UP,
                    Labels.DOWN,
                    Labels.DOWN,
                    Labels.UP,
                    Labels.UP,
                ],
            ),
        ],
        ids=["simple_reversal", "complex_sequence"],
    )
    def test_trend_transitions(self, labeller, time_series, expected_labels):
        """Test labelling when trends transition."""
        labels = labeller.get_labels(time_series, return_labels_as_int=False)
        assert labels == expected_labels

    def test_no_clear_trend(self, labeller):
        """Test labelling when there's no clear trend (small fluctuations)."""
        time_series = [
            1.0,
            1.01,
            0.99,
            1.02,
        ]  # Small fluctuations below omega threshold
        labels = labeller.get_labels(time_series)
        assert any(label == Labels.NEUTRAL for label in labels)

    @pytest.mark.parametrize(
        "return_labels_as_int,expected_type",
        [
            (True, int),
            (False, Labels),
        ],
    )
    def test_labels_output_type(self, labeller, return_labels_as_int, expected_type):
        """Test different output types for labels."""
        time_series = [1.0, 1.15, 1.2, 1.0]
        labels = labeller.get_labels(
            time_series, return_labels_as_int=return_labels_as_int
        )
        assert all(isinstance(label, expected_type) for label in labels)

    @pytest.mark.parametrize(
        "invalid_input,error_type,error_match",
        [
            ([], ValueError, "time_series_list must contain at least two elements."),
            ([1.0], ValueError, "time_series_list must contain at least two elements."),
            (
                [1.0, "2.0", 3.0],
                TypeError,
                "All elements in time_series_list must be integers or floats.",
            ),
            (
                [1.0, np.nan, 3.0],
                TypeError,
                "time_series_list cannot contain NaN values.",
            ),
        ],
    )
    def test_invalid_inputs(self, labeller, invalid_input, error_type, error_match):
        """Test various invalid inputs."""
        with pytest.raises(error_type, match=error_match):
            labeller.get_labels(invalid_input)
