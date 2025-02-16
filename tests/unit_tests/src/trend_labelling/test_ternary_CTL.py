import pytest
import numpy as np
from trend_labelling.ternary_CTL import TernaryCTL
from trend_labelling.label_scaling import Labels


@pytest.fixture
def labeller():
    return TernaryCTL(marginal_change_thres=0.1, window_size=3)


class TestTernaryCTL:

    @pytest.mark.parametrize(
        "invalid_params,error_type,error_match",
        [
            (
                {"marginal_change_thres": 1, "window_size": 3},
                TypeError,
                "marginal_change_thres must be a float.",
            ),
            (
                {"marginal_change_thres": 0.1, "window_size": 3.2},
                TypeError,
                "window_size must be an integer.",
            ),
        ],
    )
    def test_initialization_invalid_type(self, invalid_params, error_type, error_match):
        """Test initialization with invalid parameter types."""
        with pytest.raises(error_type, match=error_match):
            TernaryCTL(**invalid_params)

    @pytest.mark.parametrize(
        "time_series,expected_labels,test_name",
        [
            (
                [1.0, 1.15, 1.2, 1.25],  # Each step > 10% increase from initial
                [Labels.UP, Labels.UP, Labels.UP, Labels.UP],
                "strong_uptrend",
            ),
            (
                [1.0, 0.85, 0.8, 0.75],  # Each step > 10% decrease from initial
                [Labels.DOWN, Labels.DOWN, Labels.DOWN, Labels.DOWN],
                "strong_downtrend",
            ),
        ],
        ids=lambda x: x[2] if isinstance(x, tuple) else str(x),
    )
    def test_simple_trends(self, labeller, time_series, expected_labels, test_name):
        """Test labelling of simple trends."""
        labels = labeller.get_labels(time_series, return_labels_as_int=False)
        assert labels == expected_labels

    @pytest.mark.parametrize(
        "time_series,expected_labels",
        [
            (
                [1.0, 1.2, 1.3, 1.0, 0.8],  # Clear trend changes > 10%
                [
                    Labels.UP,
                    Labels.UP,
                    Labels.UP,
                    Labels.DOWN,
                    Labels.DOWN,
                ],
            ),
            (
                [
                    1.0,
                    1.05,
                    1.08,
                    1.0,
                    1.1,
                    0.85,
                    0.75,
                ],  # Bumpy then sharp trend change
                [
                    Labels.UP,
                    Labels.UP,
                    Labels.UP,
                    Labels.UP,
                    Labels.UP,
                    Labels.DOWN,
                    Labels.DOWN,
                ],
            ),
            (
                [
                    1.0,
                    0.95,
                    0.98,
                    0.93,
                    0.90,
                    0.95,
                    1.0,
                ],  # General bumpy downwards trend then upwards
                [
                    Labels.DOWN,
                    Labels.DOWN,
                    Labels.DOWN,
                    Labels.DOWN,
                    Labels.DOWN,
                    Labels.UP,
                    Labels.UP,
                ],
            ),
        ],
        ids=["trend_reversal", "general_upwards_trend", "general_downwards_trend"],
    )
    def test_trend_transitions(self, labeller, time_series, expected_labels):
        """Test labelling when trends transition between states."""
        labels = labeller.get_labels(time_series, return_labels_as_int=False)
        assert labels == expected_labels

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
