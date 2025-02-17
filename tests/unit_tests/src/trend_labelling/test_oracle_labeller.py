import pytest
import numpy as np
from trend_labelling.oracle_labeller import (
    OracleBinaryTrendLabeller,
    OracleTernaryTrendLabeller,
)
from trend_labelling.label_scaling import Labels


@pytest.fixture
def binary_labeller():
    return OracleBinaryTrendLabeller(transaction_cost=0.001)


@pytest.fixture
def ternary_labeller():
    return OracleTernaryTrendLabeller(transaction_cost=0.001, trend_coeff=0.5)


class TestOracleBinaryTrendLabeller:
    @pytest.mark.parametrize(
        "invalid_params,error_type,error_match",
        [
            (
                {"transaction_cost": 1},
                TypeError,
                "transaction_cost must be a float.",
            ),
        ],
    )
    def test_initialization_invalid_type(self, invalid_params, error_type, error_match):
        """Test initialization with invalid parameter types."""
        with pytest.raises(error_type, match=error_match):
            OracleBinaryTrendLabeller(**invalid_params)

    @pytest.mark.parametrize(
        "time_series,expected_labels,test_name",
        [
            (
                [1.0, 1.1, 1.2, 1.3],  # Clear uptrend
                [Labels.UP, Labels.UP, Labels.UP, Labels.UP],
                "strong_uptrend",
            ),
            (
                [1.0, 0.9, 0.8, 0.7],  # Clear downtrend
                [Labels.DOWN, Labels.DOWN, Labels.DOWN, Labels.DOWN],
                "strong_downtrend",
            ),
        ],
        ids=lambda x: x[2] if isinstance(x, tuple) else str(x),
    )
    def test_simple_trends(
        self, binary_labeller, time_series, expected_labels, test_name
    ):
        """Test labelling of simple trends."""
        labels = binary_labeller.get_labels(time_series, return_labels_as_int=False)
        assert labels == expected_labels

    @pytest.mark.parametrize(
        "time_series,expected_labels",
        [
            (
                [1.0, 1.1, 1.2, 1.0, 0.9],  # Trend reversal
                [
                    Labels.UP,
                    Labels.UP,
                    Labels.UP,
                    Labels.DOWN,
                    Labels.DOWN,
                ],
            ),
            (
                [1.0, 1.1, 1.0, 0.9, 1.1, 1.2],  # Multiple reversals
                [
                    Labels.UP,
                    Labels.UP,
                    Labels.DOWN,
                    Labels.UP,
                    Labels.UP,
                    Labels.UP,
                ],
            ),
        ],
        ids=["trend_reversal", "multiple_reversals"],
    )
    def test_trend_transitions(self, binary_labeller, time_series, expected_labels):
        """Test labelling when trends transition between states."""
        labels = binary_labeller.get_labels(time_series, return_labels_as_int=False)
        assert labels == expected_labels

    @pytest.mark.parametrize(
        "return_labels_as_int,expected_type",
        [
            (True, int),
            (False, Labels),
        ],
    )
    def test_labels_output_type(
        self, binary_labeller, return_labels_as_int, expected_type
    ):
        """Test different output types for labels."""
        time_series = [1.0, 1.1, 1.2, 1.0]
        labels = binary_labeller.get_labels(
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
                None,
                TypeError,
                "time_series_list must be a list.",
            ),
        ],
    )
    def test_invalid_inputs(
        self, binary_labeller, invalid_input, error_type, error_match
    ):
        """Test various invalid inputs."""
        with pytest.raises(error_type, match=error_match):
            binary_labeller.get_labels(invalid_input)


class TestOracleTernaryTrendLabeller:
    @pytest.mark.parametrize(
        "invalid_params,error_type,error_match",
        [
            (
                {"transaction_cost": 1, "trend_coeff": 0.5},
                TypeError,
                "transaction_cost must be a float.",
            ),
            (
                {"transaction_cost": 0.001, "trend_coeff": 1},
                TypeError,
                "trend_coeff must be a float.",
            ),
        ],
    )
    def test_initialization_invalid_type(self, invalid_params, error_type, error_match):
        """Test initialization with invalid parameter types."""
        with pytest.raises(error_type, match=error_match):
            OracleTernaryTrendLabeller(**invalid_params)

    @pytest.mark.parametrize(
        "time_series,expected_labels,test_name",
        [
            (
                [1.0, 1.1, 1.2, 1.3],  # Clear uptrend
                [Labels.UP, Labels.UP, Labels.UP, Labels.UP],
                "strong_uptrend",
            ),
            (
                [1.0, 0.9, 0.8, 0.7],  # Clear downtrend
                [Labels.DOWN, Labels.DOWN, Labels.DOWN, Labels.DOWN],
                "strong_downtrend",
            ),
            (
                [1.0, 1.01, 0.99, 1.05],  # Sideways movement
                [Labels.NEUTRAL, Labels.NEUTRAL, Labels.UP, Labels.UP],
                "sideways_to_up",
            ),
        ],
        ids=lambda x: x[2] if isinstance(x, tuple) else str(x),
    )
    def test_simple_trends(
        self, ternary_labeller, time_series, expected_labels, test_name
    ):
        """Test labelling of simple trends."""
        labels = ternary_labeller.get_labels(time_series, return_labels_as_int=False)
        assert labels == expected_labels

    @pytest.mark.parametrize(
        "time_series,expected_labels",
        [
            (
                [1.0, 1.1, 1.2, 1.0, 0.9],  # Through neutral to opposite trend
                [
                    Labels.NEUTRAL,
                    Labels.NEUTRAL,
                    Labels.DOWN,
                    Labels.DOWN,
                    Labels.DOWN,
                ],
            ),
            (
                [1.0, 1.1, 1.0, 0.9, 1.0, 1.1],  # Multiple state transitions
                [
                    Labels.NEUTRAL,
                    Labels.NEUTRAL,
                    Labels.NEUTRAL,
                    Labels.UP,
                    Labels.UP,
                    Labels.UP,
                ],
            ),
        ],
        ids=["trend_reversal_through_neutral", "multiple_transitions"],
    )
    def test_trend_transitions(self, ternary_labeller, time_series, expected_labels):
        """Test labelling when trends transition between states."""
        labels = ternary_labeller.get_labels(time_series, return_labels_as_int=False)
        assert labels == expected_labels

    def test_transaction_cost_impact(self, ternary_labeller):
        """Test that higher transaction costs lead to fewer transitions."""
        time_series = [1.0, 1.1, 1.0, 0.9, 1.0, 1.1]

        # Get labels with normal transaction cost
        normal_cost_labels = ternary_labeller.get_labels(
            time_series, return_labels_as_int=True
        )

        # Create new labeller with higher transaction cost
        high_cost_labeller = OracleTernaryTrendLabeller(
            transaction_cost=0.01, trend_coeff=0.5
        )
        high_cost_labels = high_cost_labeller.get_labels(
            time_series, return_labels_as_int=True
        )

        # Count transitions
        normal_transitions = sum(
            1
            for i in range(1, len(normal_cost_labels))
            if normal_cost_labels[i] != normal_cost_labels[i - 1]
        )
        high_cost_transitions = sum(
            1
            for i in range(1, len(high_cost_labels))
            if high_cost_labels[i] != high_cost_labels[i - 1]
        )

        assert high_cost_transitions <= normal_transitions

    @pytest.mark.parametrize(
        "return_labels_as_int,expected_type",
        [
            (True, int),
            (False, Labels),
        ],
    )
    def test_labels_output_type(
        self, ternary_labeller, return_labels_as_int, expected_type
    ):
        """Test different output types for labels."""
        time_series = [1.0, 1.1, 1.2, 1.0]
        labels = ternary_labeller.get_labels(
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
                None,
                TypeError,
                "time_series_list must be a list.",
            ),
        ],
    )
    def test_invalid_inputs(
        self, ternary_labeller, invalid_input, error_type, error_match
    ):
        """Test various invalid inputs."""
        with pytest.raises(error_type, match=error_match):
            ternary_labeller.get_labels(invalid_input)
