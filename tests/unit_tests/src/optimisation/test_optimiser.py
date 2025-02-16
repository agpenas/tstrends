import pytest
from typing import Type
import numpy as np
from bayes_opt import acquisition
from optimisation import Optimiser
from trend_labelling import BinaryCTL, TernaryCTL
from returns_estimation.returns_estimation import BaseReturnEstimator


class MockReturnEstimator(BaseReturnEstimator):
    """Mock return estimator for testing."""

    def estimate_return(self, prices: list[float], labels: list[int]) -> float:
        # Simple mock implementation that returns sum of prices * labels
        return sum(p * l for p, l in zip(prices, labels))


@pytest.fixture
def mock_estimator():
    """Fixture providing a mock return estimator."""
    return MockReturnEstimator()


@pytest.fixture
def simple_optimiser(mock_estimator):
    """Fixture providing an Optimiser instance with minimal iterations for testing."""
    return Optimiser(
        returns_estimator=mock_estimator,
        initial_points=2,  # Small number for faster tests
        nb_iter=3,  # Small number for faster tests
    )


@pytest.fixture
def sample_time_series():
    """Fixture providing a sample time series for testing."""
    return [100.0, 101.0, 99.0, 102.0, 103.0]


@pytest.fixture
def multiple_time_series():
    """Fixture providing multiple time series for testing."""
    return [
        [100.0, 101.0, 99.0],
        [102.0, 103.0, 101.0],
    ]


class TestOptimiser:
    """Tests for the Optimiser class."""

    def test_init(self, mock_estimator):
        """Test Optimiser initialization."""
        optimiser = Optimiser(mock_estimator)
        assert optimiser.returns_estimator == mock_estimator
        assert optimiser.initial_points == 10  # Default value
        assert optimiser.nb_iter == 1_000  # Default value

        custom_optimiser = Optimiser(mock_estimator, initial_points=5, nb_iter=100)
        assert custom_optimiser.initial_points == 5
        assert custom_optimiser.nb_iter == 100

    def test_process_parameters(self, simple_optimiser):
        """Test parameter processing."""
        # Test integer parameter conversion
        params = {
            "window_size": 3.7,  # Should be converted to int
            "omega": 0.5,  # Should remain float
        }
        processed = simple_optimiser._process_parameters(params)
        assert isinstance(processed["window_size"], int)
        assert processed["window_size"] == 3
        assert isinstance(processed["omega"], float)
        assert processed["omega"] == 0.5

    def test_single_time_series_optimization(
        self, simple_optimiser, sample_time_series
    ):
        """Test optimization with a single time series."""
        simple_optimiser.optimise(
            labeller_class=BinaryCTL, time_series_list=sample_time_series, verbose=0
        )

        # Check if optimization completed
        assert hasattr(simple_optimiser, "optimiser")
        assert "omega" in simple_optimiser.optimiser.max["params"]
        assert 0.0 <= simple_optimiser.optimiser.max["params"]["omega"] <= 1.0

    def test_multiple_time_series_optimization(
        self, simple_optimiser, multiple_time_series
    ):
        """Test optimization with multiple time series."""
        simple_optimiser.optimise(
            labeller_class=BinaryCTL, time_series_list=multiple_time_series, verbose=0
        )

        # Check if optimization completed
        assert hasattr(simple_optimiser, "optimiser")
        assert "omega" in simple_optimiser.optimiser.max["params"]
        assert 0.0 <= simple_optimiser.optimiser.max["params"]["omega"] <= 1.0

    def test_custom_bounds(self, simple_optimiser, sample_time_series):
        """Test optimization with custom bounds."""
        custom_bounds = {"omega": (0.3, 0.7)}
        simple_optimiser.optimise(
            labeller_class=BinaryCTL,
            time_series_list=sample_time_series,
            bounds=custom_bounds,
            verbose=0,
        )

        # Check if optimization respects custom bounds
        optimal_omega = simple_optimiser.optimiser.max["params"]["omega"]
        assert 0.3 <= optimal_omega <= 0.7

    def test_custom_acquisition_function(self, simple_optimiser, sample_time_series):
        """Test optimization with custom acquisition function."""
        custom_acq = acquisition.ExpectedImprovement(
            xi=0.2
        )  # Different acquisition function
        simple_optimiser.optimise(
            labeller_class=BinaryCTL,
            time_series_list=sample_time_series,
            acquisition_function=custom_acq,
            verbose=0,
        )

        # Check if optimization completed
        assert simple_optimiser.optimiser.acquisition_function == custom_acq
        assert hasattr(simple_optimiser, "optimiser")
        assert simple_optimiser.optimiser.max["params"]["omega"] is not None

    def test_ternary_optimization(self, simple_optimiser, sample_time_series):
        """Test optimization with TernaryCTL labeller."""
        simple_optimiser.optimise(
            labeller_class=TernaryCTL, time_series_list=sample_time_series, verbose=0
        )

        # Check if optimization completed with correct parameters
        result_params = simple_optimiser.optimiser.max["params"]
        assert "marginal_change_thres" in result_params
        assert "window_size" in result_params
        assert 0.0 <= result_params["marginal_change_thres"] <= 0.1
        assert 1 <= int(result_params["window_size"]) <= 5000

    @pytest.mark.parametrize(
        "invalid_series",
        [
            [],  # Empty list
            [[]],  # Empty nested list
            [1, 2, "3"],  # Invalid type
            [[1, 2], 3],  # Mixed types
        ],
        ids=[
            "empty_list",
            "empty_nested_list",
            "invalid_type",
            "mixed_types",
        ],
    )
    def test_invalid_time_series(self, simple_optimiser, invalid_series):
        """Test handling of invalid time series inputs."""
        with pytest.raises((ValueError, IndexError, TypeError)):
            simple_optimiser.optimise(
                labeller_class=BinaryCTL, time_series_list=invalid_series, verbose=0
            )
