from typing import Type

import numpy as np
import pytest
from bayes_opt import acquisition

from optimization import Optimizer
from returns_estimation.returns_estimation import BaseReturnEstimator
from trend_labelling import BinaryCTL, TernaryCTL


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
def simple_optimizer(mock_estimator):
    """Fixture providing an Optimizer instance with minimal iterations for testing."""
    return Optimizer(
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


class TestOptimizer:
    """Tests for the Optimizer class."""

    def test_init(self, mock_estimator):
        """Test Optimizer initialization."""
        optimizer = Optimizer(mock_estimator)
        assert optimizer.returns_estimator == mock_estimator
        assert optimizer.initial_points == 10  # Default value
        assert optimizer.nb_iter == 1_000  # Default value
        assert optimizer._optimizer is None  # Should start as None

        custom_optimizer = Optimizer(mock_estimator, initial_points=5, nb_iter=100)
        assert custom_optimizer.initial_points == 5
        assert custom_optimizer.nb_iter == 100

    def test_get_optimizer_before_optimization(self, simple_optimizer):
        """Test get_optimizer raises error when called before optimization."""
        with pytest.raises(ValueError, match="Optimizer not initialized"):
            simple_optimizer.get_optimizer()

    def test_process_parameters(self, simple_optimizer):
        """Test parameter processing."""
        # Test integer parameter conversion
        params = {
            "window_size": 3.7,  # Should be converted to int
            "omega": 0.5,  # Should remain float
        }
        processed = simple_optimizer._process_parameters(params)
        assert isinstance(processed["window_size"], int)
        assert processed["window_size"] == 3
        assert isinstance(processed["omega"], float)
        assert processed["omega"] == 0.5

    def test_single_time_series_optimization(
        self, simple_optimizer, sample_time_series
    ):
        """Test optimization with a single time series."""
        result = simple_optimizer.optimize(
            labeller_class=BinaryCTL, time_series_list=sample_time_series, verbose=0
        )

        # Check result format and content
        assert isinstance(result, dict)
        assert "params" in result
        assert "target" in result
        assert isinstance(result["target"], float)
        assert "omega" in result["params"]
        assert 0.0 <= result["params"]["omega"] <= 1.0

        # Test get_optimizer after optimization
        optimizer = simple_optimizer.get_optimizer()
        assert optimizer is not None
        assert hasattr(optimizer, "max")

    def test_multiple_time_series_optimization(
        self, simple_optimizer, multiple_time_series
    ):
        """Test optimization with multiple time series."""
        result = simple_optimizer.optimize(
            labeller_class=BinaryCTL, time_series_list=multiple_time_series, verbose=0
        )

        # Check result format and content
        assert isinstance(result, dict)
        assert "params" in result
        assert "target" in result
        assert isinstance(result["target"], float)
        assert "omega" in result["params"]
        assert 0.0 <= result["params"]["omega"] <= 1.0

    def test_custom_bounds(self, simple_optimizer, sample_time_series):
        """Test optimization with custom bounds."""
        custom_bounds = {"omega": (0.3, 0.7)}
        result = simple_optimizer.optimize(
            labeller_class=BinaryCTL,
            time_series_list=sample_time_series,
            bounds=custom_bounds,
            verbose=0,
        )

        # Check if optimization respects custom bounds
        assert 0.3 <= result["params"]["omega"] <= 0.7

    def test_custom_acquisition_function(self, simple_optimizer, sample_time_series):
        """Test optimization with custom acquisition function."""
        custom_acq = acquisition.ExpectedImprovement(xi=0.2)
        result = simple_optimizer.optimize(
            labeller_class=BinaryCTL,
            time_series_list=sample_time_series,
            acquisition_function=custom_acq,
            verbose=0,
        )

        # Check if optimization completed successfully
        assert isinstance(result, dict)
        assert "params" in result
        assert "target" in result

        # Verify the acquisition function was used
        optimizer = simple_optimizer.get_optimizer()
        assert optimizer.acquisition_function == custom_acq

    def test_ternary_optimization(self, simple_optimizer, sample_time_series):
        """Test optimization with TernaryCTL labeller."""
        result = simple_optimizer.optimize(
            labeller_class=TernaryCTL, time_series_list=sample_time_series, verbose=0
        )

        # Check result format and content
        assert isinstance(result, dict)
        assert "params" in result
        assert "target" in result

        # Check specific parameters for TernaryCTL
        params = result["params"]
        assert "marginal_change_thres" in params
        assert "window_size" in params
        assert 0.0 <= params["marginal_change_thres"] <= 0.1
        assert 1 <= int(params["window_size"]) <= 5000

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
    def test_invalid_time_series(self, simple_optimizer, invalid_series):
        """Test handling of invalid time series inputs."""
        with pytest.raises((ValueError, IndexError, TypeError)):
            simple_optimizer.optimize(
                labeller_class=BinaryCTL, time_series_list=invalid_series, verbose=0
            )
