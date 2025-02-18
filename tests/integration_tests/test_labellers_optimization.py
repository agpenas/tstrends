"""Integration tests for trend labeller optimization and labelling."""

import json
from pathlib import Path

import numpy as np
import pytest

from optimization import OptimizationBounds, Optimizer
from returns_estimation.fees_config import FeesConfig
from returns_estimation.returns_estimation import ReturnsEstimatorWithFees
from trend_labelling import (
    BinaryCTL,
    OracleBinaryTrendLabeller,
    OracleTernaryTrendLabeller,
    TernaryCTL,
)

# Constants for floating point comparisons
PARAM_RTOL = 1e-5  # Relative tolerance for parameter comparisons
TARGET_RTOL = 1e-3  # Relative tolerance for target value comparisons


# ---- Test Data Loading Utilities ----


def load_optimization_results() -> dict:
    results_path = Path(__file__).parent / "optimization_results.json"
    with open(results_path) as f:
        return json.load(f)


def load_expected_labels(labeller_name: str, length: int) -> list[int]:
    labels = np.zeros(length, dtype=int)
    data_path = (
        Path(__file__).parent.parent / "data" / f"{labeller_name}_labels_intervals.csv"
    )
    with open(data_path) as f:
        intervals = [
            (int(l), int(i)) for l, i in [line.split() for line in f if line.strip()]
        ]
        for i, (label, start) in enumerate(intervals):
            end = intervals[i + 1][1] if i < len(intervals) - 1 else length
            labels[start:end] = label
    return labels.tolist()


# ---- Test Fixtures ----


@pytest.fixture
def fees_config():
    return FeesConfig(
        lp_transaction_fees=0.01,
        sp_transaction_fees=0.02,
        lp_holding_fees=0.0001,
        sp_holding_fees=0.0002,
    )


@pytest.fixture
def optimizer(fees_config):
    return Optimizer(
        returns_estimator=ReturnsEstimatorWithFees(fees_config),
        initial_points=5,  # Small number for faster tests
        nb_iter=10,  # Small number for faster tests
        random_state=42,  # Fixed random state for reproducibility
    )


@pytest.fixture
def bounds():
    return OptimizationBounds()


@pytest.fixture
def sample_prices():
    data_path = Path(__file__).parent.parent / "data" / "closing_prices.csv"
    with open(data_path, "r") as f:
        # Read closing prices from first column
        prices = [float(line.strip()) for line in f]
    return prices


# ---- Test Helper Methods ----


def verify_optimization_result(result: dict) -> None:
    assert isinstance(result, dict), "Optimization result must be a dictionary"
    assert "params" in result, "Result must contain 'params'"
    assert "target" in result, "Result must contain 'target'"
    assert isinstance(result["target"], float), "Target must be a float"


def verify_labels(labels: list[int]) -> None:
    """Verify that labels meet basic requirements.

    Args:
        labels (list[int]): Labels to verify
    """
    assert all(isinstance(label, int) for label in labels), "Labels must be integers"
    assert all(label in [-1, 0, 1] for label in labels), "Labels must be -1, 0, or 1"


def verify_parameters_match(
    actual_params: dict,
    expected_params: dict,
    rtol: float = PARAM_RTOL,
) -> None:
    """Verify that actual parameters match expected values within tolerance.

    Args:
        actual_params (dict): Actual parameters from optimization
        expected_params (dict): Expected parameter values
        rtol (float, optional): Relative tolerance for comparison
    """
    for param_name, expected_value in expected_params.items():
        assert param_name in actual_params, f"Missing parameter: {param_name}"
        actual_value = float(actual_params[param_name])
        assert np.isclose(
            actual_value,
            expected_value,
            rtol=rtol,
        ), f"Parameter {param_name} differs from expected"


# ---- Test Classes ----


class TestLabellersOptimization:
    """Tests for the optimization and labelling pipeline of all labellers."""

    @pytest.mark.parametrize(
        "labeller_class",
        [
            BinaryCTL,
            TernaryCTL,
            OracleBinaryTrendLabeller,
            OracleTernaryTrendLabeller,
        ],
    )
    def test_full_pipeline(self, labeller_class, optimizer, bounds, sample_prices):
        """
        Test the complete optimization and labelling pipeline for each labeller.

        This test verifies:
        1. Parameter optimization produces expected results
        2. Generated labels match expected labels
        3. Optimization target (return) matches expected value

        Args:
            labeller_class: The labeller class to test
            optimizer: Optimizer instance
            bounds: OptimizationBounds instance
            sample_prices: Sample price data
        """
        # Load expected optimization results
        expected_results = load_optimization_results()[labeller_class.__name__]

        # Run optimization with default bounds
        result = optimizer.optimize(
            labeller_class=labeller_class,
            time_series_list=sample_prices,
            bounds=bounds.get_bounds(labeller_class),
            verbose=0,
        )

        # Verify optimization results
        verify_optimization_result(result)
        verify_parameters_match(result["params"], expected_results["params"])
        assert np.isclose(
            result["target"],
            expected_results["target"],
            rtol=TARGET_RTOL,
        ), "Optimization target differs from expected"

        # Generate and verify labels
        params = {
            k: float(v) if isinstance(v, np.floating) else v
            for k, v in result["params"].items()
        }
        if "window_size" in params:
            params["window_size"] = int(params["window_size"])

        labeller = labeller_class(**params)
        labels = labeller.get_labels(sample_prices)
        expected_labels = load_expected_labels(
            labeller_class.__name__, len(sample_prices)
        )

        verify_labels(labels)
        assert len(labels) == len(expected_labels), "Number of labels doesn't match"
        assert labels == expected_labels, "Labels don't match expected values"
