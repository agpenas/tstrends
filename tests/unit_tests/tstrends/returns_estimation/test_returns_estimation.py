import pytest

from tstrends.returns_estimation.fees_config import FeesConfig
from tstrends.returns_estimation.returns_estimation import (
    ReturnsEstimatorWithFees,
    SimpleReturnEstimator,
)


@pytest.fixture
def simple_estimator():
    """Fixture providing a SimpleReturnEstimator instance."""
    return SimpleReturnEstimator()


@pytest.fixture
def fees_estimator():
    """Fixture providing a ReturnsEstimatorWithFees instance with sample fees."""
    fees_config = FeesConfig(
        lp_transaction_fees=0.001,  # 0.1% fee
        sp_transaction_fees=0.002,  # 0.2% fee
        lp_holding_fees=0.0005,  # 0.05% fee
        sp_holding_fees=0.0010,  # 0.1% fee
    )
    return ReturnsEstimatorWithFees(fees_config)


@pytest.fixture
def zero_fees_estimator():
    """Fixture providing a ReturnsEstimatorWithFees instance with zero fees."""
    return ReturnsEstimatorWithFees()


class TestSimpleReturnEstimator:
    def test_valid_input(self, simple_estimator):
        """Test estimation with valid input data."""
        prices = [100.0, 101.0, 99.0, 102.0]
        labels = [1, 1, -1, 1]
        result = simple_estimator.estimate_return(prices, labels)
        # (101-100)*1 + (99-101)*(-1) + (102-99)*1 = 1 + 2 + 3 = 6
        assert result == 6.0

    def test_zero_returns(self, simple_estimator):
        """Test estimation with no position changes."""
        prices = [100.0, 100.0, 100.0]
        labels = [1, 1, 1]
        result = simple_estimator.estimate_return(prices, labels)
        assert result == 0.0

    def test_neutral_position(self, simple_estimator):
        """Test estimation with neutral positions."""
        prices = [100.0, 101.0, 99.0, 102.0]
        labels = [0, 0, 0, 0]
        result = simple_estimator.estimate_return(prices, labels)
        assert result == 0.0

    @pytest.mark.parametrize(
        "invalid_prices,invalid_labels,error_type,error_match",
        [
            (
                [100, "101", 99.0],
                [1, 1, 1],
                ValueError,
                "Prices must be a list of floats",
            ),
            (
                [100.0, 101.0],
                [1, 1, 1],
                ValueError,
                "Prices and labels must have the same length",
            ),
            (
                [100.0, 101.0, 99.0],
                [1, 2, 1],
                ValueError,
                "Labels must be -1, 0, or 1",
            ),
            (
                [100.0, 101.0, 99.0],
                [1.0, 0.0, -1.0],
                ValueError,
                "Labels must be a list of integers",
            ),
        ],
        ids=[
            "invalid_price_type",
            "length_mismatch",
            "invalid_label_value",
            "invalid_label_type",
        ],
    )
    def test_invalid_inputs(
        self, simple_estimator, invalid_prices, invalid_labels, error_type, error_match
    ):
        """Test handling of invalid inputs."""
        with pytest.raises(error_type, match=error_match):
            simple_estimator.estimate_return(invalid_prices, invalid_labels)

    class TestHoldingFees:
        """Tests specifically for holding fees calculation."""

        def test_lp_position(self, fees_estimator):
            """Test holding fees for a single continuous position."""
            prices = [100.0, 101.0, 102.0]
            labels = [1, 1, 1]  # Continuous long position

            # Three long positions * 0.0005 = 0.0015
            expected_fees = 0.0015

            result = fees_estimator._estimate_holding_fees(prices, labels)
            assert abs(result - expected_fees) < 1e-10

        def test_mixed_positions(self, fees_estimator):
            """Test holding fees for mixed positions."""
            prices = [100.0, 101.0, 100.0, 99.0]
            labels = [1, 1, -1, -1]  # Long -> Neutral -> Short -> Long

            # Two long positions * 0.0005 + Two short positions * 0.0010
            expected_fees = 2 * 0.0005 + 2 * 0.0010

            result = fees_estimator._estimate_holding_fees(prices, labels)
            assert abs(result - expected_fees) < 1e-10

        def test_no_positions(self, fees_estimator):
            """Test holding fees with no positions."""
            prices = [100.0, 101.0, 102.0]
            labels = [0, 0, 0]

            result = fees_estimator._estimate_holding_fees(prices, labels)
            assert result == 0.0

    class TestTransactionFees:
        """Tests specifically for transaction fees calculation."""

        def test_initial_position(self, fees_estimator):
            """Test transaction fees for initial position."""
            prices = [100.0, 101.0, 102.0]
            labels = [1, 1, 1]  # Initial long position

            # Initial long position: 100 * 0.001 = 0.1
            expected_fees = 0.1

            result = fees_estimator._estimate_transaction_fees(prices, labels)
            assert abs(result - expected_fees) < 1e-10

        def test_position_changes(self, fees_estimator):
            """Test transaction fees for position changes."""
            prices = [100.0, 101.0, 99.0]
            labels = [0, 1, -1]  # Neutral -> Long -> Short

            # Long entry at price 100: 101 * 0.001 = 0.101
            # Short entry at price 101: 99 * 0.002 = 0.198
            expected_fees = 0.299

            result = fees_estimator._estimate_transaction_fees(prices, labels)
            assert abs(result - expected_fees) < 1e-10

        def test_multiple_transitions(self, fees_estimator):
            """Test transaction fees for multiple position transitions."""
            prices = [100.0, 101.0, 99.0, 102.0]
            labels = [1, 0, -1, 1]  # Long -> Neutral -> Short -> Long

            # For long positions (fee = 0.001):
            # - Initial position: 100 * 0.001 = 0.1
            # - Transition at index 3: 102 * 0.001 = 0.102
            # For short positions (fee = 0.002):
            # - Transition at index 2: 99 * 0.002 = 0.198
            expected_fees = 0.4

            result = fees_estimator._estimate_transaction_fees(prices, labels)
            assert abs(result - expected_fees) < 1e-10

        def test_no_transitions(self, fees_estimator):
            """Test transaction fees with no position changes."""
            prices = [100.0, 101.0, 102.0]
            labels = [0, 0, 0]

            result = fees_estimator._estimate_transaction_fees(prices, labels)
            assert result == 0.0


class TestReturnsEstimatorWithFees:
    def test_zero_fees_matches_simple(self, zero_fees_estimator, simple_estimator):
        """Test that zero fees estimator matches SimpleReturnEstimator."""
        prices = [100.0, 101.0, 99.0, 102.0]
        labels = [1, 1, -1, 1]

        zero_fees_result = zero_fees_estimator.estimate_return(prices, labels)
        simple_result = simple_estimator.estimate_return(prices, labels)

        assert zero_fees_result == simple_result

    def test_no_position_changes(self, fees_estimator):
        """Test estimation with no position changes."""
        prices = [100.0, 101.0, 102.0]
        labels = [0, 0, 0]  # No positions

        result = fees_estimator.estimate_return(prices, labels)
        assert result == 0.0  # No returns and no fees for neutral positions

    def test_fully_upwards_trend(self, fees_estimator):
        """Test estimation with a fully upwards trend."""
        prices = [100.0, 101.0, 102.0, 103.0]
        labels = [1, 1, 1, 1]  # Fully upwards trend

        expected_fees = 4 * 0.0005 + 100 * 0.001
        expected_return = 3 - expected_fees
        result = fees_estimator.estimate_return(prices, labels)
        assert result == expected_return

    def test_fully_downwards_trend(self, fees_estimator):
        """Test estimation with a fully downwards trend."""
        prices = [100.0, 99.0, 98.0, 97.0]
        labels = [-1, -1, -1, -1]  # Fully downwards trend

        expected_fees = 4 * 0.0010 + 100 * 0.002
        expected_return = 3 - expected_fees
        result = fees_estimator.estimate_return(prices, labels)
        assert result == expected_return

    def test_mixed_trend(self, fees_estimator):
        """Test estimation with a mixed trend."""
        prices = [100.0, 101.0, 99.0, 102.0]
        labels = [1, 0, -1, 1]  # Upwards -> Neutral -> Downwards -> Upwards

        expected_fees = 2 * 0.0005 + 0.001 + 99 * 0.002 + (100 + 102) * 0.001
        expected_return = 5 - expected_fees
        result = fees_estimator.estimate_return(prices, labels)
        assert result == expected_return
