from abc import ABC, abstractmethod
from collections import Counter

from .fees_config import FeesConfig


class BaseReturnEstimator(ABC):
    """Abstract base class for return estimators.

    This class serves as a template for all return estimation implementations.
    It provides common input validation functionality and defines the interface
    that all return estimators must implement.

    Attributes:
        None

    Example:
        To implement a new return estimator, inherit from this class and implement
        the estimate_return method::

            class MyEstimator(BaseReturnEstimator):
                def estimate_return(self, prices, labels):
                    # Implementation here
                    return calculated_return
    """

    def _verify_input_data(self, prices: list[float]) -> None:
        """Verify that the input data is valid.

        Args:
            prices (list[float]): List of prices to verify

        Raises:
            ValueError: If any specification of prices is not valid
        """
        if not isinstance(prices, list):
            raise ValueError("Prices must be a list")
        if not all(isinstance(price, float) for price in prices):
            raise ValueError("Prices must be a list of floats")

    @abstractmethod
    def estimate_return(self, prices: list[float], labels: list[int]) -> float:
        """Estimate returns based on prices and labels.

        Args:
            prices (list[float]): List of prices
            labels (list[int]): List of position labels

        Returns:
            float: Estimated return
        """
        pass


class SimpleReturnEstimator(BaseReturnEstimator):
    """
    A simple return estimator that calculates returns based on price differences and labels.

    This class implements a basic return estimation strategy by multiplying the price
    differences between consecutive periods with their corresponding labels. The labels
    indicate the position taken (-1 for short, 0 for no position, 1 for long).

    Example:
        >>> prices = [100.0, 101.0, 99.0]
        >>> labels = [1, 1, -1]
        >>> estimator = SimpleReturnEstimator()
        >>> return_value = estimator.estimate_return(prices, labels)
        >>> print(return_value)
        2.0

        In this example, the return is calculated as follows:
        (101.0 - 100.0) * 1 + (99.0 - 101.0) * -1 = 2.0
    """

    def _verify_labels(self, prices: list[float], labels: list[int]):
        """Verify that the labels are valid.

        Raises:
            ValueError: If any specification of labels is not valid
        """
        if not isinstance(labels, list):
            raise ValueError("Labels must be a list")
        if not all(isinstance(label, int) for label in labels):
            raise ValueError("Labels must be a list of integers")
        if len(prices) != len(labels):
            raise ValueError("Prices and labels must have the same length")
        if not all(label in [-1, 0, 1] for label in labels):
            raise ValueError("Labels must be -1, 0, or 1")

    def _calculate_return(self, prices: list[float], labels: list[int]) -> float:
        """Calculate the return based on price differences and labels.

        Returns:
            float: The calculated return
        """
        return_value = [
            (prices[i] - prices[i - 1]) * labels[i] for i in range(1, len(prices))
        ]
        return sum(return_value)

    def estimate_return(self, prices: list[float], labels: list[int]) -> float:
        """
        Estimate the return based on price differences and labels.

        Args:
            prices (list[float]): A list of historical prices
            labels (list[int]): A list of position labels (-1, 0, or 1)

        Returns:
            float: The estimated return
        """
        self._verify_input_data(prices)
        self._verify_labels(prices, labels)
        return self._calculate_return(prices, labels)


class ReturnsEstimatorWithFees(SimpleReturnEstimator):
    """
    A return estimator that incorporates transaction and holding fees into return calculations.

    This class extends the SimpleReturnEstimator by adding various types of fees that impact
    the overall return calculation. The goal of these fees is twofold:
    1. To account for the cost of entering and exiting positions in real life, as well as maintaining positions
    2. Act as a form of regularization to prevent overfitting the prices fluctuations,
    either by identifying ultrashort term trends or overextending trends over neutral periods.

    Transaction Fees:
        - Long Position (lp) Transaction Fees: Applied when introducing a positive (upward trend) label
        - Short Position (sp) Transaction Fees: Applied when introducing a negative (downward trend) label

    Holding Fees:
        - Long Position (lp) Holding Fees: Ongoing fees charged for maintaining a positive (upward trend) label
        - Short Position (sp) Holding Fees: Ongoing fees charged for maintaining a negative (downward trend) label

    All fees are expressed as percentages of the position value.

    The return calculation will:
        1. Include the basic price movement returns
        2. Subtract transaction fees when positions change
        3. Subtract daily holding fees based on position type

    Attributes:
        fees_config (FeesConfig): Configuration for transaction and holding fees
    """

    def __init__(self, fees_config: FeesConfig | None = None):
        """Initialize the estimator with fees configuration.

        Args:
            fees_config (FeesConfig, optional): Configuration for transaction and holding fees.
                If None, creates a config with zero fees.
        """
        self.fees_config = fees_config or FeesConfig()

    def _estimate_holding_fees(self, prices: list[float], labels: list[int]) -> float:
        """Estimate the holding fees based on the labels and prices."""
        label_counter = Counter(labels)
        return (
            label_counter[1] * self.fees_config.lp_holding_fees
            + label_counter[-1] * self.fees_config.sp_holding_fees
        )

    def _estimate_transaction_fees(
        self, prices: list[float], labels: list[int]
    ) -> float:
        """Estimate the transaction fees based on the labels and prices."""
        total_fees = 0
        for label_value, fee in zip(
            [1, -1],
            [
                self.fees_config.lp_transaction_fees,
                self.fees_config.sp_transaction_fees,
            ],
        ):
            total_fees += (
                sum(
                    prices[i]
                    for i in range(1, len(labels))
                    if labels[i] == label_value and labels[i - 1] != label_value
                )
                + int(labels[0] == label_value) * prices[0]
            ) * fee
        return total_fees

    def estimate_return(self, prices: list[float], labels: list[int]) -> float:
        """
        Estimate the return based on price differences and labels, and include fees cost if it is not zero.

        Args:
            prices (list[float]): A list of historical prices
            labels (list[int]): A list of position labels (-1, 0, or 1)

        Returns:
            float: The estimated return
        """
        self._verify_input_data(prices)
        self._verify_labels(prices, labels)
        fees = 0
        if (
            self.fees_config.lp_transaction_fees != 0
            or self.fees_config.sp_transaction_fees != 0
        ):
            fees += self._estimate_transaction_fees(prices, labels)
        if (
            self.fees_config.lp_holding_fees != 0
            or self.fees_config.sp_holding_fees != 0
        ):
            fees += self._estimate_holding_fees(prices, labels)
        return self._calculate_return(prices, labels) - fees
