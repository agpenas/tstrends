from abc import ABC, abstractmethod
from collections import Counter
from typing import List


class BaseReturnEstimator(ABC):
    """
    Base class for return estimators.
    """

    def __init__(self, prices: List[float]):
        self._verify_input_data(prices)
        self.prices = prices

    def _verify_input_data(self, prices: List[float]):
        """Verify that the input data is valid.

        Raises:
            ValueError: If any specification of prices is not valid
        """
        if not isinstance(prices, list):
            raise ValueError("Prices must be a list")
        if not all(isinstance(price, float) for price in prices):
            raise ValueError("Prices must be a list of floats")

    @abstractmethod
    def estimate_return(self, labels: List[int]) -> float:
        pass


class SimpleReturnEstimator(BaseReturnEstimator):
    """
    A simple return estimator that calculates returns based on price differences and labels.

    This class implements a basic return estimation strategy by multiplying the price
    differences between consecutive periods with their corresponding labels. The labels
    indicate the position taken (-1 for short, 0 for no position, 1 for long).

    Attributes:
        prices (List[float]): A list of historical prices

    Example:
        >>> prices = [100.0, 101.0, 99.0]
        >>> labels = [1, 1, -1]
        >>> estimator = SimpleReturnEstimator(prices)
        >>> return_value = estimator.estimate_return(labels)
        >>> print(return_value)
        2.0

        In this example, the return is calculated as follows:
        (101.0 - 100.0) * 1 + (99.0 - 101.0) * -1 = 2.0
    """

    def _verify_labels(self, labels: List[int]):
        """Verify that the labels are valid.

        Raises:
            ValueError: If any specification of labels is not valid
        """
        if not isinstance(labels, list):
            raise ValueError("Labels must be a list")
        if not all(isinstance(label, int) for label in labels):
            raise ValueError("Labels must be a list of integers")
        if len(self.prices) != len(labels):
            raise ValueError("Prices and labels must have the same length")
        if not all(label in [-1, 0, 1] for label in labels):
            raise ValueError("Labels must be -1, 0, or 1")

    def _calculate_return(self, labels: List[int]) -> float:
        """Calculate the return based on price differences and labels.

        Returns:
            float: The calculated return
        """
        return_value = [
            (self.prices[i] - self.prices[i - 1]) * labels[i]
            for i in range(1, len(self.prices))
        ]
        return sum(return_value)

    def estimate_return(self, labels: List[int]) -> float:
        """
        Estimate the return based on price differences and labels.

        Args:
            labels (List[int]): A list of position labels (-1, 0, or 1)

        Returns:
            float: The estimated return
        """
        self._verify_labels(labels)
        return self._calculate_return(labels)


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

    Attributes:
        prices (List[float]): A list of historical prices
        lp_transaction_fees (float): Transaction fee percentage for long positions
        sp_transaction_fees (float): Transaction fee percentage for short positions
        lp_holding_fees (float): Daily holding fee percentage for long positions
        sp_holding_fees (float): Daily holding fee percentage for short positions

        The return calculation will:
        1. Include the basic price movement returns
        2. Subtract transaction fees when positions change
        3. Subtract daily holding fees based on position type
    """

    def __init__(
        self,
        prices: List[float],
        lp_transaction_fees: float = 0,
        sp_transaction_fees: float = 0,
        lp_holding_fees: float = 0,
        sp_holding_fees: float = 0,
    ):
        super().__init__(prices)
        self._verify_input_fees(
            lp_transaction_fees, sp_transaction_fees, lp_holding_fees, sp_holding_fees
        )
        self.lp_transaction_fees = lp_transaction_fees
        self.sp_transaction_fees = sp_transaction_fees
        self.lp_holding_fees = lp_holding_fees
        self.sp_holding_fees = sp_holding_fees

    def _verify_input_fees(
        self,
        lp_transaction_fees: float,
        sp_transaction_fees: float,
        lp_holding_fees: float,
        sp_holding_fees: float,
    ):
        if (
            not isinstance(lp_transaction_fees, float)
            or not isinstance(sp_transaction_fees, float)
            or not isinstance(lp_holding_fees, float)
            or not isinstance(sp_holding_fees, float)
        ):
            raise ValueError(
                f"Fees must be floats. Received {lp_transaction_fees}, {sp_transaction_fees}, {lp_holding_fees}, {sp_holding_fees}"
            )
        if (
            lp_transaction_fees < 0
            or sp_transaction_fees < 0
            or lp_holding_fees < 0
            or sp_holding_fees < 0
        ):
            raise ValueError(
                f"Fees must be non-negative. Received {lp_transaction_fees}, {sp_transaction_fees}, {lp_holding_fees}, {sp_holding_fees}"
            )

    def _estimate_holding_fees(self, labels: List[int]) -> float:
        """
        Estimate the holding fees based on the labels and prices.
        """
        label_counter = Counter(labels)
        return (
            label_counter[1] * self.lp_transaction_fees
            + label_counter[-1] * self.sp_transaction_fees
        )

    def _estimate_transaction_fees(self, labels: List[int]) -> float:
        """
        Estimate the transaction fees based on the labels and prices. A transaction fee is applied when a given label is preceded by a different label.
        """
        total_fees = 0
        for label_value, fee in zip(
            [1, -1], [self.lp_transaction_fees, self.sp_transaction_fees]
        ):
            total_fees += (
                sum(
                    self.prices[i - 1]
                    for i in range(1, len(labels))
                    if labels[i] == label_value and labels[i - 1] != label_value
                )
                + int(labels[0] == label_value)
            ) * fee
        return total_fees

    def estimate_return(self, labels: List[int]) -> float:
        """
        Estimate the return based on price differences and labels, and include fees cost if it is not zero.

        Args:
            labels (List[int]): A list of position labels (-1, 0, or 1)

        Returns:
            float: The estimated return
        """
        self._verify_labels(labels)
        fees = 0
        if self.lp_transaction_fees != 0 or self.sp_transaction_fees != 0:
            fees += self._estimate_transaction_fees(labels)
        if self.lp_holding_fees != 0 or self.sp_holding_fees != 0:
            fees += self._estimate_holding_fees(labels)
        return self._calculate_return(labels) - fees
