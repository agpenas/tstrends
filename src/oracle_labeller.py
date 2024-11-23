from typing import List, Union
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


class OracleBinaryTrendLabeler:
    """
    Oracle Binary Trend Labeler class, adapted to Python from the original paper by T. Kovačević, A. Merćep, S. Begušić and Z. Kostanjčar, "Optimal Trend Labeling in Financial Time Series,", doi: 10.1109/ACCESS.2023.3303283.

    """

    def __init__(self, transaction_cost: float) -> None:
        """
        Initialize the binary trend labeler.

        Args:
            transaction_cost (float): Cost of switching trends.
        """
        if not isinstance(transaction_cost, float):
            raise TypeError("transaction_cost must be a float.")
        self.transaction_cost = transaction_cost

    def _verify_time_series(self, time_series_list: List[float]) -> None:
        """
        Verify the input time series.
        Args:
            time_series_list (List[float]): The price series.
        """
        if not isinstance(time_series_list, list):
            raise TypeError("time_series_list must be a list.")
        if not all(isinstance(price, (int, float)) for price in time_series_list):
            raise TypeError(
                "All elements in time_series_list must be integers or floats."
            )
        if len(time_series_list) < 2:
            raise ValueError("time_series_list must contain at least two elements.")

    def _compute_transition_costs(self, time_series_list: NDArray):
        """
        Initialize the transition cost matrix.
        Returns:
            P (NDArray): Transition cost matrix of shape (T-1, 2, 2).
        """
        P = np.zeros((len(time_series_list) - 1, 2, 2))

        for t in range(len(time_series_list) - 1):
            price_change = time_series_list[t + 1] - time_series_list[t]
            # Staying in the same state
            P[t, 0, 0] = 0  # No cost for staying in downtrend
            P[t, 1, 1] = price_change  # Cost for staying in uptrend

            # Switching states
            P[t, 0, 1] = (
                -time_series_list[t] * self.transaction_cost
            )  # Downtrend to uptrend
            P[t, 1, 0] = (
                -time_series_list[t] * self.transaction_cost
            )  # Uptrend to downtrend

        return P

    def _forward_pass(self, time_series_list: List[float], P: NDArray):
        """
        Perform the forward pass to calculate the state matrix.
        Args:
            P (NDArray): Transition cost matrix.
        Returns:
            S (NDArray): State matrix of cumulative returns.
        """
        S = np.zeros((len(time_series_list), 2))  # Initialize state matrix

        # Iterate over time steps in forward direction
        for t in range(1, len(time_series_list)):
            S[t, 0] = max(S[t - 1, 0] + P[t - 1, 0, 0], S[t - 1, 1] + P[t - 1, 1, 0])
            S[t, 1] = max(S[t - 1, 0] + P[t - 1, 0, 1], S[t - 1, 1] + P[t - 1, 1, 1])

        return S

    def _backward_pass(
        self, S: NDArray, P: NDArray, time_series_arr: NDArray
    ) -> NDArray:
        """
        Perform the backward pass to determine the trend labels.
        Args:
            S (NDArray): State matrix of cumulative returns.
            P (NDArray): Transition cost matrix.
        Returns:
            labels (NDArray): Optimal trend labels (0 for downtrend, 1 for uptrend).
        """
        labels = np.zeros(len(time_series_arr), dtype=int)
        labels[-1] = np.argmax(S[-1])  # Start from the last state

        for t in range(len(time_series_arr) - 2, -1, -1):
            labels[t] = np.argmax(S[t] + P[t, :, labels[t + 1]])

        return labels

    def label_trends(self, time_series_list: List[float]) -> List[int]:
        """
        Run the full Oracle Trend Labeling Algorithm over a time series.
        Args:
            time_series_list (List[float]): The price series.
        Returns:
            labels (List[int]): Optimal trend labels (0 for downtrend, 1 for uptrend).

        """
        self._verify_time_series(time_series_list)
        time_series_arr = np.array(time_series_list)

        P = self._compute_transition_costs(time_series_arr)
        S = self._forward_pass(P, time_series_arr)
        labels = self._backward_pass(S, P, time_series_arr)

        return labels.tolist()

