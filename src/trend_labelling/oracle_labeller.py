from typing import Callable, List, Union
import numpy as np
from numpy.typing import NDArray
from .base_labeller import BaseLabeller
from .label_scaling import Labels, scale_binary, scale_ternary


class BaseOracleTrendLabeler(BaseLabeller):
    """
    Base class for Oracle Trend Labelers.
    """

    def __init__(self, transaction_cost: float) -> None:
        """
        Initialize the base Oracle Trend Labeler.

        Args:
            transaction_cost (float): Cost of making a transaction
        """
        if not isinstance(transaction_cost, float):
            raise TypeError("transaction_cost must be a float.")
        self.transaction_cost = transaction_cost

    def _scale_labels(self, labels: NDArray) -> NDArray:
        """
        Scale the labels.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def _verify_time_series(self, time_series_list: list[float]) -> None:
        """
        Verify the input time series.
        Args:
            time_series_list (list[float]): The price series.
        """
        if not isinstance(time_series_list, list):
            raise TypeError("time_series_list must be a list.")
        if not all(isinstance(price, (int, float)) for price in time_series_list):
            raise TypeError(
                "All elements in time_series_list must be integers or floats."
            )
        if len(time_series_list) < 2:
            raise ValueError("time_series_list must contain at least two elements.")

    def _compute_transition_costs(self, time_series_list: NDArray) -> NDArray:
        """
        Compute the transition costs.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def _forward_pass(self, time_series_list: NDArray) -> NDArray:
        """
        Perform the forward pass to calculate the state matrix.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def _backward_pass(
        self, S: NDArray, P: NDArray, time_series_arr: NDArray
    ) -> NDArray:
        """
        Perform the backward pass to determine the trend labels.
        Args:
            S (NDArray): State matrix of cumulative returns.
            P (NDArray): Transition cost matrix.
        Returns:
            labels (NDArray): Optimal trend labels.
        """
        T = len(time_series_arr)
        labels = np.zeros(T, dtype=int)
        labels[-1] = np.argmax(S[-1])  # Start from the last state

        for t in range(T - 2, -1, -1):
            labels[t] = np.argmax(S[t] + P[t, :, labels[t + 1]])

        return labels

    def get_labels(
        self, time_series_list: list[float], return_labels_as_int: bool = True
    ) -> Union[list[int], list[Labels]]:
        """
        Run the full Oracle Trend Labeling Algorithm over a time series.

        Args:
            time_series_list (list[float]): The price series.
            return_labels_as_int (bool, optional): If True, returns integer labels (-1, 0, 1),
                                                  if False returns Labels enum values. Defaults to True.

        Returns:
            Union[list[int], list[Labels]]: Optimal trend labels. If return_labels_as_int is True, returns scaled integers,
                                          otherwise returns Labels enum values.
        """
        self._verify_time_series(time_series_list)
        time_series_arr = np.array(time_series_list)

        P = self._compute_transition_costs(time_series_arr)
        S = self._forward_pass(time_series_list, P)
        labels = self._backward_pass(S, P, time_series_arr)

        scaled_labels = self._scale_labels(labels)
        return (
            scaled_labels.tolist()
            if return_labels_as_int
            else [Labels(x) for x in scaled_labels]
        )


class OracleBinaryTrendLabeler(BaseOracleTrendLabeler):
    """
    Oracle Binary Trend Labeler class, adapted to Python from the original paper by T. Kovačević, A. Merćep, S. Begušić and Z. Kostanjčar, "Optimal Trend Labeling in Financial Time Series,", doi: 10.1109/ACCESS.2023.3303283.
    """

    def __init__(self, transaction_cost: float) -> None:
        """
        Initialize the binary trend labeler.
        """
        super().__init__(transaction_cost)

    def _scale_labels(self, labels: NDArray) -> NDArray:
        """
        Scale the labels.
        """
        return scale_binary(labels)

    def _compute_transition_costs(self, time_series_list: NDArray):
        """
        Initialize the transition cost matrix.
        Returns:
            P (NDArray): Transition cost matrix of shape (T-1, 2, 2).
        """
        ts_len = len(time_series_list)

        P = np.zeros((ts_len - 1, 2, 2))

        for t in range(ts_len - 1):
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

    def _forward_pass(self, time_series_list: list[float], P: NDArray):
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


class OracleTernaryTrendLabeler(BaseOracleTrendLabeler):
    """
    Oracle Ternary Trend Labeler class that identifies three states: downtrend (0), neutral (1), and uptrend (2).
    Transitions between downtrend and uptrend must go through the neutral state.
    """

    def __init__(self, transaction_cost: float, trend_coeff: float) -> None:
        """
        Initialize the ternary trend labeler.

        Args:
            transaction_cost (float): Cost coefficient for switching between trends.
            trend_coeff (float): Trend coefficient for weighting price changes.
        """
        super().__init__(transaction_cost)
        if not isinstance(trend_coeff, float):
            raise TypeError("trend_coeff must be a float.")
        self.trend_coeff = trend_coeff

    def _scale_labels(self, labels: NDArray) -> NDArray:
        """
        Scale the labels.
        """
        return scale_ternary(labels)

    def _compute_transition_costs(self, time_series_arr: NDArray) -> NDArray:
        """
        Initialize the transition cost matrix for three states.

        Args:
            time_series_arr (NDArray): Array of price values.
        Returns:
            NDArray: Transition cost matrix of shape (T-1, 3, 3).
        """
        T = len(time_series_arr)
        P = np.full(
            (T - 1, 3, 3), -np.inf
        )  # Initialize with -inf for forbidden transitions

        for t in range(T - 1):
            price_change = time_series_arr[t + 1] - time_series_arr[t]
            switch_cost = -time_series_arr[t] * self.transaction_cost

            # Rewards for staying in same state
            P[t, 0, 0] = -price_change  # Reward for staying in downtrend
            P[t, 1, 1] = (
                abs(price_change) * self.trend_coeff
            )  # No reward for staying neutral
            P[t, 2, 2] = price_change  # Reward for staying in uptrend

            # Rewards for allowed transitions
            P[t, 0, 1] = switch_cost  # Downtrend to neutral
            P[t, 1, 0] = switch_cost  # Neutral to downtrend
            P[t, 1, 2] = switch_cost  # Neutral to uptrend
            P[t, 2, 1] = switch_cost  # Uptrend to neutral

        return P

    def _forward_pass(self, time_series_list: list[float], P: NDArray) -> NDArray:
        """
        Perform the forward pass to calculate the state matrix.

        Args:
            time_series_list (list[float]): The price series.
            P (NDArray): Transition cost matrix.

        Returns:
            NDArray: State matrix of cumulative returns.
        """
        T = len(time_series_list)
        S = np.zeros((T, 3))  # Initialize state matrix for 3 states

        # Iterate over time steps in forward direction
        for t in range(1, T):
            # Maximum return for being in downtrend
            S[t, 0] = max(
                S[t - 1, 0] + P[t - 1, 0, 0],  # Stay in downtrend
                S[t - 1, 1] + P[t - 1, 1, 0],  # Switch from neutral
            )

            # Maximum return for being in neutral
            S[t, 1] = max(
                S[t - 1, 0] + P[t - 1, 0, 1],  # Switch from downtrend
                S[t - 1, 1] + P[t - 1, 1, 1],  # Stay in neutral
                S[t - 1, 2] + P[t - 1, 2, 1],  # Switch from uptrend
            )

            # Maximum return for being in uptrend
            S[t, 2] = max(
                S[t - 1, 1] + P[t - 1, 1, 2],  # Switch from neutral
                S[t - 1, 2] + P[t - 1, 2, 2],  # Stay in uptrend
            )

        return S
