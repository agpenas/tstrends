from typing import List
from .base_labeller import BaseLabeller


class TernaryCTL(BaseLabeller):
    """
    Three-states continuous trend labeller class, adapted from the original paper by
    Dezhkam, Arsalan et al. "A Bayesian-based classification framework for financial time series trend prediction."
    https://doi.org/10.1007/s11227-022-04834-4
    """

    def __init__(self, marginal_change_thres: float, window_size: int) -> None:
        """
        Initialize the ternary trend labeller.

        Args:
            marginal_change_thres (float): The threshold for significant price movements as a percentage.
            window_size (int): The maximum window to look for trend confirmation before resetting state.
        """
        if not isinstance(marginal_change_thres, float):
            raise TypeError("marginal_change_thres must be a float.")
        if not isinstance(window_size, int):
            raise TypeError("window_size must be an integer.")

        self.marginal_change_thres = marginal_change_thres
        self.window_size = window_size
        self.labels: List[int] = []

    def _find_upward_trend(self, time_series_list: List[float]) -> None:
        """
        Find upward trends in a time series of closing prices. This is the first step of the ternary trend labelling algorithm.

        Args:
            time_series_list (List[float]): List of closing prices.
        """
        self.labels = [0]
        for previous_price, current_price in zip(
            time_series_list[:-1], time_series_list[1:]
        ):
            if self._is_significant_upward_move(current_price, previous_price):
                self.labels.append(1)
            else:
                self.labels.append(0)

    def _is_significant_upward_move(self, current: float, reference: float) -> bool:
        """
        Check if a current price is a significant upward move compared to a reference price.

        Args:
            current (float): The current price.
            reference (float): The reference price.

        Returns:
            bool: True if the current price is a significant upward move, False otherwise.
        """
        return current >= reference + self.marginal_change_thres * reference

    def _is_significant_downward_move(self, current: float, reference: float) -> bool:
        """
        Check if a current price is a significant downward move compared to a reference price.

        Args:
            current (float): The current price.
            reference (float): The reference price.

        Returns:
            bool: True if the current price is a significant downward move, False otherwise.
        """
        return current <= reference - self.marginal_change_thres * reference

    def _update_labels(self, start: int, end: int, new_label: int) -> None:
        """
        Update the labels in a range of the labels list.

        Args:
            start (int): The start index.
            end (int): The end index.
            new_label (int): The new label to assign to the range.
        """
        self.labels[start:end] = [new_label] * (end - start)

    def get_labels(
        self,
        prices: List[float],
    ) -> List[int]:
        """
        Labels trends in a time series of closing prices using a ternary classification approach.
        The method identifies three distinct states in price movements:
            - Upward trends (label: 1)
            - Downward trends (label: -1)
            - No-action (label: 0)

        The algorithm uses two key parameters:
            - marginal_change_thres: Defines the threshold for significant price movements as a percentage
            - window_size: Maximum window to look for trend confirmation before resetting state

        The labeling process works by tracking the current state and transitioning between
        states when price movements exceed thresholds, while using the window_size parameter
        to avoid getting stuck in prolonged sideways movements.

        Parameters:
            prices (List[float]): List of closing prices.

        Returns:
            List[int]: List of labels where 1 indicates an upward trend,
                    -1 indicates a downward trend, and 0 indicates no-action.
        """
        self._verify_time_series(prices)

        # Initialize labels with upward trend detection
        self._find_upward_trend(prices)
        trend_start = 0

        for current_idx, current_price in enumerate(prices[1:], start=1):
            reference_price = prices[trend_start]
            window_exceeded = current_idx - trend_start > self.window_size

            match self.labels[current_idx]:
                case 1:  # Upward trend
                    if current_price > reference_price:
                        self._update_labels(trend_start, current_idx, 1)
                    elif self._is_significant_downward_move(
                        current_price, reference_price
                    ):
                        self._update_labels(trend_start, current_idx, -1)
                    elif window_exceeded:
                        self._update_labels(trend_start, current_idx, 0)
                    else:
                        continue
                    trend_start = current_idx

                case -1:  # Downward trend
                    if current_price < reference_price:
                        self._update_labels(trend_start, current_idx, -1)
                    elif self._is_significant_upward_move(
                        current_price, reference_price
                    ):
                        self._update_labels(trend_start, current_idx, 1)
                    elif window_exceeded:
                        self._update_labels(trend_start, current_idx, 0)
                    else:
                        continue
                    trend_start = current_idx

                case 0:  # No trend
                    if self._is_significant_upward_move(current_price, reference_price):
                        self._update_labels(trend_start, current_idx, 1)
                    elif self._is_significant_downward_move(
                        current_price, reference_price
                    ):
                        self._update_labels(trend_start, current_idx, -1)
                    elif window_exceeded:
                        self._update_labels(trend_start, current_idx, 0)
                    else:
                        continue
                    trend_start = current_idx

        return self.labels
