from typing import Union

from .base_labeller import BaseLabeller
from .label_scaling import Labels, extract_label_values


class TernaryCTL(BaseLabeller):
    """Ternary Continuous Trend Labeller.

    This class implements an adaptation of the Continuous Trend Labeller (CTL) algorithm
    to a three-state labelling approach. A somewhat not so different approach is proposed in the second
    pass of the labelling algorithm outlined in the paper by Dezhkam et al.
    "A Bayesian-based classification framework for financial time series trend prediction."

    The algorithm identifies three distinct states in price movements:
        - Upward trends (label: Labels.UP or 1)
        - Neutral trends (label: Labels.NEUTRAL or 0)
        - Downward trends (label: Labels.DOWN or -1)

    Example:
        >>> labeller = TernaryCTL(marginal_change_thres=0.1, window_size=3)
        >>> prices = [1.0, 1.15, 1.2, 1.18, 1.0]
        >>> labels = labeller.get_labels(prices)
        >>> print(labels)  # [-1, 1, 1, 0, -1]

    Note:
        The window_size parameter helps prevent the algorithm from getting stuck
        in prolonged sideways movements by forcing a state transition to NEUTRAL after the
        window is exceeded. It can artificially cut ongoing trends short, so it must be set
        carefully.
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
        self.labels: list[Labels] = list()

    def _get_first_label(self, time_series_list: list[float]) -> None:
        """
        Find upward trends in a time series of closing prices. This is the first step of the ternary trend labelling algorithm.

        Args:
            time_series_list (list[float]): List of closing prices.
        """
        if time_series_list[0] > time_series_list[1]:
            return [Labels.DOWN]
        return [Labels.UP]

    def _is_significant_upward_move(self, current: float, reference: float) -> bool:
        """
        Check if a current price is a significant upward move compared to a reference price.

        Args:
            current (float): The current price.
            reference (float): The reference price.

        Returns:
            bool: True if the current price is a significant upward move, False otherwise.
        """
        return current >= reference * (1 + self.marginal_change_thres)

    def _is_significant_downward_move(self, current: float, reference: float) -> bool:
        """
        Check if a current price is a significant downward move compared to a reference price.

        Args:
            current (float): The current price.
            reference (float): The reference price.

        Returns:
            bool: True if the current price is a significant downward move, False otherwise.
        """
        return current <= reference * (1 - self.marginal_change_thres)

    def _generate_label_values(self) -> list[int]:
        """Convert Labels enum to their integer values"""
        return [label.value for label in self.labels]

    def _right_pad_labels(self, total_length: int) -> list[int]:
        """
        Right pad the labels list by duplicating the last element.
        Args:
            total_length (int): The target length of the padded list.

        Returns:
            list[Labels]: Padded list of label values with length equal to target_length.
        """
        if len(self.labels) == 0:
            return []
        self.labels += [self.labels[-1]] * (total_length - len(self.labels))

    def _update_labels(self, trend_start: int, current_idx: int, label: Labels) -> None:
        """
        Update the labels list with a new label value.

        Args:
            trend_start (int): The starting index of the trend.
            current_idx (int): The current index of the price.
            label (Labels): The new label value to be added.
        """
        self.labels += [label] * (current_idx - trend_start)

    def _has_price_crossed_reference_price(
        self, previous_price: float, current_price: float, reference_price: float
    ) -> bool:
        """
        Check if the price has crossed the reference price.
        """
        return (previous_price - reference_price) * (
            current_price - reference_price
        ) <= 0

    def get_labels(
        self, prices: list[float], return_labels_as_int: bool = True
    ) -> Union[list[int], list[Labels]]:
        """Labels trends in a time series of closing prices using a ternary classification approach.

        The method identifies three distinct states in price movements:
            - Upward trends (label: Labels.UP)
            - Downward trends (label: Labels.DOWN)
            - No-action (label: Labels.NEUTRAL)

        The algorithm uses two key parameters:
            - marginal_change_thres: Defines the threshold for significant price movements as a percentage
            - window_size: Maximum window to look for trend confirmation before resetting state

        The labeling process works by tracking the current state and transitioning between
        states when price movements exceed thresholds, while using the window_size parameter
        to avoid getting stuck in prolonged sideways movements.

        Parameters
        ----------
        prices : list[float]
            List of closing prices.
        return_labels_as_int : bool, optional
            If True, returns integer labels (-1, 0, 1), if False returns Labels enum values.
            Defaults to True.

        Returns
        -------
        Union[list[int], list[Labels]]
            List of labels. If return_labels_as_int is True, returns integers (-1, 0, 1),
            otherwise returns Labels enum values.
        """
        self._verify_time_series(prices)
        # Initialize labels
        self.labels = self._get_first_label(prices)
        # Initialize trend start index
        trend_start = 0
        # Iterate over prices starting from the second price
        for current_idx, current_price in enumerate(prices[1:], start=1):
            reference_price = prices[trend_start]
            window_exceeded = current_idx - trend_start > self.window_size

            match self.labels[-1]:
                case Labels.UP:  # Upward trend
                    if current_price > reference_price:
                        self._update_labels(trend_start, current_idx, Labels.UP)
                    elif self._is_significant_downward_move(
                        current_price, reference_price
                    ):
                        self._update_labels(trend_start, current_idx, Labels.DOWN)
                    elif window_exceeded:
                        self._update_labels(trend_start, current_idx, Labels.NEUTRAL)
                    else:
                        continue
                    trend_start = current_idx

                case Labels.DOWN:  # Downward trend
                    if current_price < reference_price:
                        self._update_labels(trend_start, current_idx, Labels.DOWN)
                    elif self._is_significant_upward_move(
                        current_price, reference_price
                    ):
                        self._update_labels(trend_start, current_idx, Labels.UP)
                    elif window_exceeded:
                        self._update_labels(trend_start, current_idx, Labels.NEUTRAL)
                    else:
                        continue
                    trend_start = current_idx

                case Labels.NEUTRAL:  # No trend
                    if self._is_significant_upward_move(current_price, reference_price):
                        self._update_labels(trend_start, current_idx, Labels.UP)
                    elif self._is_significant_downward_move(
                        current_price, reference_price
                    ):
                        self._update_labels(trend_start, current_idx, Labels.DOWN)
                    elif window_exceeded:
                        self._update_labels(trend_start, current_idx, Labels.NEUTRAL)
                    else:
                        continue
                    trend_start = current_idx

        self._right_pad_labels(len(prices))
        return (
            extract_label_values(self.labels) if return_labels_as_int else self.labels
        )
