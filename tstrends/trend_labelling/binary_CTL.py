from dataclasses import dataclass
from typing import Union

from .base_labeller import BaseLabeller
from .label_scaling import Labels, extract_label_values


@dataclass
class TrendState:
    """Holds the state for trend labelling process."""

    current_high: float = 0.0
    current_low: float = 0.0
    curr_high_time: int = 0
    curr_low_time: int = 0
    current_direction: int = Labels.NEUTRAL
    extreme_point_idx: int = 0

    def set_upwards_trend(self, price: float, time_idx: int) -> None:
        """Set the state for an upward trend.

        Args:
            price: The current price value
            time_idx: The current time index
        """
        self.current_high = price
        self.curr_high_time = time_idx
        self.extreme_point_idx = time_idx
        self.current_direction = Labels.UP

    def set_downwards_trend(self, price: float, time_idx: int) -> None:
        """Set the state for a downward trend.

        Args:
            price: The current price value
            time_idx: The current time index
        """
        self.current_low = price
        self.curr_low_time = time_idx
        self.extreme_point_idx = time_idx
        self.current_direction = Labels.DOWN


class BinaryCTL(BaseLabeller):
    """Binary Continuous Trend Labeller.

    This class implements a binary trend labelling algorithm based on the paper by
    Wu, D., Wang, X., Su, J., Tang, B., & Wu, S. "A Labeling Method for Financial
    Time Series Prediction Based on Trends".

    The algorithm identifies two distinct states in price movements:
        - Upward trends (label: Labels.UP or 1)
        - Downward trends (label: Labels.DOWN or -1)

    Example:
        >>> labeller = BinaryCTL(omega=0.1)
        >>> prices = [1.0, 1.15, 1.2, 1.0]
        >>> labels = labeller.get_labels(prices)
        >>> print(labels)  # [-1, 1, 1, -1]

    Note:
        The omega parameter determines how significant a price movement must be
        to be considered a trend change. Higher values result in fewer trend
        changes being identified.
    """

    def __init__(self, omega: float) -> None:
        """
        Initialize the continuous trend labeller.

        Args:
            omega (float): The proportion threshold parameter of the trend definition.
        """
        if not isinstance(omega, float):
            raise TypeError("omega must be a float.")
        self.omega = omega
        self._state: TrendState = TrendState()
        self._labels: list[Labels] = list()

    def _initialize_labels(self, length: int) -> None:
        """Initialize the labels list with neutral values.

        Args:
            length: Length of the time series
        """
        self._labels = [Labels.NEUTRAL] * length

    def _update_labels(self, start_idx: int, end_idx: int, label_value: int) -> None:
        """Update a range of labels with the specified value.

        Args:
            start_idx: Start index (inclusive)
            end_idx: End index (inclusive)
            label_value: The label value to set
        """

        for i in range(start_idx, end_idx + 1):
            self._labels[i] = label_value

    def _detect_initial_trend(self, time_series_list: list[float]) -> None:
        """
        Detect the initial trend direction by finding the first significant price movement.

        Args:
            time_series_list: The input time series data
        """

        first_price = time_series_list[0]

        for i, price in enumerate(time_series_list):
            if price > first_price * (1 + self.omega):
                self._state.set_upwards_trend(price, i)
                self._update_labels(0, i - 1, Labels.UP)
                return

            elif price < first_price * (1 - self.omega):
                self._state.set_downwards_trend(price, i)
                self._update_labels(0, i - 1, Labels.DOWN)
                return

    def _handle_uptrend(self, price: float, time_idx: int) -> None:
        """
        Handle the uptrend case in continuous trend detection.

        Args:
            price: Current price value
            time_idx: Current time index
        """

        if price > self._state.current_high:
            self._state.set_upwards_trend(price, time_idx)
            return

        elif price < self._state.current_high * (1 - self.omega):
            self._update_labels(
                self._state.curr_low_time + 1,
                self._state.curr_high_time,
                Labels.UP,
            )
            self._state.set_downwards_trend(price, time_idx)

    def _handle_downtrend(self, price: float, time_idx: int) -> None:
        """
        Handle the downtrend case in continuous trend detection.

        Args:
            price: Current price value
            time_idx: Current time index
        """

        if price < self._state.current_low:
            self._state.set_downwards_trend(price, time_idx)

        elif price > self._state.current_low * (1 + self.omega):
            self._update_labels(
                self._state.curr_high_time + 1,
                self._state.curr_low_time,
                Labels.DOWN,
            )
            self._state.set_upwards_trend(price, time_idx)

    def get_labels(
        self, time_series_list: list[float], return_labels_as_int: bool = True
    ) -> Union[list[int], list[Labels]]:
        """Auto-labels a price time series based on the provided algorithm.

        Parameters
        ----------
        time_series_list : list[float]
            The original time series data X = [x1, x2, ..., xN]
        return_labels_as_int : bool, optional
            If True, returns integer labels (-1, 1), if False returns Labels enum values.
            Defaults to True.

        Returns
        -------
        Union[list[int], list[Labels]]
            The label vector Y. If return_labels_as_int is True, returns integers (-1, 1),
            otherwise returns Labels enum values (Labels.DOWN, Labels.UP).
        """
        self._verify_time_series(time_series_list)

        # Initialize labels and state
        self._initialize_labels(len(time_series_list))

        # Detect initial trend direction
        self._detect_initial_trend(time_series_list)

        # Continue trend detection for the rest of the series
        for i in range(self._state.extreme_point_idx + 1, len(time_series_list)):
            if self._state.current_direction == Labels.UP:
                self._handle_uptrend(time_series_list[i], i)
            elif self._state.current_direction == Labels.DOWN:
                self._handle_downtrend(time_series_list[i], i)

        # Label the last interval
        if self._state.curr_low_time != self._state.curr_high_time:
            self._update_labels(
                min(self._state.curr_low_time, self._state.curr_high_time) + 1,
                len(time_series_list) - 1,
                self._state.current_direction,
            )

        return (
            extract_label_values(self._labels) if return_labels_as_int else self._labels
        )
