"""
Remaining value change tuner for trend labels.

This module provides a tuner that enhances trend labels with information about
the remaining value change until the end of a continuous trend interval.
"""

from itertools import pairwise
import numpy as np

from tstrends.label_tuning.base_tuner import BaseLabelTuner


class RemainingValueTuner(BaseLabelTuner):
    """
    A tuner that calculates the remaining value change in intervals of continuous labels.

    For each point in the time series, it calculates the absolute value change
    from the current position to the end of the current trend interval.

    Attributes:
        normalize (bool): Whether to normalize the remaining value change to a [-1, 1] range.
    """

    def __init__(self):
        """
        Initialize the RemainingValueTuner.

        """

    def tune(
        self,
        time_series: list[float],
        labels: list[int],
        enforce_monotonicity: bool = False,
        normalize_over_interval: bool = False,
        shift_periods: int = 0,
    ) -> list[float]:
        """
        Tune trend labels to provide information about remaining value change.

        For each point in the time series, calculates how much the value will change
        until the end of the current trend interval. The sign of the result matches
        the original label.

        Args:
            time_series (list[float]): The price series used for trend detection.
            labels (list[int]): The original trend labels (-1, 1) or (-1, 0, 1).
            enforce_monotonicity (bool, optional): If True, the labels in each interval will not reverse on uncaptured countertrends.
            normalize_over_interval (bool, optional): If True, the remaining value change will be normalized over the interval.
            shift_periods (int, optional): The number of periods to shift the labels forward (if positive) or backward (if negative).

        Returns:
            list[float]: Enhanced labels with information about remaining value change:
                       - For uptrends (1): positive values indicating remaining uplift
                       - For downtrends (-1): negative values indicating remaining downside
                       - For neutral trends (0): values close to zero
        """
        self._verify_inputs(time_series, labels)

        # Convert inputs to numpy arrays for vectorized operations
        ts_array = np.array(time_series)
        labels_array = np.array(labels)

        intervals = list(pairwise(self._find_trend_intervals(labels)))
        result = np.zeros(len(time_series))

        for start, end in intervals:
            if labels_array[end] == 0:
                continue

            interval_slice = slice(start, end)
            end_value = ts_array[end]

            if enforce_monotonicity:
                cum_func = (
                    np.minimum.accumulate
                    if labels_array[end] == -1
                    else np.maximum.accumulate
                )
                reference_values = cum_func(ts_array[interval_slice])
            else:
                reference_values = ts_array[interval_slice]

            interval_values = end_value - reference_values

            if normalize_over_interval:
                interval_values = self._normalize_values(interval_values)

            result[interval_slice] = interval_values

        # Roll and pad with zeros if necessary
        result = np.roll(result, shift_periods)
        if shift_periods > 0:
            result[:shift_periods] = 0
        elif shift_periods < 0:
            result[-shift_periods:] = 0

        return result.tolist()

    def _find_trend_intervals(self, labels: list[int]) -> list[int]:
        """
        Find the first index of each continuous label interval in the label series.

        Args:
            labels (list[int]): The original trend labels (-1, 1) or (-1, 0, 1).

        Returns:
            list[int]: List of indices where each interval starts, including 0.
        """
        # Start with 0 as first interval always starts at beginning
        change_indices = [0]

        # Add indices where values change (start of new intervals)
        change_indices.extend(
            i + 1 for i in range(len(labels) - 1) if labels[i] != labels[i + 1]
        )

        return change_indices

    def _normalize_values(self, values: np.ndarray) -> np.ndarray:
        """
        Normalize values to a [-1, 1] range while preserving the sign.

        Args:
            values (np.ndarray): Array of values to normalize.

        Returns:
            np.ndarray: Normalized values in [-1, 1] range.
        """
        if not values.any():  # More idiomatic check for all zeros
            return values

        max_abs = np.abs(values).max()  # More concise max absolute value
        return np.clip(values / max_abs if max_abs > 0 else values, -1.0, 1.0)
