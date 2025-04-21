"""
Smoothing implementations for trend labels.

This module provides various smoothing algorithm implementations
for trend label processing.
"""

from typing import Optional, Union

import numpy as np
from scipy import signal

from tstrends.label_tuning.base import BaseSmoother
from tstrends.label_tuning.smoothing_direction import Direction


class SimpleMovingAverage(BaseSmoother):
    """Simple moving average smoother with equal weights.

    This smoother applies equal weights to all values in the window, resulting in
    uniform smoothing. Each point in the window has the same influence on the result.

    Examples:
        For a window size of 3:
        - With values [10, 20, 30], each value gets 1/3 weight (33.3%)
        - Result = (10 * 0.333) + (20 * 0.333) + (30 * 0.333) = 20

    This approach produces gradual smoothing with consistent lag across all frequencies,
    making it good for removing noise but less responsive to recent changes.
    """

    def __init__(self, window_size: int = 3, direction: Union[str, Direction] = "left"):
        super().__init__(window_size, direction)

    def smooth(self, values: list[float]) -> np.ndarray:
        array = np.asarray(values)
        window = np.ones(self.window_size) / self.window_size

        if self.direction == Direction.LEFT:
            # Left-sided (causal) moving average
            smoothed = np.convolve(array, window, mode="full")[-len(array) :]
            return smoothed

        # Centered moving average
        smoothed = np.convolve(array, window, mode="same")
        return smoothed


class LinearWeightedAverage(BaseSmoother):
    """Linear weighted moving average smoother.

    This smoother applies linearly increasing weights to values in the window,
    giving more importance to recent values and less to older ones. This creates
    a more responsive smoothing that better preserves the shape of trends.

    Examples:
        For a window size of 3 with left-sided smoothing:
        - With values [10, 20, 30], weights are distributed as:
          - Oldest value (10): 1/6 weight (16.7%)
          - Middle value (20): 2/6 weight (33.3%)
          - Recent value (30): 3/6 weight (50.0%)
        - Result = (10 * 0.167) + (20 * 0.333) + (30 * 0.5) = 23.33

    Compared to SimpleMovingAverage, LinearWeightedAverage:
    - Responds more quickly to recent changes
    - Reduces lag in trend detection
    - Better preserves the shape of peaks and valleys
    - More effective for early trend detection
    """

    def __init__(self, window_size: int = 3, direction: Union[str, Direction] = "left"):
        super().__init__(window_size, direction)

    def smooth(self, values: list[float]) -> np.ndarray:
        array = np.asarray(values)

        if self.direction == Direction.LEFT:
            # Linear weights increasing toward most recent value
            weights = np.arange(1, self.window_size + 1)
            weights = weights / weights.sum()

            # Apply convolution and align with original array
            smoothed = np.convolve(array, weights, mode="full")[-len(array) :]
            return smoothed

        # Triangular weights centered on each point
        weights = signal.windows.triang(self.window_size)
        weights = weights / weights.sum()
        smoothed = np.convolve(array, weights, mode="same")

        return smoothed
