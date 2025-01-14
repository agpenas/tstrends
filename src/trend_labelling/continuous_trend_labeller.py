from typing import List
from .base_labeller import BaseLabeller


class BinaryCTL(BaseLabeller):
    """Continuous Trend Labeller class, adapted to Python from the original paper by Wu, D., Wang, X., Su, J., Tang, B., & Wu, S. "A Labeling Method for Financial Time Series Prediction Based on Trends". https://doi.org/10.3390/e22101162"""

    def __init__(self, omega: float) -> None:
        """
        Initialize the continuous trend labeller.

        Args:
            omega (float): The proportion threshold parameter of the trend definition.
        """
        if not isinstance(omega, float):
            raise TypeError("omega must be a float.")
        self.omega = omega

    def get_labels(self, time_series_list: List[float]) -> List[int]:
        """
        Auto-labels a price time series based on the provided algorithm.

        Parameters:
        time_series_list (List[float]): The original time series data X = [x1, x2, ..., xN]

        Returns:
        List[int]: The label vector Y = [label1, label2, ..., labelN]. Possible values for labels are 1 (uptrend), 0 (no trend), and -1 (downtrend).
        """
        self._verify_time_series(time_series_list)

        ts_len = len(time_series_list)
        labels = [0] * ts_len  # Initialize the label vector

        # Initialization of related variables
        first_price = time_series_list[0]  # First price
        current_high, curr_high_time = time_series_list[0], 0  # Highest price and time
        current_low, curr_low_time = time_series_list[0], 0  # Lowest price and time
        current_direction = 0  # Current direction of labeling
        extreme_point_idx = 0  # Index of the highest or lowest point initially

        # First loop to determine the initial direction and significant point
        for i, price in enumerate(time_series_list):

            if price > first_price * (1 + self.omega):
                current_high, curr_high_time, extreme_point_idx, current_direction = (
                    price,
                    i,
                    i,
                    1,
                )
                break

            elif price < first_price * (1 - self.omega):
                current_low, curr_low_time, extreme_point_idx, current_direction = (
                    price,
                    i,
                    i,
                    -1,
                )
                break

        # Second loop to label the rest of the time series
        for i in range(extreme_point_idx + 1, ts_len):
            if current_direction > 0:  # Uptrend
                if time_series_list[i] > current_high:
                    # Update the current high point and time
                    current_high, curr_high_time = time_series_list[i], i
                if (
                    time_series_list[i] < current_high - current_high * self.omega
                    and curr_low_time <= curr_high_time
                ):
                    # Label the time series between the current high and low points as uptrend
                    for j in range(curr_low_time + 1, curr_high_time + 1):
                        labels[j] = 1
                    # Update the current low point and time, and change the direction to downtrend
                    current_low, curr_low_time, current_direction = (
                        time_series_list[i],
                        i,
                        -1,
                    )

            elif current_direction < 0:  # Downtrend
                if time_series_list[i] < current_low:
                    current_low, curr_low_time = time_series_list[i], i
                if (
                    time_series_list[i] > current_low + current_low * self.omega
                    and curr_high_time <= curr_low_time
                ):
                    # Label the time series between the current high and low points as downtrend
                    for j in range(curr_high_time + 1, curr_low_time + 1):
                        labels[j] = -1
                    # Update the current high point and time, and change the direction to uptrend
                    current_high, curr_high_time, current_direction = (
                        time_series_list[i],
                        i,
                        1,
                    )

        return labels
