from typing import List


class TernaryCTL:
    """
    Three-states continuous trend labeller class, adapted from the original paper by
    Dezhkam, Arsalan et al. “A Bayesian-based classification framework for financial time series trend prediction.”
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

    def get_labels(
        self,
        time_series_list: List[float],
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
            time_series_list (List[float]): List of closing prices.

        Returns:
            List[int]: List of labels where 1 indicates an upward trend,
                    -1 indicates a downward trend, and 0 indicates no-action.
        """
        ts_len = len(time_series_list)
        labels = [0] * ts_len  # Initialize all labels as 0 (no-action)

        i = 0
        current_state = 0  # 0: no-action, 1: upward, -1: downward

        while i < ts_len - 1:

            j = i + 1

            if current_state == 0:  # No-action state

                if (
                    time_series_list[j]
                    >= time_series_list[i]
                    + self.marginal_change_thres * time_series_list[i]
                ):
                    current_state = 1

                elif (
                    time_series_list[j]
                    <= time_series_list[i]
                    - self.marginal_change_thres * time_series_list[i]
                ):
                    current_state = -1

                elif j - i > self.window_size:
                    i += 1
                    continue

            elif current_state == 1:  # Upward trend

                while j < ts_len and time_series_list[j] > time_series_list[i]:
                    labels[j] = 1
                    j += 1

                if (
                    j < ts_len
                    and time_series_list[i] - time_series_list[j]
                    >= self.marginal_change_thres * time_series_list[i]
                ):
                    current_state = -1

                elif j < ts_len and j - i > self.window_size:
                    current_state = 0

            elif current_state == -1:  # Downward trend

                while j < ts_len and time_series_list[j] < time_series_list[i]:
                    labels[j] = -1
                    j += 1

                if (
                    j < ts_len
                    and time_series_list[j]
                    >= time_series_list[i]
                    + self.marginal_change_thres * time_series_list[i]
                ):
                    current_state = 1

                elif j < ts_len and j - i > self.window_size:
                    current_state = 0

            i = j

        return labels
