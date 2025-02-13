from abc import ABC, abstractmethod
from typing import List


class BaseLabeller(ABC):
    """
    Abstract base class for trend labellers.

    All trend labellers should inherit from this class and implement the get_labels method.
    The class provides common input validation functionality.
    """

    def _verify_time_series(self, time_series_list: list[float]) -> None:
        """
        Verify that the input time series is valid.

        Args:
            time_series_list (list[float]): The price series to verify.

        Raises:
            TypeError: If time_series_list is not a list or contains non-numeric values.
            ValueError: If time_series_list is empty or too short.
        """
        if not isinstance(time_series_list, list):
            raise TypeError("time_series_list must be a list.")
        if not all(isinstance(price, (int, float)) for price in time_series_list):
            raise TypeError(
                "All elements in time_series_list must be integers or floats."
            )
        if len(time_series_list) < 2:
            raise ValueError("time_series_list must contain at least two elements.")

    @abstractmethod
    def get_labels(self, time_series_list: list[float]) -> list[int]:
        """
        Label trends in a time series of prices.

        Args:
            time_series_list (list[float]): List of prices to label.

        Returns:
            list[int]: List of trend labels.
        """
        pass
