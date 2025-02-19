import math
from abc import ABC, abstractmethod
from typing import TypeVar

from .label_scaling import Labels

T = TypeVar("T", list[int], list[Labels])


class BaseLabeller(ABC):
    """Abstract base class for trend labellers.

    This class serves as a template for all trend labelling implementations.
    It provides common input validation functionality and defines the interface
    that all trend labellers must implement.

    Attributes:
        None

    Example:
        To implement a new trend labeller, inherit from this class and implement
        the get_labels method::

            class MyLabeller(BaseLabeller):
                def get_labels(self, time_series_list, return_labels_as_int=True):
                    # Implementation here
                    pass
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
        if any(math.isnan(price) for price in time_series_list):
            raise TypeError("time_series_list cannot contain NaN values.")

        if len(time_series_list) < 2:
            raise ValueError("time_series_list must contain at least two elements.")

    @abstractmethod
    def get_labels(
        self, time_series_list: list[float], return_labels_as_int: bool = True
    ) -> T:
        """
        Label trends in a time series of prices.

        Args:
            time_series_list (list[float]): List of prices to label.
            return_labels_as_int (bool, optional): If True, returns integer labels (-1, 0, 1),
                                                  if False returns Labels enum values. Defaults to True.

        Returns:
            Union[list[int], list[Labels]]: List of trend labels, either as integers or Labels enum values.
        """
        pass
