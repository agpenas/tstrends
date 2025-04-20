from abc import ABC, abstractmethod
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from tstrends.trend_labelling.label_scaling import Labels

T = TypeVar("T", list[float], list[int], NDArray)


class BaseLabelTuner(ABC):
    """Abstract base class for all label tuners.

    This class serves as a template for all label tuners.
    Label tuners take standard trend labels (-1, 1) or (-1, 0, 1) and enhance them with
    additional information about the potential trend magnitude.

    Attributes:
        None
    """

    def _verify_inputs(self, time_series: list[float], labels: list[int]) -> None:
        """
        Verify that the input time series and labels are valid.

        Args:
            time_series (list[float]): The price series to use for tuning.
            labels (list[int]): The trend labels (-1, 1) or (-1, 0, 1) to tune.

        Raises:
            TypeError: If inputs are not lists or contain invalid values.
            ValueError: If inputs are empty or have incompatible lengths.
        """
        # Verify time_series
        if not isinstance(time_series, (list, np.ndarray)):
            raise TypeError("time_series must be a list or numpy array.")
        if len(time_series) == 0:
            raise ValueError("time_series cannot be empty.")
        if not all(isinstance(price, (int, float)) for price in time_series):
            raise TypeError("All elements in time_series must be numeric.")

        # Verify labels
        if not isinstance(labels, (list, np.ndarray)):
            raise TypeError("labels must be a list or numpy array.")
        if len(labels) == 0:
            raise ValueError("labels cannot be empty.")
        if not all(label in (-1, 0, 1) for label in labels):
            raise ValueError("labels must only contain values -1, 0, or 1.")

        # Verify compatibility
        if len(time_series) != len(labels):
            raise ValueError("time_series and labels must have the same length.")

    @abstractmethod
    def tune(self, time_series: list[float], labels: list[int]) -> T:
        """
        Tune trend labels to provide more information about trend magnitude.

        Args:
            time_series (list[float]): The price series used for trend detection.
            labels (list[int]): The original trend labels (-1, 1) or (-1, 0, 1).

        Returns:
            T: Enhanced labels with additional information about trend magnitude.
               The exact format depends on the specific implementation.
        """
        pass
