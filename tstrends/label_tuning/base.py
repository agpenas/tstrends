from abc import ABC, abstractmethod

import numpy as np

from tstrends.label_tuning.smoothing_direction import Direction


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
    def tune(
        self, time_series: list[float], labels: list[int], **kwargs
    ) -> list[float]:
        """
        Tune trend labels to provide more information about trend magnitude.

        Args:
            time_series (list[float]): The price series used for trend detection.
            labels (list[int]): The original trend labels (-1, 1) or (-1, 0, 1).

        Returns:
            list[float]: Enhanced labels with additional information about trend magnitude.
        """
        pass


class BaseSmoother(ABC):
    """
    Abstract base class for all label smoothers.

    Label smoothers take tuned label values and apply various smoothing techniques,
    particularly to transfer trend signals to earlier time points.

    Attributes:
        window_size (int): Size of the smoothing window.
    """

    def __init__(self, window_size: int = 3, direction: str | Direction = "left"):
        """
        Initialize the smoother with a window size.

        Args:
            window_size (int): Number of periods to include in the smoothing window.
            direction (Union[str, Direction]): Direction of smoothing, either "left" or "centered".
                Can be provided as string or Direction enum.

        Raises:
            ValueError: If window_size < 2 or direction is invalid.
            TypeError: If direction is not a string or Direction enum.
        """
        if not isinstance(window_size, int) or window_size < 2:
            raise ValueError("window_size must be a positive integer >= 2")
        self.window_size = window_size

        # Validate direction type and value
        if isinstance(direction, str):
            try:
                self.direction = Direction(direction)
            except ValueError:
                raise ValueError(
                    f"direction must be one of {[d.value for d in Direction]}"
                )
        elif isinstance(direction, Direction):
            self.direction = direction
        else:
            raise TypeError("direction must be a string or Direction enum")

    @abstractmethod
    def smooth(self, values: list[float]) -> np.ndarray:
        """
        Apply smoothing to the input values.

        Args:
            values (list[float]): The input values to smooth.

        Returns:
            np.ndarray: The smoothed values, same length as input.
        """
        pass
