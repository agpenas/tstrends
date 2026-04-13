from abc import ABC, abstractmethod

import numpy as np

from tstrends.label_tuning.smoothing_direction import Direction


def verify_time_series_and_labels(
    time_series: list[float] | np.ndarray,
    labels: list[int] | np.ndarray,
) -> None:
    """
    Verify that time series and labels are valid for label tuning or filtering.

    Args:
        time_series: The price series.
        labels: The trend labels (-1, 1) or (-1, 0, 1).

    Raises:
        TypeError: If inputs are not lists/arrays or contain invalid values.
        ValueError: If inputs are empty or have incompatible lengths.
    """
    if not isinstance(
        time_series, (list, np.ndarray)
    ):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError("time_series must be a list or numpy array.")
    if len(time_series) == 0:
        raise ValueError("time_series cannot be empty.")
    if not all(
        isinstance(price, (int, float)) for price in time_series
    ):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError("All elements in time_series must be numeric.")

    if not isinstance(
        labels, (list, np.ndarray)
    ):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError("labels must be a list or numpy array.")
    if len(labels) == 0:
        raise ValueError("labels cannot be empty.")
    if not all(label in (-1, 0, 1) for label in labels):
        raise ValueError("labels must only contain values -1, 0, or 1.")

    if len(time_series) != len(labels):
        raise ValueError("time_series and labels must have the same length.")


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
        verify_time_series_and_labels(time_series, labels)

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


class BasePostprocessor(ABC):
    """Common interface for post-processing tuned label values.

    Filters, smoothers, and shifters implement :meth:`process` and can be chained
    by :class:`tstrends.label_tuning.RemainingValueTuner` via a ``postprocessors`` list.
    """

    @abstractmethod
    def process(
        self,
        values: list[float] | np.ndarray,
        time_series: list[float] | np.ndarray,
        labels: list[int] | np.ndarray,
    ) -> np.ndarray:
        """
        Transform ``values`` in the tuner pipeline.

        Args:
            values: Tuned magnitudes aligned with ``time_series``.
            time_series: The price series (unused by some subclasses).
            labels: The trend labels (-1, 0, 1); unused by some subclasses.

        Returns:
            Transformed values as a float array of the same length as ``values``.
        """
        pass


class BaseFilter(BasePostprocessor):
    """Abstract base class for per-timestep filters on tuned label values.

    Filters compute coefficients in ``[0, 1]`` (or a bounded range after flooring)
    that can be multiplied element-wise with tuned magnitudes to emphasize or
    suppress regions within each trend interval.
    """

    def _verify_inputs(
        self,
        time_series: list[float] | np.ndarray,
        labels: list[int] | np.ndarray,
    ) -> None:
        """Validate ``time_series`` and ``labels`` (same rules as :class:`BaseLabelTuner`)."""
        verify_time_series_and_labels(time_series, labels)

    def _find_trend_intervals(self, labels: list[int] | np.ndarray) -> list[int]:
        """
        Find indices that bound contiguous runs of equal labels.

        Returns starts of each interval plus the last series index, matching
        :meth:`RemainingValueTuner._find_trend_intervals`.

        Args:
            labels: Trend labels (-1, 0, 1).

        Returns:
            Sorted indices: first index of each run, then ``len(labels) - 1``.
        """
        change_indices = [0]
        label_list = list(labels)
        change_indices.extend(
            i + 1
            for i in range(len(label_list) - 1)
            if label_list[i] != label_list[i + 1]
        )
        return change_indices + [len(label_list) - 1]

    @abstractmethod
    def get_coefficients(
        self,
        time_series: list[float] | np.ndarray,
        labels: list[int] | np.ndarray,
    ) -> np.ndarray:
        """
        Compute per-timestep multiplicative coefficients.

        Args:
            time_series: The price series used for efficiency (or other) metrics.
            labels: The trend labels (-1, 0, 1).

        Returns:
            One-dimensional array of coefficients, same length as inputs.
        """
        pass

    def filter(
        self,
        values: list[float] | np.ndarray,
        time_series: list[float] | np.ndarray,
        labels: list[int] | np.ndarray,
    ) -> np.ndarray:
        """
        Multiply ``values`` by ``get_coefficients(time_series, labels)``.

        Args:
            values: Tuned or other values aligned with ``time_series``.
            time_series: The price series.
            labels: The trend labels.

        Returns:
            Element-wise product as a float array.

        Raises:
            ValueError: If ``values`` length differs from ``time_series``.
        """
        self._verify_inputs(time_series, labels)
        values_array = np.asarray(values, dtype=float)
        if values_array.shape[0] != len(time_series):
            raise ValueError("values must have the same length as time_series.")
        coefficients = self.get_coefficients(time_series, labels)
        return values_array * coefficients

    def process(
        self,
        values: list[float] | np.ndarray,
        time_series: list[float] | np.ndarray,
        labels: list[int] | np.ndarray,
    ) -> np.ndarray:
        return self.filter(values, time_series, labels)


class BaseSmoother(BasePostprocessor):
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
        if (
            not isinstance(window_size, int)
            or window_size < 2  # pyright: ignore[reportUnnecessaryIsInstance]
        ):
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
        elif isinstance(
            direction, Direction
        ):  # pyright: ignore[reportUnnecessaryIsInstance]
            self.direction = direction
        else:
            raise TypeError(
                "direction must be a string or Direction enum"
            )  # pyright: ignore[reportUnreachable]

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

    def process(
        self,
        values: list[float] | np.ndarray,
        time_series: list[float] | np.ndarray,
        labels: list[int] | np.ndarray,
    ) -> np.ndarray:
        _ = (time_series, labels)  # unused; signature kept for pipeline uniformity
        vals: list[float]
        if isinstance(values, np.ndarray):
            vals = values.astype(float).tolist()
        else:
            vals = [float(v) for v in values]
        return self.smooth(vals)
