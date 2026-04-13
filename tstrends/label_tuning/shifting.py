"""
Temporal shifting of tuned label sequences.

Used as a post-processor in :class:`~tstrends.label_tuning.base.BasePostprocessor`
pipelines (e.g. :meth:`~tstrends.label_tuning.RemainingValueTuner.tune`).
"""

import numpy as np

from tstrends.label_tuning.base import BasePostprocessor


class Shifter(BasePostprocessor):
    """
    Shift tuned values forward (positive ``periods``) or backward (negative).

    Forward shift moves each value to a later index, padding the start with zeros.
    Backward shift moves values to earlier indices, padding the end with zeros.

    Attributes:
        periods: Number of steps to shift; must be non-zero. Sign sets direction.
    """

    def __init__(self, periods: int) -> None:
        if not isinstance(periods, int):
            raise TypeError("periods must be an int")
        if periods == 0:
            raise ValueError(
                "periods must be non-zero; omit Shifter from postprocessors instead."
            )
        self.periods = periods

    def process(
        self,
        values: list[float] | np.ndarray,
        time_series: list[float] | np.ndarray,
        labels: list[int] | np.ndarray,
    ) -> np.ndarray:
        _ = (time_series, labels)  # index-only transform; context is ignored
        result = np.asarray(values, dtype=float)
        shifted = np.zeros_like(result)

        if self.periods > 0:
            shifted[self.periods :] = result[: -self.periods]
        else:
            shifted[: self.periods] = result[-self.periods :]

        return shifted
