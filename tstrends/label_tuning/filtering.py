"""
Efficiency-based intra-trend weighting for tuned labels.

Within each non-neutral trend interval, weights emphasize timesteps where
forward-looking directional efficiency (net move over path length) is high.
"""

from itertools import pairwise

import numpy as np

from tstrends.label_tuning.base import BaseFilter

_EPS = 1e-8


def _parse_window_pair(
    abs_val: int | None,
    rel_val: float | None,
    *,
    abs_param: str,
    rel_param: str,
) -> tuple[int | None, float | None]:
    """Validate mutually exclusive absolute / relative window and return the pair."""
    if abs_val is not None:
        if not isinstance(abs_val, int) or abs_val < 1:
            raise ValueError(f"{abs_param} must be an integer >= 1")
        return abs_val, None
    if rel_val is not None:
        fr = float(rel_val)
        if fr <= 0.0 or fr >= 1.0:
            raise ValueError(f"{rel_param} must be a float in (0, 1)")
        return None, fr
    raise ValueError(f"either {abs_param} or {rel_param} must be provided")


def _window_length_for_interval(
    abs_val: int | None,
    rel_val: float | None,
    interval_len: int,
) -> int:
    """Map stored abs/rel spec to an integer window for an interval of length ``n``."""
    if abs_val is not None:
        return abs_val
    n = interval_len
    return max(1, min(n - 1, int(round(rel_val * n))))


class ForwardLookingFilter(BaseFilter):
    """
    Per-timestep weights from a normalized forward-looking efficiency metric.

    For timestep ``t`` inside an interval (with horizon ``h``):

    .. math::

        e_t = \\frac{|x_{t+h} - x_t|}{\\sum_{k=1}^{h} |x_{t+k} - x_{t+k-1}| + \\epsilon}

    Weights are smoothed to avoid propagating time series noise into the labels and scaled by a
    high quantile (avoiding a max to add robustness) so that low-efficiency regions are downweighted but not zeroed.
    """

    def __init__(
        self,
        forward_window: int | None = None,
        forward_window_rel: float | None = None,
        smoothing_window: int | None = None,
        smoothing_window_rel: float | None = None,
        quantile: float = 0.95,
    ) -> None:
        """
        Args:
            forward_window: Absolute horizon ``h`` for efficiency (net vs path length).
            forward_window_rel: Relative horizon as a fraction of each trend interval length
                (mutually exclusive with ``forward_window``).
            smoothing_window: Length of the centered moving-average kernel on the filtering weights.
            smoothing_window_rel: Relative length of the centered moving-average kernel on the filtering weights, as a fraction of each trend interval length
                (mutually exclusive with ``smoothing_window``).
            quantile: High quantile for robust normalization denominator.
        """
        self._forward_spec = _parse_window_pair(
            forward_window,
            forward_window_rel,
            abs_param="forward_window",
            rel_param="forward_window_rel",
        )
        self._smooth_spec = _parse_window_pair(
            smoothing_window,
            smoothing_window_rel,
            abs_param="smoothing_window",
            rel_param="smoothing_window_rel",
        )
        if not isinstance(quantile, (int, float)) or not 0.0 < float(quantile) <= 1.0:
            raise ValueError("quantile must be in (0, 1]")
        self.quantile = float(quantile)

    def _compute_efficiency(self, x: np.ndarray, h: int) -> np.ndarray:
        """Raw efficiency ``e``; trailing ``h`` positions are zero (undefined horizon)."""
        n = len(x)
        e = np.zeros(n, dtype=float)
        abs_diff = np.abs(np.diff(x))
        cumsum = np.concatenate(([0.0], np.cumsum(abs_diff)))
        path = cumsum[h:] - cumsum[:-h]
        net = np.abs(x[h:] - x[:-h])
        e[: n - h] = net / (path + _EPS)
        return e

    def _smooth_efficiency(self, e: np.ndarray) -> np.ndarray:
        """Centered moving average (same length as ``e``)."""
        window = _window_length_for_interval(*self._smooth_spec, len(e))
        kernel = np.ones(window, dtype=float) / window
        return np.convolve(e, kernel, mode="same")

    def _robust_normalize(self, e: np.ndarray) -> np.ndarray:
        """Scale by high quantile and clip to ``[0, 1]``."""
        denom = float(np.quantile(e, q=self.quantile))
        w = e / (denom + _EPS)
        return np.clip(w, 0.0, 1.0)

    def _weights_for_interval(self, interval_ts: np.ndarray) -> np.ndarray:
        """Full stabilization pipeline for one contiguous interval."""
        x = np.asarray(interval_ts, dtype=float)
        n = len(x)
        if n == 0:
            return np.array([], dtype=float)

        h = _window_length_for_interval(*self._forward_spec, n)
        if n <= h:
            return np.ones(n, dtype=float)

        e = self._compute_efficiency(x, h)
        e_smooth = self._smooth_efficiency(e)
        return self._robust_normalize(e_smooth)

    def get_coefficients(
        self,
        time_series: list[float] | np.ndarray,
        labels: list[int] | np.ndarray,
    ) -> np.ndarray:
        self._verify_inputs(time_series, labels)
        ts_array = np.asarray(time_series, dtype=float)
        labels_array = np.asarray(labels)

        coefficients = np.ones(len(time_series), dtype=float)
        bounds = self._find_trend_intervals(labels)
        n_labels = len(labels_array)
        for start, end in pairwise(bounds):
            if labels_array[start] == 0:
                continue
            # ``end`` is the last index of the series for the final pair; otherwise it
            # is the first index of the next segment (exclusive upper bound).
            stop_exclusive = end + 1 if end == n_labels - 1 else end
            interval_slice = slice(start, stop_exclusive)
            interval_ts = ts_array[interval_slice]
            coefficients[interval_slice] = self._weights_for_interval(interval_ts)

        return coefficients
