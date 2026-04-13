"""Unit tests for efficiency-based forward-looking filters."""

import numpy as np
import pytest

from tstrends.label_tuning.filtering import ForwardLookingFilter


class TestForwardLookingFilter:
    """Tests for ForwardLookingFilter intra-trend weighting."""

    @pytest.fixture
    def filter_default(self) -> ForwardLookingFilter:
        return ForwardLookingFilter(
            forward_window=5,
            smoothing_window=3,
            quantile=0.9,
        )

    def test_clean_uptrend_high_weights(
        self, filter_default: ForwardLookingFilter
    ) -> None:
        """Monotonic move has high efficiency; core region stays near ceiling."""
        prices = [100.0 + float(i) for i in range(30)]
        labels = [1] * len(prices)
        w = filter_default.get_coefficients(prices, labels)
        assert w.min() >= 0.0 - 1e-9
        assert w.max() <= 1.0 + 1e-9
        h = 5  # matches ``filter_default`` forward_window
        # Avoid boundary rolls from undefined tail efficiencies and convolution edges
        core = slice(h + 2, -(h + 2))
        assert np.median(w[core]) > 0.85

    def test_plateau_weights_near_zero(
        self, filter_default: ForwardLookingFilter
    ) -> None:
        """Flat prices imply ~0 efficiency → weights near zero after normalization."""
        prices = [100.0] * 12
        labels = [1] * len(prices)
        w = filter_default.get_coefficients(prices, labels)
        assert np.allclose(w, 0.0, atol=0.05)

    def test_choppy_between_plateau_and_clean(
        self, filter_default: ForwardLookingFilter
    ) -> None:
        """Oscillating path has lower efficiency than a straight ramp."""
        ramp = np.arange(20, dtype=float)
        choppy = np.array([0, 2, 0, 2, 0, 2] * 3 + [0, 2], dtype=float)
        labels_r = [1] * len(ramp)
        labels_c = [1] * len(choppy)
        w_ramp = filter_default.get_coefficients(ramp.tolist(), labels_r)
        w_choppy = filter_default.get_coefficients(choppy.tolist(), labels_c)
        core = slice(5, 15)
        assert np.mean(w_ramp[core]) > np.mean(w_choppy[core])

    def test_short_interval_all_ones(self) -> None:
        """When interval length <= forward_window, coefficients are 1.0."""
        flt = ForwardLookingFilter(forward_window=10, smoothing_window=3)
        prices = [1.0, 2.0, 3.0]
        labels = [1, 1, 1]
        w = flt.get_coefficients(prices, labels)
        assert np.allclose(w, 1.0)

    def test_coefficients_bounded(self, filter_default: ForwardLookingFilter) -> None:
        """All weights lie in [0, 1] after robust normalization."""
        np.random.seed(0)
        prices = (100.0 + np.cumsum(np.random.randn(30) * 0.5)).tolist()
        labels = [1] * len(prices)
        w = filter_default.get_coefficients(prices, labels)
        assert (w >= 0.0 - 1e-9).all() and (w <= 1.0 + 1e-9).all()

    def test_apply_matches_elementwise_product(
        self, filter_default: ForwardLookingFilter
    ) -> None:
        """apply() equals values * get_coefficients()."""
        prices = list(np.linspace(10.0, 20.0, 15))
        labels = [-1] * len(prices)
        values = np.linspace(1.0, 2.0, len(prices))
        w = filter_default.get_coefficients(prices, labels)
        applied = filter_default.filter(values, prices, labels)
        assert np.allclose(applied, values * w)

    def test_forward_window_changes_coefficients(self) -> None:
        """A wider forward horizon changes the efficiency profile vs a narrow one."""
        prices = [100.0 + 0.1 * (-1) ** i for i in range(20)]
        labels = [1] * len(prices)
        narrow = ForwardLookingFilter(forward_window=3, smoothing_window=3, quantile=0.9)
        wide = ForwardLookingFilter(forward_window=10, smoothing_window=3, quantile=0.9)
        w_narrow = narrow.get_coefficients(prices, labels)
        w_wide = wide.get_coefficients(prices, labels)
        assert not np.allclose(w_narrow, w_wide)

    def test_neutral_interval_passthrough(
        self, filter_default: ForwardLookingFilter
    ) -> None:
        """Label=0 intervals keep weight 1.0."""
        prices = [1.0, 1.0, 1.0, 50.0, 51.0, 52.0]
        labels = [0, 0, 0, 1, 1, 1]
        w = filter_default.get_coefficients(prices, labels)
        assert np.allclose(w[:3], 1.0)

    @pytest.mark.parametrize(
        "prices,labels,error_type",
        [
            ([], [1, 2, 3], ValueError),
            ([1, 2, 3], [], ValueError),
            ([1, 2], [1, 2, 3], ValueError),
            ([1, "2", 3], [1, 1, 1], TypeError),
            ([1, 2, 3], [1, 2, 3], ValueError),
        ],
    )
    def test_invalid_inputs(
        self,
        filter_default: ForwardLookingFilter,
        prices: object,
        labels: object,
        error_type: type[BaseException],
    ) -> None:
        with pytest.raises(error_type):
            filter_default.get_coefficients(prices, labels)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"forward_window": 0, "smoothing_window": 3},
            {"forward_window": 5, "smoothing_window": 0},
            {"forward_window": 5, "smoothing_window": 3, "quantile": 0.0},
            {"forward_window": 5, "smoothing_window": 3, "quantile": 1.5},
            {
                "forward_window": None,
                "forward_window_rel": 0.0,
                "smoothing_window": 3,
            },
            {
                "forward_window": None,
                "forward_window_rel": 1.0,
                "smoothing_window": 3,
            },
            {
                "forward_window": 5,
                "smoothing_window": None,
                "smoothing_window_rel": 0.0,
            },
            {
                "forward_window": 5,
                "smoothing_window": None,
                "smoothing_window_rel": 1.0,
            },
        ],
    )
    def test_invalid_constructor(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            ForwardLookingFilter(**kwargs)

    def test_apply_length_mismatch(self, filter_default: ForwardLookingFilter) -> None:
        with pytest.raises(ValueError, match="same length"):
            filter_default.filter([1.0, 2.0], [1.0], [1])
