"""Unit tests for temporal Shifter post-processor."""

import numpy as np
import pytest

from tstrends.label_tuning.shifting import Shifter


class TestShifter:
    """Tests for Shifter in the postprocessor pipeline."""

    def test_forward_shift(self) -> None:
        shifter = Shifter(2)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = shifter.process(values, [], [])
        assert np.allclose(out, [0.0, 0.0, 1.0, 2.0, 3.0])

    def test_backward_shift(self) -> None:
        shifter = Shifter(-2)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = shifter.process(values, [], [])
        assert np.allclose(out, [3.0, 4.0, 5.0, 0.0, 0.0])

    def test_process_ignores_time_series_and_labels(self) -> None:
        """Shifter is purely index-based; dummies should not affect output."""
        shifter = Shifter(1)
        values = [10.0, 20.0]
        out = shifter.process(values, ["ignored"], [99])
        assert np.allclose(out, [0.0, 10.0])

    @pytest.mark.parametrize("bad", [0, 1.5, "2"])
    def test_constructor_rejects_invalid_periods(self, bad: object) -> None:
        with pytest.raises((TypeError, ValueError)):
            Shifter(bad)
