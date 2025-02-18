import numpy as np
import pytest

from time_series_trends.trend_labelling.base_labeller import BaseLabeller
from time_series_trends.trend_labelling.label_scaling import Labels


class ConcreteLabeller(BaseLabeller):
    def get_labels(
        self, time_series_list: list[float], return_labels_as_int: bool = True
    ) -> list[int] | list[Labels]:
        if return_labels_as_int:
            return [0] * len(time_series_list)
        return [Labels.NEUTRAL] * len(time_series_list)


@pytest.fixture()
def labeller():
    return ConcreteLabeller()


class TestBaseLabeller:

    def test_verify_time_series_valid_input(self, labeller):
        valid_series = [1.0, 2.0, 3.0]
        labeller._verify_time_series(valid_series)

    def test_verify_time_series_mixed_numeric(self, labeller):
        mixed_series = [1, 2.0, 3, 4.5]
        labeller._verify_time_series(mixed_series)

    def test_verify_time_series_non_list_input(self, labeller):
        with pytest.raises(TypeError, match="time_series_list must be a list."):
            labeller._verify_time_series((1.0, 2.0, 3.0))

    def test_verify_time_series_non_numeric_elements(self, labeller):
        with pytest.raises(
            TypeError,
            match="All elements in time_series_list must be integers or floats.",
        ):
            labeller._verify_time_series([1.0, "2.0", 3.0])

    def test_verify_time_series_nan_values(self, labeller):
        with pytest.raises(
            TypeError,
            match="time_series_list cannot contain NaN values.",
        ):
            labeller._verify_time_series([1.0, np.nan, 3.0])

    def test_verify_time_series_empty_list(self, labeller):
        with pytest.raises(
            ValueError, match="time_series_list must contain at least two elements."
        ):
            labeller._verify_time_series([])

    def test_verify_time_series_single_element(self, labeller):
        with pytest.raises(
            ValueError, match="time_series_list must contain at least two elements."
        ):
            labeller._verify_time_series([1.0])

    def test_abstract_class_instantiation(self):
        with pytest.raises(TypeError):
            BaseLabeller()

    def test_get_labels_returns_ints_by_default(self, labeller):
        """Test that get_labels returns integers by default."""
        result = labeller.get_labels([1.0, 2.0, 3.0])
        assert all(isinstance(x, int) for x in result)

    def test_get_labels_returns_labels_when_requested(self, labeller):
        """Test that get_labels returns Labels enum values when return_labels_as_int is False."""
        result = labeller.get_labels([1.0, 2.0, 3.0], return_labels_as_int=False)
        assert all(isinstance(x, Labels) for x in result)

    def test_get_labels_correct_length(self, labeller):
        """Test that get_labels returns correct length for both return types."""
        time_series = [1.0, 2.0, 3.0]
        int_result = labeller.get_labels(time_series, return_labels_as_int=True)
        enum_result = labeller.get_labels(time_series, return_labels_as_int=False)
        assert len(int_result) == len(time_series)
        assert len(enum_result) == len(time_series)
