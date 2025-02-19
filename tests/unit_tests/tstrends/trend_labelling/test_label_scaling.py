"""Tests for label scaling functions."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from tstrends.trend_labelling.label_scaling import (
    BINARY_MAP,
    TERNARY_MAP,
    Labels,
    scale_binary,
    scale_ternary,
)


class TestBinaryScaling:
    """Test cases for binary label scaling."""

    @pytest.mark.parametrize(
        "input_labels,expected",
        [
            (np.array([0]), np.array([Labels.DOWN])),
            (np.array([1]), np.array([Labels.UP])),
            (np.array([0, 1, 0]), np.array([Labels.DOWN, Labels.UP, Labels.DOWN])),
            (np.array([1, 0, 1]), np.array([Labels.UP, Labels.DOWN, Labels.UP])),
        ],
        ids=["single_zero", "single_one", "multiple_start_zero", "multiple_start_one"],
    )
    def test_valid_binary_scaling(self, input_labels, expected):
        """Test scaling of valid binary inputs."""
        scaled = scale_binary(input_labels)
        assert_array_equal(scaled, expected)

    @pytest.mark.parametrize(
        "input_shape",
        [(2, 2), (3, 1), (1, 3), (2, 2, 2)],
        ids=lambda x: f"shape={'x'.join(str(dim) for dim in x)}",
    )
    def test_binary_preserves_shape(self, input_shape):
        """Test that binary scaling preserves input array shapes."""
        labels = np.random.randint(0, 2, size=input_shape)
        scaled = scale_binary(labels)
        assert scaled.shape == input_shape


class TestTernaryScaling:
    """Test cases for ternary label scaling."""

    @pytest.mark.parametrize(
        "input_labels,expected",
        [
            (np.array([0]), np.array([Labels.DOWN])),
            (np.array([1]), np.array([Labels.NEUTRAL])),
            (np.array([2]), np.array([Labels.UP])),
            (
                np.array([0, 1, 2]),
                np.array([Labels.DOWN, Labels.NEUTRAL, Labels.UP]),
            ),
            (
                np.array([2, 1, 0]),
                np.array([Labels.UP, Labels.NEUTRAL, Labels.DOWN]),
            ),
        ],
        ids=["down", "neutral", "up", "ascending", "descending"],
    )
    def test_valid_ternary_scaling(self, input_labels, expected):
        """Test scaling of valid ternary inputs."""
        scaled = scale_ternary(input_labels)
        assert_array_equal(scaled, expected)

    @pytest.mark.parametrize(
        "input_shape",
        [(2, 2), (3, 1), (1, 3), (2, 2, 2)],
        ids=lambda x: f"shape={'x'.join(str(dim) for dim in x)}",
    )
    def test_ternary_preserves_shape(self, input_shape):
        """Test that ternary scaling preserves input array shapes."""
        labels = np.random.randint(0, 3, size=input_shape)
        scaled = scale_ternary(labels)
        assert scaled.shape == input_shape


class TestMappingConsistency:
    """Test cases for mapping consistency."""

    def test_binary_map_values(self):
        """Test that binary map uses correct Label enum values."""
        assert BINARY_MAP[0] == Labels.DOWN
        assert BINARY_MAP[1] == Labels.UP

    def test_ternary_map_values(self):
        """Test that ternary map uses correct Label enum values."""
        assert TERNARY_MAP[0] == Labels.DOWN
        assert TERNARY_MAP[1] == Labels.NEUTRAL
        assert TERNARY_MAP[2] == Labels.UP
