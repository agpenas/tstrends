"""Unit tests for the smoothing direction enum.

This module contains tests for the Direction enum used in smoothing operations.
The tests verify the enum's values, creation methods, comparison behavior,
and error handling.
"""

import pytest

from tstrends.label_tuning.smoothing_direction import Direction


def test_direction_values():
    """Test that Direction enum has the expected values."""
    assert Direction.LEFT.value == "left"
    assert Direction.CENTERED.value == "centered"


def test_direction_from_string():
    """Test creating Direction enum from string values."""
    assert Direction("left") == Direction.LEFT
    assert Direction("centered") == Direction.CENTERED


def test_direction_invalid_value():
    """Test that invalid direction values raise ValueError."""
    with pytest.raises(ValueError):
        Direction("invalid")
    with pytest.raises(ValueError):
        Direction("right")  # Plausible but invalid
