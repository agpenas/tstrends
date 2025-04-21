"""
Direction enums for smoothing operations.

This module provides the Direction enum used to specify smoothing directions
in time series processing.
"""

from enum import Enum


class Direction(Enum):
    """Direction of smoothing to apply.

    The direction determines which values are included in the smoothing window
    relative to the current point being smoothed.

    Values:
        LEFT: Only uses past values (causal smoothing).
              For point at time t, uses points [t-n+1, t-n+2, ..., t].


        CENTERED: Uses both past and future values (non-causal smoothing).
              For point at time t, uses points [t-n/2, ..., t, ..., t+n/2].

    Example:
        For a time series [10, 20, 30, 40, 50] with window_size=3:
        - LEFT smoothing for point at index 2 (value 30) uses [10, 20, 30]
        - CENTERED smoothing for point at index 2 (value 30) uses [20, 30, 40]
    """

    LEFT = "left"
    CENTERED = "centered"
