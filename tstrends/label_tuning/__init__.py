"""
Label tuning module for enhancing trend labels with magnitude information.

This module provides tools to tune trend labels (UP/NEUTRAL/DOWN) by adding
information about the potential trend magnitude or remaining potential until
the next trend change.
"""

from tstrends.label_tuning.base import BasePostprocessor
from tstrends.label_tuning.filtering import ForwardLookingFilter
from tstrends.label_tuning import RemainingValueTuner
from tstrends.label_tuning.shifting import Shifter
from tstrends.label_tuning.smoothing import SimpleMovingAverage, LinearWeightedAverage
from tstrends.label_tuning.smoothing_direction import Direction

__all__ = [
    "BasePostprocessor",
    "ForwardLookingFilter",
    "RemainingValueTuner",
    "Shifter",
    "SimpleMovingAverage",
    "LinearWeightedAverage",
    "Direction",
]
