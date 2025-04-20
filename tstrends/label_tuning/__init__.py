"""
Label tuning module for enhancing trend labels with magnitude information.

This module provides tools to tune trend labels (UP/NEUTRAL/DOWN) by adding
information about the potential trend magnitude or remaining potential until
the next trend change.
"""

from tstrends.label_tuning.base_tuner import BaseLabelTuner
from tstrends.label_tuning.remaining_value_tuner import RemainingValueTuner

__all__ = ["BaseLabelTuner", "RemainingValueTuner"]
