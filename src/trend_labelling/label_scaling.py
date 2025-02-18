"""Label scaling utilities for trend labelling."""

from enum import IntEnum

import numpy as np
from numpy.typing import NDArray


class Labels(IntEnum):
    """Standard label values for trend classification."""

    DOWN = -1
    NEUTRAL = 0
    UP = 1


# Explicit mappings from input labels to standardized values
BINARY_MAP = {0: Labels.DOWN, 1: Labels.UP}

TERNARY_MAP = {0: Labels.DOWN, 1: Labels.NEUTRAL, 2: Labels.UP}


def scale_binary(labels: NDArray) -> NDArray:
    """Scale binary labels from {0,1} to {-1,1}.

    Args:
        labels (NDArray): Input labels (must contain only 0s and 1s)

    Returns:
        NDArray: Scaled labels in {-1,1}
    """
    return np.vectorize(BINARY_MAP.get)(labels)


def scale_ternary(labels: NDArray) -> NDArray:
    """Scale ternary labels from {0,1,2} to {-1,0,1}.

    Args:
        labels (NDArray): Input labels (must contain only 0s, 1s, and 2s)

    Returns:
        NDArray: Scaled labels in {-1,0,1}
    """
    return np.vectorize(TERNARY_MAP.get)(labels)


def extract_label_values(labels: list[Labels]) -> list[int]:
    """Convert a list of Labels enum to their integer values.

    Args:
        labels (list[Labels]): List of Labels enum values.

    Returns:
        list[int]: List of integer values corresponding to the Labels enum values.
    """
    return [label.value for label in labels]
