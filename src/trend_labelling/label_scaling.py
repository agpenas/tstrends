from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray


class LabelScaler(ABC):
    """Abstract base class for label scaling strategies"""

    @abstractmethod
    def scale(self, labels: NDArray) -> NDArray:
        """Scale labels to a specific range

        Args:
            labels (NDArray): Input labels to scale

        Returns:
            NDArray: Scaled labels
        """
        raise NotImplementedError("Subclasses must implement this method")


class DefaultLabelScaler(LabelScaler):
    """Scales labels to range [-1, 1]"""

    range_width = 2
    offset = 1

    def scale(self, labels: NDArray) -> NDArray:
        """Scale labels to range [-1, 1]

        Args:
            labels (NDArray): Input labels to scale

        Returns:
            NDArray: Labels scaled to [-1, 1] range

        Raises:
            ValueError: If labels array contains all zeros
        """

        if np.count_nonzero(labels) < 1:
            return labels
        max_val = np.abs(labels).max()
        return (labels / max_val) * self.range_width - self.offset
