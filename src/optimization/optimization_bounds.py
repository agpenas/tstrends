from typing import Type
from trend_labelling import (
    BaseLabeller,
    BinaryCTL,
    TernaryCTL,
    OracleBinaryTrendLabeller,
    OracleTernaryTrendLabeller,
)


class OptimizationBounds:
    """Class to provide default bounds for optimization parameters."""

    implemented_labellers = [
        BinaryCTL,
        TernaryCTL,
        OracleBinaryTrendLabeller,
        OracleTernaryTrendLabeller,
    ]

    def get_bounds(
        self, labeller_class: Type[BaseLabeller]
    ) -> dict[str, tuple[float, float]]:
        """
        Get the default bounds for a given labeller class.

        Args:
            labeller_class (Type[BaseLabeller]): The labeller class to get bounds for.

        Returns:
            dict[str, tuple[float, float]]: A dictionary mapping parameter names to their bounds.

        Raises:
            ValueError: If the labeller class is not supported.
        """
        if labeller_class not in self.implemented_labellers:
            raise ValueError(f"No default bounds for labeller class {labeller_class}")
        if labeller_class == BinaryCTL:
            return {"omega": (0.0, 0.01)}
        elif labeller_class == TernaryCTL:
            return {
                "marginal_change_thres": (0.000001, 0.1),
                "window_size": (1, 5000),
            }
        elif labeller_class == OracleBinaryTrendLabeller:
            return {"transaction_cost": (0.0, 0.01)}
        elif labeller_class == OracleTernaryTrendLabeller:
            return {
                "transaction_cost": (0.0, 0.01),
                "trend_coeff": (0.0, 0.1),
            }
