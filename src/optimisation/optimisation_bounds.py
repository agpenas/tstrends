from typing import Type
from trend_labelling import (
    BaseLabeller,
    BinaryCTL,
    TernaryCTL,
    OracleBinaryTrendLabeler,
    OracleTernaryTrendLabeler,
)


class OptimisationBounds:
    """
    Class to get the bounds for the optimisation of the trend labelling parameters.
    """

    implemented_labellers = [
        BinaryCTL,
        TernaryCTL,
        OracleBinaryTrendLabeler,
        OracleTernaryTrendLabeler,
    ]

    def get_bounds(
        self, labeller_class: Type[BaseLabeller]
    ) -> dict[str, tuple[float, float]]:
        """
        Get the bounds for the optimisation of the trend labelling parameters.

        Args:
            labeller_class (Type[BaseLabeller]): The trend labeller class to get the bounds for.

        Returns:
            dict[str, tuple[float, float]]: The bounds for the trend labelling parameters.
        """
        if labeller_class not in self.implemented_labellers:
            raise ValueError(
                f"Default bounds not implemented for labeller class: {labeller_class}"
            )
        if labeller_class is BinaryCTL:
            return {"omega": (0.0, 1.0)}
        elif labeller_class is TernaryCTL:
            return {
                "marginal_change_thres": (0.0, 0.1),
                "window_size": (1, 5_000),
            }
        elif labeller_class is OracleBinaryTrendLabeler:
            return {"transaction_cost": (0.0, 1.0)}
        elif labeller_class is OracleTernaryTrendLabeler:
            return {
                "transaction_cost": (0.0, 1.0),
                "trend_coeff": (0.0, 1.0),
            }
