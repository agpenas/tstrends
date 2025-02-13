from typing import Callable, Tuple, Type, Union
from bayes_opt import BayesianOptimization
from bayes_opt import acquisition


from src.optimisation.optimisation_bounds import OptimisationBounds
from src.returns_estimation.returns_estimation import BaseReturnEstimator
from src.trend_labelling import BaseLabeller

# Constants for parameter types
INTEGER_PARAMS = ["window_size"]


class Optimiser:
    def __init__(
        self,
        returns_estimator: Type[BaseReturnEstimator],
        initial_points: int = 10,
        nb_iter: int = 1_000,
    ) -> None:
        self.returns_estimator = returns_estimator
        self.initial_points = initial_points
        self.nb_iter = nb_iter

    def _process_parameters(
        self, params: dict[str, float]
    ) -> dict[str, Union[int, float]]:
        """
        Process optimization parameters and convert specific parameters to required types.

        Args:
            params (dict[str, float]): Raw parameters from the optimizer.

        Returns:
            dict[str, Union[int, float]]: Processed parameters with correct types.
        """
        return {
            key: int(value) if key in INTEGER_PARAMS else value
            for key, value in params.items()
        }

    def optimise(
        self,
        labeller_class: Type[BaseLabeller],
        time_series_list: Union[list[float], list[list[float]]],
        bounds: dict[str, tuple[float, float]] = None,
        acquisition_function: Type[
            acquisition.AcquisitionFunction
        ] = acquisition.UpperConfidenceBound(kappa=2),
        verbose: int = 0,
    ) -> None:
        """
        Optimise the trend labelling parameters.

        Args:
            labeller_class (Type[BaseLabeller]): The trend labeller class to optimise.
            time_series_list (Union[list[float], list[list[float]]]): Either a single time series list or a list of time series lists
                to optimise the trend labelling parameters on.
            bounds (dict[str, tuple[float, float]], optional): The bounds of the parameters to optimise, as retrieved from the OptimisationBounds class.
                If not provided, the bounds will be the default bounds.
            acquisition_function (Type[acquisition.AcquisitionFunction], optional): The acquisition function to use.
                If not provided, the default acquisition function UpperConfidenceBound(kappa=2) will be used.
            verbose (int, optional): Verbosity level for optimization output. Defaults to 0.
        """
        bounds = bounds or OptimisationBounds().get_bounds(labeller_class)

        def objective_function(**params: dict[str, float]) -> float:
            processed_params = self._process_parameters(params)
            labeller = labeller_class(**processed_params)

            if isinstance(time_series_list[0], float):
                # First element is a float (an not a list) -> Single time series case
                return self.returns_estimator.estimate_return(
                    time_series_list,
                    labeller.get_labels(time_series_list),
                )

            # Multiple time series case
            total_return = 0.0
            for series in time_series_list:
                total_return += self.returns_estimator.estimate_return(
                    series,
                    labeller.get_labels(series),
                )
            return total_return

        self.optimiser = BayesianOptimization(
            f=objective_function,
            pbounds=bounds,
            verbose=verbose,
            acquisition_function=acquisition_function,
        )
        self.optimiser.maximize(init_points=self.initial_points, n_iter=self.nb_iter)
