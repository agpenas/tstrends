from typing import Optional, Type, Union

from bayes_opt import BayesianOptimization, acquisition

from ..returns_estimation import BaseReturnEstimator
from ..trend_labelling import BaseLabeller

from .optimization_bounds import OptimizationBounds

# Constants for parameter types
INTEGER_PARAMS = ["window_size"]


class Optimizer:
    """Bayesian optimization for trend labelling parameters.

    This class implements Bayesian optimization to find the optimal parameters
    for trend labelling algorithms. It uses the returns from a returns estimator
    as the objective function to maximize.

    Attributes:
        returns_estimator (Type[BaseReturnEstimator]): The returns estimator class to use.
        initial_points (int): Number of initial random points for optimization.
        nb_iter (int): Number of optimization iterations.
        random_state (int | None): Random seed for reproducibility.
        _optimizer (BayesianOptimization | None): Internal optimizer instance.

    Example:
        >>> from returns_estimation import SimpleReturnEstimator
        >>> optimizer = Optimizer(SimpleReturnEstimator, initial_points=5, nb_iter=100)
        >>> result = optimizer.optimize(BinaryCTL, prices)
        >>> print(result['params'])  # {'omega': 0.005}

    Note:
        The optimization process uses Bayesian optimization with Gaussian processes,
        which is particularly effective for expensive-to-evaluate objective functions
        with a small number of parameters.
    """

    def __init__(
        self,
        returns_estimator: Type[BaseReturnEstimator],
        initial_points: int = 10,
        nb_iter: int = 1_000,
        random_state: int | None = None,
    ) -> None:
        self.returns_estimator = returns_estimator
        self.initial_points = initial_points
        self.nb_iter = nb_iter
        self.random_state = random_state
        self._optimizer = None

    def get_optimizer(self) -> BayesianOptimization:
        """
        Get the underlying BayesianOptimization object if you need access to the full optimization history.

        Returns:
            BayesianOptimization: The optimizer instance with optimization results.

        Raises:
            ValueError: If optimize() hasn't been called yet.
        """
        if self._optimizer is None:
            raise ValueError(
                "Optimizer not initialized. You need to call the optimize method first."
            )
        return self._optimizer

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

    def optimize(
        self,
        labeller_class: Type[BaseLabeller],
        time_series_list: Union[list[float], list[list[float]]],
        bounds: Optional[dict[str, tuple[float, float]]] = None,
        acquisition_function: Optional[Type[acquisition.AcquisitionFunction]] = None,
        verbose: Optional[int] = 0,
    ) -> dict[str, Union[dict, float]]:
        """
        Optimize the trend labelling parameters.

        Args:
            labeller_class (Type[BaseLabeller]): The trend labeller class to optimize.
            time_series_list (Union[list[float], list[list[float]]]): Either a single time series list or a list of time series lists
                to optimize the trend labelling parameters on.
            bounds (Optional[dict[str, tuple[float, float]]], optional): The bounds of the parameters to optimize.
                If not provided, the bounds will be the default bounds. Defaults to None.
            acquisition_function (Optional[Type[acquisition.AcquisitionFunction]], optional): The acquisition function to use.
                If not provided, the default acquisition function UpperConfidenceBound(kappa=2) will be used.
            verbose (Optional[int], optional): Verbosity level for optimization output. Defaults to 0.

        Returns:
            dict[str, Union[dict, float]]: A dictionary containing:
                - 'params': Dictionary of optimal parameters
                - 'target': The maximum target value achieved
        """
        if acquisition_function is None:
            acquisition_function = acquisition.UpperConfidenceBound(
                kappa=2, random_state=self.random_state
            )
        bounds = bounds or OptimizationBounds().get_bounds(labeller_class)

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

        self._optimizer = BayesianOptimization(
            f=objective_function,
            pbounds=bounds,
            verbose=verbose,
            random_state=self.random_state,
            acquisition_function=acquisition_function,
        )
        self._optimizer.maximize(init_points=self.initial_points, n_iter=self.nb_iter)

        # Return the optimization results directly
        return {
            "params": self._process_parameters(self._optimizer.max["params"]),
            "target": self._optimizer.max["target"],
        }
