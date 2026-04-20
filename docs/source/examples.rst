Usage Examples
====================

This page provides detailed examples of using TStrends in various scenarios, as a quickstart overview of the right syntax and capabilities of the package.

The labelling results might not be interesting nor significant, since these are just dummy time series.

.. note::
   For interactive examples with visualization and more interesting labelling results, check out the :doc:`notebook_examples` page
   which contains Jupyter notebooks with runnable code and detailed explanations.

Trend Labelling Examples
----------------------------
All labellers are parametrized by one or more parameters, which tune up their behavior.

Binary CTL with Different Omega Values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `omega` parameter in Binary CTL controls the sensitivity to trend changes. Here's how different values affect the labelling:

.. code-block:: python

   import numpy as np
   from tstrends.trend_labelling import BinaryCTL
   import matplotlib.pyplot as plt

   # Generate sample data (labellers require a Python list of ints/floats, not an ndarray)
   np.random.seed(42)
   prices = (np.random.randn(100).cumsum() + 100).tolist()

   # Test different omega values
   omega_values = [0.001, 0.005, 0.01, 0.015]
   labels = {}

   for omega in omega_values:
       labeller = BinaryCTL(omega=omega)
       labels[omega] = labeller.get_labels(prices)

Ternary CTL Parameter Effects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `TernaryCTL` labeller uses two main parameters:

.. code-block:: python

   import numpy as np
   from tstrends.trend_labelling import TernaryCTL

   np.random.seed(42)
   prices = (np.random.randn(100).cumsum() + 100).tolist()

   # Different parameter combinations
   configs = [
       {'marginal_change_thres': 0.01, 'window_size': 10},
       {'marginal_change_thres': 0.02, 'window_size': 20},
       {'marginal_change_thres': 0.005, 'window_size': 5}
   ]

   for config in configs:
       labeller = TernaryCTL(**config)
       labels = labeller.get_labels(prices)

Returns Estimation Examples
----------------------------------
Return estimators provide a convenient way to calculate the extent to which the labelling set up captures the actual time series shifts. 
A simple return estimator is provided by the `SimpleReturnEstimator` class, while the `ReturnsEstimatorWithFees` class allows for the estimation of returns taking into account transaction and holding costs.

Simple Returns vs Transaction Costs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compare returns with and without transaction costs:

.. code-block:: python

   from tstrends.trend_labelling import BinaryCTL
   from tstrends.returns_estimation import (
       SimpleReturnEstimator,
       ReturnsEstimatorWithFees,
       FeesConfig,
   )

   prices = [100.0 + 0.5 * i for i in range(50)]
   labels = BinaryCTL(omega=0.01).get_labels(prices)

   # Simple returns
   simple_estimator = SimpleReturnEstimator()
   simple_returns = simple_estimator.estimate_return(prices, labels)

   # Returns with fees
   fees_config = FeesConfig(
       lp_transaction_fees=0.001,
       sp_transaction_fees=0.001,
       lp_holding_fees=0.0001,
       sp_holding_fees=0.0001
   )
   fees_estimator = ReturnsEstimatorWithFees(fees_config)
   fees_returns = fees_estimator.estimate_return(prices, labels)

Position-Specific Fee Structures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Example with different fees for long and short positions:

.. code-block:: python

   from tstrends.trend_labelling import BinaryCTL
   from tstrends.returns_estimation import ReturnsEstimatorWithFees, FeesConfig

   prices = [100.0 + 0.5 * i for i in range(50)]
   labels = BinaryCTL(omega=0.01).get_labels(prices)

   # Asymmetric fees configuration
   asymmetric_fees = FeesConfig(
       lp_transaction_fees=0.001,  # 0.1% for long positions
       sp_transaction_fees=0.002,  # 0.2% for short positions
       lp_holding_fees=0.0001,    # Lower holding fee for longs
       sp_holding_fees=0.0002     # Higher holding fee for shorts
   )

   estimator = ReturnsEstimatorWithFees(asymmetric_fees)
   returns = estimator.estimate_return(prices, labels)

Parameter Optimization Example
--------------------------------

The optimization process allows to find the labeller parameters that lead to the highest returns, given a specific returns estimator. Therefore, tweaking the fees structure (for instance to penalise downtrends more) will lead to different optimal parameters and labelling behaviour.

Optimize parameters for a single time series:

.. code-block:: python

   import numpy as np
   from tstrends.optimization import Optimizer
   from tstrends.returns_estimation import SimpleReturnEstimator
   from tstrends.trend_labelling import BinaryCTL

   np.random.seed(42)
   prices = (np.random.randn(100).cumsum() + 100).tolist()

   # Create optimizer (pass a returns estimator instance, not the class)
   optimizer = Optimizer(
       returns_estimator=SimpleReturnEstimator(),
       initial_points=5,
       nb_iter=100,
   )

   # Define custom bounds
   bounds = {
       'omega': (0.001, 0.1)
   }

   # Optimize
   result = optimizer.optimize(
       labeller_class=BinaryCTL,
       time_series_list=prices,
       bounds=bounds,
       verbose=1
   )

Label Tuning Examples
--------------------------------

Label tuning transforms discrete trend labels into continuous values that express the potential of the trend at each point. Optional **postprocessors** run in order after the magnitudes are computed: use :class:`~tstrends.label_tuning.filtering.ForwardLookingFilter` to downweight low forward-looking efficiency within each trend leg, :class:`~tstrends.label_tuning.shifting.Shifter` to align signals in time, and smoothers from :mod:`tstrends.label_tuning.smoothing` to pool neighboring values.

Remaining Value Tuning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `RemainingValueTuner` transforms labels based on the difference between the current value and the maximum/minimum value reached by the end of the trend:

.. code-block:: python

   import numpy as np
   from tstrends.trend_labelling import OracleTernaryTrendLabeller
   from tstrends.label_tuning import RemainingValueTuner
   from tstrends.label_tuning.smoothing import LinearWeightedAverage

   np.random.seed(42)
   prices = (np.random.randn(80).cumsum() + 100).tolist()

   # Generate trend labels
   labeller = OracleTernaryTrendLabeller(transaction_cost=0.006, neutral_reward_factor=0.03)
   labels = labeller.get_labels(prices)

   # Create a smoother for enhancing the tuned labels (optional)
   smoother = LinearWeightedAverage(window_size=5, direction="left")

   # Tune the labels (optional smoother via postprocessors)
   tuner = RemainingValueTuner(postprocessors=[smoother])
   tuned_labels = tuner.tune(
       time_series=prices,
       labels=labels,
       enforce_monotonicity=True,
       normalize_over_interval=False,
   )

Forward-looking filter: tuning down early signal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A **binary** segmentation can still yield rich float labels if you pass :class:`~tstrends.label_tuning.filtering.ForwardLookingFilter` (and optional smoothers) as ``postprocessors`` to :class:`~tstrends.label_tuning.RemainingValueTuner`. After remaining-value magnitudes are computed, the filter rescales them using a forward-looking **efficiency** measure: strong net move over the next few steps, relative to the path length, keeps the signal; weak or choppy forward paths damp it. Horizons can be set in bars or as a fraction of each non-neutral interval.

Typical pattern: :class:`~tstrends.trend_labelling.OracleBinaryTrendLabeller` for labels, relative filter windows, then a left-looking moving average as the last postprocessor (postprocessors run in order).

.. code-block:: python

   import numpy as np
   from tstrends.trend_labelling import OracleBinaryTrendLabeller
   from tstrends.label_tuning import RemainingValueTuner, ForwardLookingFilter
   from tstrends.label_tuning.smoothing import SimpleMovingAverage

   np.random.seed(42)
   prices = (np.random.randn(120).cumsum() + 100).tolist()
   labels = OracleBinaryTrendLabeller(transaction_cost=0.006).get_labels(prices)

   forward_filter = ForwardLookingFilter(
       forward_window_rel=0.2,
       smoothing_window_rel=0.1,
   )
   # Wider windows suit longer series; the label_tuner_example notebook uses 150 on synthetic data.
   smoother = SimpleMovingAverage(window_size=25, direction="left")

   tuner = RemainingValueTuner(
       postprocessors=[forward_filter, smoother],
   )
   tuned_labels = tuner.tune(
       time_series=prices,
       labels=labels,
       normalize_over_interval=True,
   )

If your downstream model accepts floats, this keeps **binary** labeller settings (often simpler to tune than multi-class labellers) while encoding “how much is left, weighted by near-term path quality” without extra discrete label states.

Absolute windows, shifting, and pipeline order
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use integer ``forward_window`` and ``smoothing_window`` instead of the ``*_rel`` arguments, optionally chain :class:`~tstrends.label_tuning.shifting.Shifter` so positive ``periods`` shift values later in time (earlier indices padded with zero), and mix in other smoothers—each step receives the output of the previous one:

.. code-block:: python

   import numpy as np
   from tstrends.label_tuning import (
       RemainingValueTuner,
       ForwardLookingFilter,
       Shifter,
   )
   from tstrends.label_tuning.smoothing import LinearWeightedAverage
   from tstrends.trend_labelling import OracleTernaryTrendLabeller

   np.random.seed(42)
   prices = (np.random.randn(80).cumsum() + 100).tolist()
   labeller = OracleTernaryTrendLabeller(transaction_cost=0.006, neutral_reward_factor=0.03)
   labels = labeller.get_labels(prices)

   forward_filter = ForwardLookingFilter(
       forward_window=5,
       smoothing_window=3,
       quantile=0.95,
   )
   shifter = Shifter(periods=2)
   smoother = LinearWeightedAverage(window_size=5, direction="left")

   tuner = RemainingValueTuner(
       postprocessors=[forward_filter, shifter, smoother],
   )
   tuned_labels = tuner.tune(
       time_series=prices,
       labels=labels,
       enforce_monotonicity=True,
       normalize_over_interval=False,
   )

Smoothing Options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply different smoothing options to the tuned labels:

.. code-block:: python

   from tstrends.label_tuning.smoothing import SimpleMovingAverage, LinearWeightedAverage

   # Example tuned series (e.g. from RemainingValueTuner); list of floats
   tuned_labels = [0.1, 0.2, 0.15, -0.1, -0.05] * 10

   # Simple moving average (equal weights)
   simple_smoother = SimpleMovingAverage(window_size=5, direction="left")
   simple_smoothed = simple_smoother.smooth(tuned_labels)

   # Linear weighted average (higher weights on more recent values)
   weighted_smoother = LinearWeightedAverage(window_size=5, direction="centered")
   weighted_smoothed = weighted_smoother.smooth(tuned_labels)

Real-World Application
---------------------------

Here's a complete example combining all components for a real-world scenario:

.. code-block:: python

   import yfinance as yf
   from tstrends.trend_labelling import TernaryCTL
   from tstrends.returns_estimation import ReturnsEstimatorWithFees, FeesConfig
   from tstrends.optimization import Optimizer

   # Download stock data
   stock = yf.Ticker("AAPL")
   df = stock.history(period="1y")
   prices = df["Close"].tolist()

   # Setup fees configuration
   fees_config = FeesConfig(
       lp_transaction_fees=0.001,
       sp_transaction_fees=0.001,
       lp_holding_fees=0.0001,
       sp_holding_fees=0.0001
   )

   # Create returns estimator
   estimator = ReturnsEstimatorWithFees(fees_config)

   # Create and configure optimizer
   optimizer = Optimizer(
       returns_estimator=estimator,
       initial_points=10,
       nb_iter=200
   )

   # Optimize parameters
   result = optimizer.optimize(
       labeller_class=TernaryCTL,
       time_series_list=prices
   )

   # Create labeller with optimal parameters
   optimal_labeller = TernaryCTL(**result['params'])
   labels = optimal_labeller.get_labels(prices)

   # Calculate final returns
   final_returns = estimator.estimate_return(prices, labels)

   print(f"Optimal parameters: {result['params']}")
   print(f"Total returns: {final_returns}") 