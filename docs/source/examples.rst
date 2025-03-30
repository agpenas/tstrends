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

   # Generate sample data
   np.random.seed(42)
   prices = np.random.randn(100).cumsum() + 100

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

   from tstrends.trend_labelling import TernaryCTL

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

   from tstrends.returns_estimation import (
       SimpleReturnEstimator,
       ReturnsEstimatorWithFees,
       FeesConfig
   )

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

   from tstrends.parameter_optimization import Optimizer
   from tstrends.trend_labelling import BinaryCTL

   # Create optimizer
   optimizer = Optimizer(
       returns_estimator=SimpleReturnEstimator,
       initial_points=5,
       nb_iter=100
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

Real-World Application
---------------------------

Here's a complete example combining all components for a real-world scenario:

.. code-block:: python

   import yfinance as yf
   from tstrends.trend_labelling import TernaryCTL
   from tstrends.returns_estimation import ReturnsEstimatorWithFees, FeesConfig
   from tstrends.parameter_optimization import Optimizer

   # Download stock data
   stock = yf.Ticker("AAPL")
   df = stock.history(period="1y")
   prices = df['Close'].values

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