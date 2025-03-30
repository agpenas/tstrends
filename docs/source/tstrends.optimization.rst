Parameter Optimization
===========================

This module provides tools for optimizing trend labelling parameters.

.. currentmodule:: tstrends.optimization

Optimizer
----------------

.. autoclass:: Optimizer
   :members: optimize, get_best_params
   :special-members: __init__
   :exclude-members: _verify_time_series

OptimizationBounds
--------------------------------

.. autoclass:: OptimizationBounds
   :members: get_bounds
   :special-members: __init__
   :exclude-members: _verify_time_series
