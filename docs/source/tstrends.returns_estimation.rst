Returns Estimation
============================

This module provides tools for estimating returns based on price series and trend labels.

.. currentmodule:: tstrends.returns_estimation

Estimator Classes
--------------------------------

.. autoclass:: SimpleReturnEstimator
   :members:
   :special-members: __init__
   :exclude-members: _verify_time_series

.. autoclass:: ReturnsEstimatorWithFees
   :members: get_returns, estimate_return
   :special-members: __init__
   :exclude-members: _verify_time_series

Configuration
---------------------

.. autoclass:: FeesConfig
   :members:
   :special-members: __init__
   :exclude-members: __post_init__

.. py:attribute:: tstrends.returns_estimation.FeesConfig.lp_transaction_fees
   :noindex:

.. py:attribute:: tstrends.returns_estimation.FeesConfig.sp_transaction_fees
   :noindex:

.. py:attribute:: tstrends.returns_estimation.FeesConfig.lp_holding_fees
   :noindex:

.. py:attribute:: tstrends.returns_estimation.FeesConfig.sp_holding_fees
   :noindex:
