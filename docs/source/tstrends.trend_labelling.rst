Trend Labelling
==========================
This module provides tools for labelling trends in time series data.

.. currentmodule:: tstrends.trend_labelling

Continuous Trend Labellers
----------------------------------

.. autoclass:: BinaryCTL
   :members: get_labels, get_state
   :special-members: __init__
   :exclude-members: _verify_time_series, _state, _labels

.. autoclass:: TernaryCTL
   :members: get_labels, get_state
   :special-members: __init__
   :exclude-members: _verify_time_series

Oracle Labellers
------------------------------

.. autoclass:: OracleBinaryTrendLabeller
   :members: get_labels, get_state
   :special-members: __init__
   :exclude-members: _verify_time_series

.. autoclass:: OracleTernaryTrendLabeller
   :members: get_labels, get_state
   :special-members: __init__
   :exclude-members: _verify_time_series

Enums
----------

.. autoclass:: Labels
   :members:
   :undoc-members:

