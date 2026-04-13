Label Tuning
====================

The label tuning module enhances trend labels (UP/NEUTRAL/DOWN) by adding information about the potential trend magnitude, making them more useful for training prediction models.

After raw magnitudes are computed, :class:`~tstrends.label_tuning.remaining_value_tuner.RemainingValueTuner` can run an ordered list of **postprocessors**—filters, shifters, and smoothers—that all implement :class:`tstrends.label_tuning.base.BasePostprocessor` (see the :mod:`tstrends.label_tuning.base` section in the package API).

.. contents:: Contents
   :local:

RemainingValueTuner
-------------------

.. autoclass:: tstrends.label_tuning.RemainingValueTuner
   :members:
   :undoc-members:
   :show-inheritance:

Forward-looking filter
----------------------

.. autoclass:: tstrends.label_tuning.ForwardLookingFilter
   :members:
   :undoc-members:
   :show-inheritance:

Shifter
-------

.. autoclass:: tstrends.label_tuning.Shifter
   :members:
   :undoc-members:
   :show-inheritance:

Smoothing
---------

.. autoclass:: tstrends.label_tuning.smoothing.SimpleMovingAverage
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: tstrends.label_tuning.smoothing.LinearWeightedAverage
   :members:
   :undoc-members:
   :show-inheritance:

Smoothing Direction
------------------

.. autoclass:: tstrends.label_tuning.smoothing_direction.Direction
   :members:
   :undoc-members:
   :show-inheritance: 