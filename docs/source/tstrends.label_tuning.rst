Label Tuning
====================

The label tuning module enhances trend labels (UP/NEUTRAL/DOWN) by adding information about the potential trend magnitude, making them more useful for training prediction models.

.. contents:: Contents
   :local:

RemainingValueTuner
-------------------

.. autoclass:: tstrends.label_tuning.RemainingValueTuner
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