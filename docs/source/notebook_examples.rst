Notebook Examples
====================

These interactive notebooks provide in-depth examples of TStrends functionality. 

.. note::
   The full notebooks contain many interactive elements and visualizations which may take time to load.
   Below are links to each notebook with a brief description of what they contain.

Trend Labelling Catalogue
--------------------------

This notebook demonstrates all the available trend labellers and shows their behavior across different parameters.

* **Content**: Examples of Binary CTL, Ternary CTL, Binary Oracle, and Ternary Oracle labellers
* **Visualizations**: Multiple plots showing labelling results with different parameters
* **Advanced Topics**: Parameter sensitivity analysis

`View the Trend Labelling Catalogue on GitHub <https://github.com/agpenas/tstrends/blob/main/notebooks/labellers_catalogue.ipynb>`__

How to Use Simple Labellers
----------------------------

Learn how to use the basic trend labelling functionality in TStrends.

* **Content**: Step-by-step guide to using labellers
* **Examples**: Finding good parameters for labellers
* **Advanced Topics**: Return estimation with different fee structures

`View the Simple Labeller Tutorial on GitHub <https://github.com/agpenas/tstrends/blob/main/notebooks/simple_labeller_howto.ipynb>`__

Parameter Optimization Example
------------------------------

This notebook shows how to optimize trend labeller parameters using Bayesian optimization.

* **Content**: Optimizing Ternary Oracle labeller parameters
* **Examples**: Applying optimization to different time series
* **Advanced Topics**: Handling different time series characteristics

`View the Parameter Optimization Example on GitHub <https://github.com/agpenas/tstrends/blob/main/notebooks/optimization_example.ipynb>`__

Label Tuning Example
------------------------------

This notebook demonstrates how to transform discrete trend labels into continuous values expressing trend potential.

* **Content**: Using the ``RemainingValueTuner`` to enhance trend labels and optional postprocessor pipelines
* **Examples**: Tuning parameters, smoothing, forward-looking filters, and temporal shifts
* **Advanced Topics**: Visualizing the effect of tuning on prediction models

`View the Label Tuning Example on GitHub <https://github.com/agpenas/tstrends/blob/main/notebooks/label_tuner_example.ipynb>`__
