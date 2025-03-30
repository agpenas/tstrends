.. TStrends documentation master file, created by
   sphinx-quickstart on Sun Mar  2 21:28:20 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TStrends Documentation
===========================

TStrends is a Python package for automated trend labelling in time series data. The philosophy behind it is to make available in Python the state of the art in automatic trend labelling (check the :doc:`bibliography` for more details).
It provides various approaches for identifying trends, estimating returns, and optimizing parameters.

Features
----------------

* **Labelling**: Multiple trend labelling approaches:
  * Binary and ternary trend labelling
  * Continuous trend labelling (CTL)
  * Oracle-based labelling
* **Evaluating the labelling**: Returns estimation with transaction costs
* **Optimizing the labelling**: Parameter optimization using Bayesian optimization
* **Visualization utilities**: For visual inspection of the labelling results

Requirements
---------------------

* Python 3.10+
* NumPy
* Pandas
* bayesian-optimization
* matplotlib

Quick Start
--------------------

Here's a simple example of how to use TStrends:

.. code-block:: python

   from tstrends.trend_labelling import BinaryCTL
   from tstrends.returns_estimation import ReturnsEstimatorWithFees
   from tstrends.optimization import Optimizer

   # Create a trend labeller
   labeller = BinaryCTL(omega=0.1)

   # Create a returns estimator
   estimator = ReturnsEstimatorWithFees()

   # Create an optimizer
   optimizer = Optimizer()

   # Use the components together
   labels = labeller.get_labels(prices)
   returns = estimator.get_returns(prices, labels)
   optimal_params = optimizer.optimize(labeller, prices)

For more detailed examples, see the :doc:`examples` page.

Contents
----------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   examples
   notebook_examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   tstrends.trend_labelling
   tstrends.visualization
   tstrends.optimization
   tstrends.returns_estimation

.. toctree::
   :maxdepth: 2
   :caption: Development

   contributing
   bibliography

Bibliography
============================

The algorithms implemented in this package are based on or inspired by the following academic papers:

**[1]** Wu, D., Wang, X., Su, J., Tang, B., & Wu, S. (2020). **"A Labeling Method for Financial Time Series Prediction Based on Trends"**. Entropy, 22(10), 1162.
   The foundation for our Binary CTL (Continuous Trend Labelling) implementation.

**[2]** Dezhkam, A., Manzuri, M. T., Aghapour, A., Karimi, A., Rabiee, A., & Shalmani, S. M. (2023). **"A Bayesian-based classification framework for financial time series trend prediction"**. The Journal of supercomputing, 79(4), 4622–4659.
   Inspired our Ternary CTL implementation.

**[3]** Kovačević, T., Merćep, A., Begušić, S., & Kostanjčar, Z. (2023). **"Optimal Trend Labeling in Financial Time Series"**. IEEE Access. PP. 1-1.
   The basis for our Oracle Binary and Ternary Trend Labellers.

For a more comprehensive bibliography with detailed descriptions, see the :doc:`bibliography` page.

Indices and tables
============================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`