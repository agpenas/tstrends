Installation
==================

Requirements
-----------------------

TStrends requires Python 3.11 or later. Tested on Python 3.11–3.14. The package dependencies are:

* numpy
* pandas
* scipy
* bayesian-optimization
* matplotlib
* pillow
* fonttools

Installing TStrends
---------------------------

You can install TStrends using pip:

.. code-block:: bash

   pip install tstrends

Development Installation
-----------------------------

For development installation:

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/agpenas/tstrends.git
      cd tstrends

2. Install development dependencies (includes the Sphinx stack needed to build the HTML docs, e.g. nbsphinx and the sitemap extension):

   .. code-block:: bash

      pip install -r requirements-dev.txt

3. Install the package in editable mode:

   .. code-block:: bash

      pip install -e . 