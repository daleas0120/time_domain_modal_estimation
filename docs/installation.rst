Installation
============

Requirements
------------

* Python >= 3.8
* NumPy >= 1.20.0
* Matplotlib >= 3.3.0

Standard Installation
---------------------

From Source
~~~~~~~~~~~

Clone the repository and install:

.. code-block:: bash

   git clone https://github.com/daleas0120/time_domain_modal_estimation.git
   cd time_domain_modal_estimation
   pip install .

Development Installation
------------------------

For development work, install in editable mode:

.. code-block:: bash

   pip install -e .

With Development Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To install with testing and linting tools:

.. code-block:: bash

   pip install -e ".[dev]"

With Documentation Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To build the documentation locally:

.. code-block:: bash

   pip install -e ".[docs]"

All Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~

To install everything:

.. code-block:: bash

   pip install -e ".[dev,docs]"

Verifying Installation
----------------------

After installation, verify that the package is working:

.. code-block:: python

   import time_domain_modal_estimation
   print(time_domain_modal_estimation.__version__)

Or run a simple test:

.. code-block:: bash

   python examples/basic_usage.py

Building from Source
--------------------

To build distribution packages:

.. code-block:: bash

   pip install build
   python -m build

This creates wheel and source distributions in the ``dist/`` directory.
