Installing ``radio-beam``
============================

Requirements
------------

This package has the following dependencies:

* `Python <http://www.python.org>`_ 3.6 or later
* `Numpy <http://www.numpy.org>`_ 1.8 or later
* `Astropy <http://www.astropy.org>`__ 1.0 or later

Installation
------------

To install the latest stable release, you can type::

    pip install radio-beam

or you can download the latest tar file from
`PyPI <https://pypi.python.org/pypi/radio-beam>`_ and install it using::

    pip install -e .

Developer version
-----------------

If you want to install the latest developer version of the radio-beam code, you
can do so from the git repository::

    git clone https://github.com/radio-astro-tools/radio-beam.git
    cd radio-beam
    pip install -e .

You may need to add the ``--user`` option to the last line `if you do not
have root access <https://docs.python.org/2/install/#alternate-installation-the-user-scheme>`_.
You can also install the latest developer version in a single line with pip::

    pip install git+https://github.com/radio-astro-tools/radio-beam.git

Installing into CASA
--------------------

For use with CASA, please use the modular versions of CASA described `here <https://casadocs.readthedocs.io/en/stable/notebooks/introduction.html#id1>`_.
