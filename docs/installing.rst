Installing ``spectral-cube``
============================

Requirements
------------

This package has the following dependencies:

* `Python <http://www.python.org>`_ 2.7 or later (Python 3.x is supported)
* `Numpy <http://www.numpy.org>`_ 1.8 or later
* `Astropy <http://www.astropy.org>`__ 1.0 or later

Installation
------------

To install the latest stable release, you can type::

    pip install radio-beam

or you can download the latest tar file from
`PyPI <https://pypi.python.org/pypi/radio-beam>`_ and install it using::

    python setup.py install

Developer version
-----------------

If you want to install the latest developer version of the spectral cube code, you
can do so from the git repository::

    git clone https://github.com/radio-astro-tools/radio-beam.git
    cd radio-beam
    python setup.py install

You may need to add the ``--user`` option to the last line `if you do not
have root access <https://docs.python.org/2/install/#alternate-installation-the-user-scheme>`_.
You can also install the latest developer version in a single line with pip::

    pip install git+https://github.com/radio-astro-tools/radio-beam.git

Installing into CASA
--------------------
Installing packages in CASA is fairly straightforward.  The process is
described `here
<http://docs.astropy.org/en/stable/install.html#installing-astropy-into-casa>`_.
In short, you can do the following:

First, we need to make sure `pip <https://pypi.python.org/pypi/pip>`__ is
installed. Start up CASA as normal, and type::

    CASA <1>: from setuptools.command import easy_install

    CASA <2>: easy_install.main(['--user', 'pip'])

Now, quit CASA and re-open it, then type the following to install ``radio-beam``::

    CASA <1>: import pip

    CASA <2>: pip.main(['install', 'radio-beam', '--user'])

