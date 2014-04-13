#!/usr/bin/env python

import sys
if 'build_sphinx' in sys.argv or 'develop' in sys.argv:
    from setuptools import setup, Command
else:
    from distutils.core import setup, Command

with open('README.rst') as file:
    long_description = file.read()

with open('CHANGES') as file:
    long_description += file.read()

# no versions yet from agpy import __version__ as version

class PyTest(Command):
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        import sys,subprocess
        errno = subprocess.call([sys.executable, 'runtests.py'])
        raise SystemExit(errno)

execfile('radio_beam/version.py')

setup(name='radio_beam',
      version=__version__,
      description='Radio Beam object for use with astropy',
      long_description=long_description,
      author='Adam Leroy, Adam Ginsburg, Erik Rosolowsky, and Tom Robitaille',
      author_email='adam.g.ginsburg@gmail.com',
      url='https://github.com/akleroy/radio_beam',
      packages=['radio_beam'], 
      cmdclass = {'test': PyTest},
     )
