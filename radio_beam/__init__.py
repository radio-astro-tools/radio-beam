# Licensed under a 3-clause BSD style license - see LICENSE.rst

from ._astropy_init import __version__, test

from pkg_resources import get_distribution, DistributionNotFound

from .beam import (Beam, EllipticalGaussian2DKernel,
                   EllipticalTophat2DKernel)
from .multiple_beams import Beams

__all__ = ['Beam', 'EllipticalTophat2DKernel',
           'EllipticalGaussian2DKernel', 'Beams']
