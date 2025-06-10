# Licensed under a 3-clause BSD style license - see LICENSE.rst

from .version import version as __version__

from .beam import (Beam, EllipticalGaussian2DKernel,
                   EllipticalTophat2DKernel)
from .multiple_beams import Beams

__all__ = ['Beam', 'EllipticalTophat2DKernel',
           'EllipticalGaussian2DKernel', 'Beams']
