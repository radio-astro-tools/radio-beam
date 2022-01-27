# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest

from astropy.io import fits
from astropy import units as u
import os
import warnings
import numpy as np
import numpy.testing as npt
from astropy.tests.helper import assert_quantity_allclose
from itertools import product

from astropy.modeling.models import Gaussian2D

from ..beam import Beam
from ..utils import BeamError
from ..source_deconvolution import deconvolve_source


def test_simple_source_deconvolution():

    source = Gaussian2D(amplitude=1., x_stddev=2 * u.deg,
                        y_stddev=2*u.deg)
    mybeam = Beam(1 * u.deg)

    deconv_source = deconvolve_source(source, mybeam)

    # Deconvolution between 2 circular Gaussians is:
    # np.sqrt(fwhm**2 - beam_major**2)

    npt.assert_allclose(deconv_source.x_stddev.value, np.sqrt(2**2 - 1))
    npt.assert_allclose(deconv_source.y_stddev.value, np.sqrt(2**2 - 1))
    assert deconv_source.theta.value == 0.

@pytest.mark.parametrize(("x_stddev", "y_stddev", "theta"),
                         [(2, 1.5, 60),
                          (2, 2.5, -120),
                          (2, 2.1, -300),
                          (1.05, 1.01, 240)])
def test_simple_source_deconvolution_ellipse(x_stddev, y_stddev, theta):

    source = Gaussian2D(amplitude=1.,
                        x_stddev=x_stddev * u.deg,
                        y_stddev=y_stddev * u.deg,
                        theta=theta * u.deg)
    mybeam = Beam(1 * u.deg)

    deconv_source = deconvolve_source(source, mybeam)

    # Deconvolution between 2 circular Gaussians is:
    # np.sqrt(fwhm**2 - beam_major**2)

    npt.assert_allclose(deconv_source.x_stddev.value, np.sqrt(x_stddev**2 - mybeam.major.value))
    npt.assert_allclose(deconv_source.y_stddev.value, np.sqrt(y_stddev**2 - mybeam.major.value))

    npt.assert_allclose(deconv_source.theta.value, theta % 180.)
