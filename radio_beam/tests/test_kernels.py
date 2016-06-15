# Licensed under a 3-clause BSD style license - see LICENSE.rst
from .. import beam as radio_beam
from astropy import units as u
import numpy.testing as npt
import numpy as np
import pytest
from pkg_resources import parse_version
from astropy.version import version

SIGMA_TO_FWHM = radio_beam.SIGMA_TO_FWHM

min_astropy_version = parse_version("1.1")


@pytest.mark.skipif(parse_version(version) < min_astropy_version,
                    reason="Must have astropy version >1.1")
def test_gauss_kernel():

    fake_beam = radio_beam.Beam(10)

    # Let pixscale be 0.1 deg/pix
    kernel = fake_beam.as_kernel(0.1*u.deg)

    direct_kernel = \
        radio_beam.EllipticalGaussian2DKernel(100. / SIGMA_TO_FWHM,
                                              100. / SIGMA_TO_FWHM,
                                              0.0)

    npt.assert_allclose(kernel.array, direct_kernel.array)


@pytest.mark.skipif(parse_version(version) < min_astropy_version,
                    reason="Must have astropy version >1.1")
def test_tophat_kernel():

    fake_beam = radio_beam.Beam(10)

    # Let pixscale be 0.1 deg/pix
    kernel = fake_beam.as_tophat_kernel(0.1*u.deg)

    direct_kernel = \
        radio_beam.EllipticalTophat2DKernel(100. / (SIGMA_TO_FWHM/np.sqrt(2)),
                                            100. / (SIGMA_TO_FWHM/np.sqrt(2)),
                                            0.0)

    npt.assert_allclose(kernel.array, direct_kernel.array)
