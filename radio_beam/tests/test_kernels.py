# Licensed under a 3-clause BSD style license - see LICENSE.rst
from .. import beam as radio_beam
import numpy.testing as npt
import numpy as np


SIGMA_TO_FWHM = radio_beam.SIGMA_TO_FWHM


def test_gauss_kernel():

    fake_beam = radio_beam.Beam(10)

    # Let pixscale be 0.1 pix/deg
    kernel = fake_beam.as_kernel(0.1)

    direct_kernel = \
        radio_beam.EllipticalGaussian2DKernel(100. / SIGMA_TO_FWHM,
                                              100. / SIGMA_TO_FWHM,
                                              0.0)

    npt.assert_allclose(kernel.array, direct_kernel.array)


def test_tophat_kernel():

    fake_beam = radio_beam.Beam(10)

    # Let pixscale be 0.1 pix/deg
    kernel = fake_beam.as_tophat_kernel(0.1)

    direct_kernel = \
        radio_beam.EllipticalTophat2DKernel(100. / (SIGMA_TO_FWHM/np.sqrt(2)),
                                            100. / (SIGMA_TO_FWHM/np.sqrt(2)),
                                            0.0)

    npt.assert_allclose(kernel.array, direct_kernel.array)
