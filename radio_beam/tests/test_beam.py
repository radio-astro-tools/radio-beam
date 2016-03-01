# Licensed under a 3-clause BSD style license - see LICENSE.rst
from .. import beam as radio_beam
from astropy.io import fits
from astropy import units as u
import os
import numpy as np

def data_path(filename):
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    return os.path.join(data_dir, filename)

def test_classic_header():
    # Instantiate from header
    fname = data_path("NGC0925.bima.mmom0.fits.gz")
    hdr = fits.getheader(fname)
    bima_beam_file = radio_beam.Beam.from_fits_header(fname)
    bima_beam_hdr = radio_beam.Beam.from_fits_header(hdr)

def test_from_aips_test():
    aips_fname = data_path("ngc0925_na.fits.gz")
    aips_hdr = fits.getheader(aips_fname)
    aips_beam_hdr = radio_beam.Beam.from_fits_header(aips_hdr)
    np.testing.assert_almost_equal(aips_beam_hdr.sr.value, 9.029858054819811e-10)
    aips_beam_file = radio_beam.Beam.from_fits_header(aips_fname)
    np.testing.assert_almost_equal(aips_beam_file.sr.value, 9.029858054819811e-10)

def test_from_casa_test():
    casa_fname = data_path("m83.moment0.fits.gz")
    casa_hdr = fits.getheader(casa_fname)
    casa_beam_hdr = radio_beam.Beam.from_fits_header(casa_hdr)
    np.testing.assert_almost_equal(casa_beam_hdr.sr.value, 2.98323984597532e-11)
    casa_beam_file = radio_beam.Beam.from_fits_header(casa_fname)
    np.testing.assert_almost_equal(casa_beam_file.sr.value, 2.98323984597532e-11)

def test_manual():
    # Instantiate from command line
    man_beam_val = radio_beam.Beam(0.1, 0.1, 30)
    np.testing.assert_almost_equal(man_beam_val.value, 3.451589629868801e-06)
    man_beam_rad = radio_beam.Beam(0.1*u.rad, 0.1*u.rad, 30*u.deg)
    np.testing.assert_almost_equal(man_beam_rad.value, 0.011330900354567986)
    man_beam_deg = radio_beam.Beam(0.1*u.deg, 0.1*u.deg, 1.0*u.rad)
    np.testing.assert_almost_equal(man_beam_deg.value, 3.451589629868801e-06)

# def test_deconv():
#     # Deconvolution and convolution
#     beam_1 = radio_beam.Beam(10.*u.arcsec, 5.*u.arcsec, 30.*u.deg)
#     beam_2 = radio_beam.Beam(5.*u.arcsec, 3.*u.arcsec, 120.*u.deg)

#     beam_3 = beam_1.convolve(beam_2)
#     print "test1: ",beam_2 == beam_3.deconvolve(beam_1)
#     print "test1: ",beam_2 - beam_3.deconvolve(beam_1)
#     print "test2: ",beam_1 == beam_3.deconvolve(beam_2)
#     print "test2: ",beam_1 - beam_3.deconvolve(beam_2)
#     print "beam1: ",beam_1
#     print "beam2: ",beam_2
#     print "beam3: ",beam_3
#     print "beam3.deconv(beam2): ",beam_3.deconvolve(beam_2)
#     np.testing.assert_almost_equal(beam_3.deconvolve(beam_2).sr.value, beam_1.sr.value)

#     # Area
#     print beam_3.sr
#     print beam_2.sr
#     #  <Quantity 3.994895404940742e-10 sr>
#     print beam_1.sr
#     print beam_1

#     # Janskies to Kelvin
#     np.testing.assert_almost_equal(beam_2.jtok(1.e9), 81474.701386)
    # 81474

    # Return as array

    # Return as array given WCS
