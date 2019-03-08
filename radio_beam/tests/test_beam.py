# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest

from ..beam import Beam
from astropy.io import fits
from astropy import units as u
import os
import warnings
import numpy as np
import numpy.testing as npt
from astropy.tests.helper import assert_quantity_allclose
from itertools import product

try:
    from taskinit import ia
    HAS_CASA = True
except ImportError:
    HAS_CASA = False

from ..utils import RadioBeamDeprecationWarning


data_dir = os.path.join(os.path.dirname(__file__), 'data')


def data_path(filename):
    return os.path.join(data_dir, filename)

def test_classic_header():
    # Instantiate from header
    fname = data_path("NGC0925.bima.mmom0.fits.gz")
    hdr = fits.getheader(fname)
    bima_beam_file = Beam.from_fits_header(fname)

    npt.assert_equal(bima_beam_file.major.value, hdr["BMAJ"])
    npt.assert_equal(bima_beam_file.minor.value, hdr["BMIN"])
    npt.assert_equal(bima_beam_file.pa.value, hdr["BPA"])

    bima_beam_hdr = Beam.from_fits_header(hdr)

    npt.assert_equal(bima_beam_hdr.major.value, hdr["BMAJ"])
    npt.assert_equal(bima_beam_hdr.minor.value, hdr["BMIN"])
    npt.assert_equal(bima_beam_hdr.pa.value, hdr["BPA"])


def test_from_aips_test():
    aips_fname = data_path("ngc0925_na.fits.gz")
    aips_hdr = fits.getheader(aips_fname)
    aips_beam_hdr = Beam.from_fits_header(aips_hdr)
    npt.assert_almost_equal(aips_beam_hdr.sr.value, 9.029858054819811e-10)
    aips_beam_file = Beam.from_fits_header(aips_fname)
    npt.assert_almost_equal(aips_beam_file.sr.value, 9.029858054819811e-10)

def test_fits_from_casa():
    casa_fname = data_path("m83.moment0.fits.gz")
    casa_hdr = fits.getheader(casa_fname)
    casa_beam_hdr = Beam.from_fits_header(casa_hdr)
    npt.assert_almost_equal(casa_beam_hdr.sr.value, 2.98323984597532e-11)
    casa_beam_file = Beam.from_fits_header(casa_fname)
    npt.assert_almost_equal(casa_beam_file.sr.value, 2.98323984597532e-11)

def test_manual():
    # Instantiate from command line
    man_beam_val = Beam(0.1, 0.1, 30)
    npt.assert_almost_equal(man_beam_val.value, 3.451589629868801e-06)
    man_beam_rad = Beam(0.1*u.rad, 0.1*u.rad, 30*u.deg)
    npt.assert_almost_equal(man_beam_rad.value, 0.011330900354567986)
    man_beam_deg = Beam(0.1*u.deg, 0.1*u.deg, 1.0*u.rad)
    npt.assert_almost_equal(man_beam_deg.value, 3.451589629868801e-06)

def test_bintable():

    beams = np.recarray(4, dtype=[('BMAJ', '>f4'), ('BMIN', '>f4'),
                                  ('BPA', '>f4'), ('CHAN', '>i4'),
                                  ('POL', '>i4')])
    beams['BMIN'] = [0.1,0.1001,0.09999,0.099999] # arcseconds
    beams['BMAJ'] = [0.2,0.2001,0.1999,0.19999]
    beams['BPA'] = [45.1,45.101,45.102,45.099] # degrees
    beams['CHAN'] = [0,0,0,0]
    beams['POL'] = [0,0,0,0]
    beams = fits.BinTableHDU(beams)

    beam = Beam.from_fits_bintable(beams)

    npt.assert_almost_equal(beam.minor.to(u.arcsec).value,
                            0.10002226, decimal=4)
    npt.assert_almost_equal(beam.major.to(u.arcsec).value,
                            0.19999751, decimal=4)
    npt.assert_almost_equal(beam.pa.to(u.deg).value,
                            45.10050065568665, decimal=4)


@pytest.mark.skipif("not HAS_CASA")
def test_from_casa_image():
    # Extract from tar
    import tarfile
    fname_tar = data_path("NGC0925.bima.mmom0.image.tar.gz")
    tar = tarfile.open(fname_tar)
    tar.extractall(path=data_dir)
    tar.close()
    fname = data_path("NGC0925.bima.mmom0.image")
    bima_casa_beam = Beam.from_casa_image(fname)


def test_attach_to_header():
    fname = data_path("NGC0925.bima.mmom0.fits.gz")
    hdr = fits.getheader(fname)
    hdr_copy = hdr.copy()
    del hdr_copy["BMAJ"], hdr_copy["BMIN"], hdr_copy["BPA"]

    bima_beam = Beam.from_fits_header(fname)

    new_hdr = bima_beam.attach_to_header(hdr_copy)

    npt.assert_equal(new_hdr["BMAJ"], hdr["BMAJ"])
    npt.assert_equal(new_hdr["BMIN"], hdr["BMIN"])
    npt.assert_equal(new_hdr["BPA"], hdr["BPA"])


def test_beam_projected_area():

    distance = 250 * u.pc

    major = 0.1 * u.rad
    beam = Beam(major, major, 30 * u.deg)

    beam_sr = (major**2 * 2 * np.pi / (8 * np.log(2))).to(u.sr)

    assert_quantity_allclose(beam_sr.value * distance ** 2,
                             beam.beam_projected_area(distance))


def test_jtok():

    major = 0.1 * u.rad
    beam = Beam(major, major, 30 * u.deg)

    freq = 1.42 * u.GHz

    conv_factor = u.brightness_temperature(beam.sr, freq)

    assert_quantity_allclose((1 * u.Jy).to(u.K, equivalencies=conv_factor),
                             beam.jtok(freq))


def test_jtok_equiv():

    major = 0.1 * u.rad
    beam = Beam(major, major, 30 * u.deg)

    freq = 1.42 * u.GHz

    conv_factor = u.brightness_temperature(beam.sr, freq)
    conv_beam_factor = beam.jtok_equiv(freq)

    assert_quantity_allclose((1 * u.Jy).to(u.K, equivalencies=conv_factor),
                             (1 * u.Jy).to(u.K, equivalencies=conv_beam_factor))

    assert_quantity_allclose((1 * u.K).to(u.Jy, equivalencies=conv_factor),
                             (1 * u.K).to(u.Jy, equivalencies=conv_beam_factor))


def test_convolution():

    # equations from:
    # https://github.com/pkgw/carma-miriad/blob/CVSHEAD/src/subs/gaupar.for
    # (github checkin of MIRIAD, code by Sault)

    major1 = 1 * u.deg
    minor1 = 0.5 * u.deg
    pa1 = 0.0 * u.deg
    beam1 = Beam(major1, minor1, pa1)

    major2 = 1 * u.deg
    minor2 = 0.75 * u.deg
    pa2 = 90.0 * u.deg
    beam2 = Beam(major2, minor2, pa2)

    alpha = (major1 * np.cos(pa1))**2 + (minor1 * np.sin(pa1))**2 + \
        (major2 * np.cos(pa2))**2 + (minor2 * np.sin(pa2))**2
    beta = (major1 * np.sin(pa1))**2 + (minor1 * np.cos(pa1))**2 + \
        (major2 * np.sin(pa2))**2 + (minor2 * np.cos(pa2))**2
    gamma = 2 * ((minor1**2 - major1**2) * np.sin(pa1) * np.cos(pa1) +
                 (minor2**2 - major2**2) * np.sin(pa2) * np.cos(pa2))

    s = alpha + beta
    t = np.sqrt((alpha - beta)**2 + gamma**2)

    conv_major = np.sqrt(0.5 * (s + t))
    conv_minor = np.sqrt(0.5 * (s - t))
    conv_pa = 0.5 * np.arctan2(- gamma, alpha - beta)

    conv_beam = beam1.convolve(beam2)

    assert_quantity_allclose(conv_major, conv_beam.major)
    assert_quantity_allclose(conv_minor, conv_beam.minor)
    assert_quantity_allclose(conv_pa, conv_beam.pa)


def test_deconvolution():

    # equations from:
    # https://github.com/pkgw/carma-miriad/blob/CVSHEAD/src/subs/gaupar.for
    # (github checkin of MIRIAD, code by Sault)

    major1 = 2.0 * u.deg
    minor1 = 1.0 * u.deg
    pa1 = 45.0 * u.deg
    beam1 = Beam(major1, minor1, pa1)

    major2 = 1 * u.deg
    minor2 = 0.5 * u.deg
    pa2 = 0.0 * u.deg
    beam2 = Beam(major2, minor2, pa2)

    alpha = (major1 * np.cos(pa1))**2 + (minor1 * np.sin(pa1))**2 - \
        (major2 * np.cos(pa2))**2 - (minor2 * np.sin(pa2))**2
    beta = (major1 * np.sin(pa1))**2 + (minor1 * np.cos(pa1))**2 - \
        (major2 * np.sin(pa2))**2 - (minor2 * np.cos(pa2))**2
    gamma = 2 * ((minor1**2 - major1**2) * np.sin(pa1) * np.cos(pa1) +
                 (minor2**2 - major2**2) * np.sin(pa2) * np.cos(pa2))

    s = alpha + beta
    t = np.sqrt((alpha - beta)**2 + gamma**2)

    deconv_major = np.sqrt(0.5 * (s + t))
    deconv_minor = np.sqrt(0.5 * (s - t))
    deconv_pa = 0.5 * np.arctan2(- gamma, alpha - beta)

    deconv_beam = beam1.deconvolve(beam2)

    assert_quantity_allclose(deconv_major, deconv_beam.major)
    assert_quantity_allclose(deconv_minor, deconv_beam.minor)
    assert_quantity_allclose(deconv_pa, deconv_beam.pa)


def test_conv_deconv():

    beam1 = Beam(10. * u.arcsec, 5. * u.arcsec, 30. * u.deg)
    beam2 = Beam(5. * u.arcsec, 3. * u.arcsec, 120. * u.deg)

    beam3 = beam1.convolve(beam2)

    assert beam2 == beam3.deconvolve(beam1)
    assert beam1 == beam3.deconvolve(beam2)

    assert beam1.convolve(beam2) == beam2.convolve(beam1)

    # Test multiplication and subtraction (i.e., convolution and deconvolution)
    # subtraction-as-deconvolution is deprecated. Check that one of the gives
    # the warning

    with warnings.catch_warnings(record=True) as w:
        assert beam2 == beam3 - beam1

    assert len(w) == 1
    assert w[0].category == RadioBeamDeprecationWarning
    assert str(w[0].message) == ("Subtraction-as-deconvolution is deprecated. "
                                 "Use division instead.")

    # Dividing should give the same thing
    assert beam2 == beam3 / beam1
    assert beam1 == beam3 / beam2

    assert beam3 == beam1 * beam2


@pytest.mark.parametrize(('major', 'minor', 'pa', 'return_pointlike'),
                         [[maj, min, pa, ret] for maj, min, pa, ret in
                         product([10], np.arange(1, 11),
                                 np.linspace(0, 180, 10), [True, False])])
def test_deconv_pointlike(major, minor, pa, return_pointlike):

    beam1 = Beam(major * u.arcsec, major * u.arcsec, pa * u.deg)

    if return_pointlike:
        point_beam = Beam(0 * u.deg, 0 * u.deg, 0 * u.deg)
        point_beam == beam1.deconvolve(beam1, failure_returns_pointlike=True)
    else:
        try:
            beam1.deconvolve(beam1, failure_returns_pointlike=False)
        except ValueError:
            pass


def test_isfinite():

    beam1 = Beam(10. * u.arcsec, 5. * u.arcsec, 30. * u.deg)

    assert beam1.isfinite

    # raises an exception because major < minor
    #beam2 = Beam(-10. * u.arcsec, 5. * u.arcsec, 30. * u.deg)

    #assert not beam2.isfinite

    beam3 = Beam(10. * u.arcsec, -5. * u.arcsec, 30. * u.deg)

    assert not beam3.isfinite


@pytest.mark.parametrize(("major", "minor", "pa"),
                         [(10, 10, 60),
                          (10, 10, -120),
                          (10, 10, -300),
                          (10, 10, 240),
                          (10, 10, 59),
                          (10, 10, -121)])
def test_beam_equal(major, minor, pa):

    beam1 = Beam(10 * u.deg, 10 * u.deg, 60 * u.deg)

    beam2 = Beam(major * u.deg, minor * u.deg, pa * u.deg)

    assert beam1 == beam2
    assert not beam1 != beam2


@pytest.mark.parametrize(("major", "minor", "pa"),
                         [(10, 8, 60),
                          (10, 8, -120),
                          (10, 8, 240)])
def test_beam_equal_noncirc(major, minor, pa):
    '''
    Beams with PA +/- 180 deg are equal
    '''

    beam1 = Beam(10 * u.deg, 8 * u.deg, 60 * u.deg)

    beam2 = Beam(major * u.deg, minor * u.deg, pa * u.deg)

    assert beam1 == beam2
    assert not beam1 != beam2


@pytest.mark.parametrize(("major", "minor", "pa"),
                         [(10, 8, 60),
                          (12, 10, 60),
                          (12, 10, 59)])
def test_beam_not_equal(major, minor, pa):

    beam1 = Beam(10 * u.deg, 10 * u.deg, 60 * u.deg)

    beam2 = Beam(major * u.deg, minor * u.deg, pa * u.deg)

    assert beam1 != beam2

def test_from_aips_issue43():
    """ regression test for issue 43 """
    aips_fname = data_path("header_aips.hdr")
    aips_hdr = fits.Header.fromtextfile(aips_fname)
    aips_beam_hdr = Beam.from_fits_header(aips_hdr)
    npt.assert_almost_equal(aips_beam_hdr.pa.value, -15.06)

def test_small_beam_convolution():
    # regression test for #68
    beam1 = Beam((0.1*u.arcsec).to(u.deg), (0.00001*u.arcsec).to(u.deg), 30*u.deg)
    beam2 = Beam((0.3*u.arcsec).to(u.deg), (0.00001*u.arcsec).to(u.deg), 120*u.deg)
    conv = beam1.convolve(beam2)

    np.testing.assert_almost_equal(conv.pa.to(u.deg).value, -60)

def test_major_minor_swap():

    with pytest.raises(ValueError) as exc:
        beam1 = Beam(minor=10. * u.arcsec, major=5. * u.arcsec,
                     pa=30. * u.deg)

    assert "Minor axis greater than major axis." in exc.value.args[0]
