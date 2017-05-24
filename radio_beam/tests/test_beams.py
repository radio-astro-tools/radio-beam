
from astropy.io import fits

from ..multiple_beams import Beams

from .test_beam import data_path


def test_beams_from_fits_bintable():

    fname = data_path("m33_beams_bintable.fits.gz")

    bintable = fits.open(fname)[1]

    beams = Beams.from_fits_bintable(bintable)

    assert (beams.majors.value == bintable.data['BMAJ']).all()
    assert (beams.minors.value == bintable.data['BMIN']).all()
    assert (beams.pas.value == bintable.data['BPA']).all()
