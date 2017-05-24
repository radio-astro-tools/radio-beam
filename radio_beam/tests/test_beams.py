import numpy as np

from astropy import units as u
from astropy.io import fits

from ..multiple_beams import Beams
from ..beam import Beam

from .test_beam import data_path


def test_beams_from_fits_bintable():

    fname = data_path("m33_beams_bintable.fits.gz")

    bintable = fits.open(fname)[1]

    beams = Beams.from_fits_bintable(bintable)

    assert (beams.major.value == bintable.data['BMAJ']).all()
    assert (beams.minor.value == bintable.data['BMIN']).all()
    assert (beams.pa.value == bintable.data['BPA']).all()

def test_indexing():

    beams = Beams(major=[1,1,1,2,3,4]*u.arcsec)

    assert hasattr(beams[:3], 'major')
    assert np.all(beams[:3].major.value == [1,1,1])

    assert hasattr(beams[3], 'major')
    assert beams[3].major.value == 2
    assert isinstance(beams[4], Beam)

    assert hasattr(beams[np.array([True,False,True,False,True,True], dtype='bool')], 'major')
    assert np.all(beams[np.array([True,False,True,False,True,True], dtype='bool')].major.value == [1,1,3,4])
