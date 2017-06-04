import numpy as np

from astropy import units as u
from astropy.io import fits

from ..multiple_beams import Beams
from ..beam import Beam

from .test_beam import data_path


def beams_for_tests():

    majors = [1, 1, 1, 2, 3, 4] * u.arcsec

    return Beams(major=majors), majors


def test_beams_from_fits_bintable():

    fname = data_path("m33_beams_bintable.fits.gz")

    bintable = fits.open(fname)[1]

    beams = Beams.from_fits_bintable(bintable)

    assert (beams.major.value == bintable.data['BMAJ']).all()
    assert (beams.minor.value == bintable.data['BMIN']).all()
    assert (beams.pa.value == bintable.data['BPA']).all()


def test_indexing():

    beams, majors = beams_for_tests()

    assert hasattr(beams[slice(0, 3)], 'major')
    assert np.all(beams[slice(0, 3)].major.value == majors[:3].value)
    assert np.all(beams[slice(0, 3)].minor.value == majors[:3].value)

    assert hasattr(beams[:3], 'major')
    assert np.all(beams[:3].major.value == majors[:3].value)
    assert np.all(beams[:3].minor.value == majors[:3].value)

    assert hasattr(beams[3], 'major')
    assert beams[3].major.value == 2
    assert beams[3].minor.value == 2
    assert isinstance(beams[4], Beam)

    mask = np.array([True, False, True, False, True, True], dtype='bool')
    assert hasattr(beams[mask], 'major')
    assert np.all(beams[mask].major.value == majors[mask].value)


def test_average_beams():

    beams, majors = beams_for_tests()

    assert np.all(beams.average_beam().major.value == majors.mean().value)

    mask = np.array([True, False, True, False, True, True], dtype='bool')

    assert np.all(beams[mask].average_beam().major.value == majors[mask].mean().value)


def test_largest_beams():

    beams, majors = beams_for_tests()

    assert np.all(beams.largest_beam().major.value == majors.max().value)
    assert np.all(beams.largest_beam().minor.value == majors.max().value)

    # Slice the object
    mask = np.array([True, False, True, False, True, True], dtype='bool')

    assert np.all(beams[mask].largest_beam().major.value == majors[mask].max().value)
    assert np.all(beams[mask].largest_beam().minor.value == majors[mask].max().value)

    # Apply a mask only for the operation
    assert np.all(beams.largest_beam(mask).major.value == majors[mask].max().value)
    assert np.all(beams.largest_beam(mask).minor.value == majors[mask].max().value)


def test_smallest_beams():

    beams, majors = beams_for_tests()

    assert np.all(beams.smallest_beam().major.value == majors.min().value)
    assert np.all(beams.smallest_beam().minor.value == majors.min().value)

    # Slice the object
    mask = np.array([True, False, True, False, True, True], dtype='bool')

    assert np.all(beams[mask].smallest_beam().major.value == majors[mask].min().value)
    assert np.all(beams[mask].smallest_beam().minor.value == majors[mask].min().value)

    # Apply a mask only for the operation
    assert np.all(beams.smallest_beam(mask).major.value == majors[mask].min().value)
    assert np.all(beams.smallest_beam(mask).minor.value == majors[mask].min().value)


def test_extrema_beams():

    beams, majors = beams_for_tests()

    extrema = beams.extrema_beams()
    assert np.all(extrema[0].major.value == majors.min().value)
    assert np.all(extrema[0].minor.value == majors.min().value)

    assert np.all(extrema[1].major.value == majors.max().value)
    assert np.all(extrema[1].minor.value == majors.max().value)

    # Slice the object
    mask = np.array([True, False, True, False, True, True], dtype='bool')
    extrema = beams[mask].extrema_beams()
    assert np.all(extrema[0].major.value == majors[mask].min().value)
    assert np.all(extrema[0].minor.value == majors[mask].min().value)

    assert np.all(extrema[1].major.value == majors[mask].max().value)
    assert np.all(extrema[1].minor.value == majors[mask].max().value)

    # Apply a mask only for the operation
    extrema = beams.extrema_beams(mask)
    assert np.all(extrema[0].major.value == majors[mask].min().value)
    assert np.all(extrema[0].minor.value == majors[mask].min().value)

    assert np.all(extrema[1].major.value == majors[mask].max().value)
    assert np.all(extrema[1].minor.value == majors[mask].max().value)


def test_beams_with_invalid():

    majors = [1, 1, 1, 2, np.NaN, 4] * u.arcsec

    beams = Beams(major=majors)

    # Average
    assert np.all(beams.average_beam().major.value == np.nanmean(majors).value)
    # Largest
    assert np.all(beams.largest_beam().major.value == np.nanmax(majors).value)
    # Smallest
    assert np.all(beams.smallest_beam().major.value == np.nanmin(majors).value)
    # Extrema
    extrema = beams.extrema_beams()
    assert np.all(extrema[0].major.value == np.nanmin(majors).value)
    assert np.all(extrema[1].major.value == np.nanmax(majors).value)

    # Additional masking
    mask = np.array([True, False, True, False, True, True], dtype='bool')

    # Average
    assert np.all(beams[mask].average_beam().major.value == np.nanmean(majors[mask]).value)
    # Largest
    assert np.all(beams[mask].largest_beam().major.value == np.nanmax(majors[mask]).value)
    # Smallest
    assert np.all(beams[mask].smallest_beam().major.value == np.nanmin(majors[mask]).value)
    # Extrema
    extrema = beams[mask].extrema_beams()
    assert np.all(extrema[0].major.value == np.nanmin(majors[mask]).value)
    assert np.all(extrema[1].major.value == np.nanmax(majors[mask]).value)
