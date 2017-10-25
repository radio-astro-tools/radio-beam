import numpy as np
import numpy.testing as npt

from astropy import units as u
from astropy.io import fits

import pytest

from ..multiple_beams import Beams
from ..beam import Beam

from .test_beam import data_path


def symm_beams_for_tests():

    majors = [1, 1, 1, 2, 3, 4] * u.arcsec
    minors = majors
    pas = [0] * 6 * u.deg

    return Beams(major=majors, minor=minors, pa=pas), majors, minors, pas


def asymm_beams_for_tests():

    majors = [1, 1, 1, 2, 3, 4] * u.arcsec
    minors = majors / 2.
    pas = [-36, 20, 80, 41, -82, 11] * u.deg

    return Beams(major=majors, minor=minors, pa=pas), majors, minors, pas


def load_commonbeam_comparisons():

    common_beams = np.loadtxt(data_path("commonbeam_CASA_comparison.csv"),
                              delimiter=',')

    return common_beams


def test_beams_from_fits_bintable():

    fname = data_path("m33_beams_bintable.fits.gz")

    bintable = fits.open(fname)[1]

    beams = Beams.from_fits_bintable(bintable)

    assert (beams.major.value == bintable.data['BMAJ']).all()
    assert (beams.minor.value == bintable.data['BMIN']).all()
    assert (beams.pa.value == bintable.data['BPA']).all()


def test_indexing():

    beams, majors = symm_beams_for_tests()[:2]

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

    beams, majors = symm_beams_for_tests()[:2]

    assert np.all(beams.average_beam().major.value == majors.mean().value)

    mask = np.array([True, False, True, False, True, True], dtype='bool')

    assert np.all(beams[mask].average_beam().major.value == majors[mask].mean().value)


@pytest.mark.parametrize(("beams", "majors", "minors", "pas"),
                         [symm_beams_for_tests(), asymm_beams_for_tests()])
def test_largest_beams(beams, majors, minors, pas):

    assert beams.largest_beam().major.value == majors.max().value
    assert beams.largest_beam().minor.value == minors.max().value

    # Slice the object
    mask = np.array([True, False, True, False, True, True], dtype='bool')

    assert beams[mask].largest_beam().major.value == majors[mask].max().value
    assert beams[mask].largest_beam().minor.value == minors[mask].max().value

    # Apply a mask only for the operation
    assert beams.largest_beam(mask).major.value == majors[mask].max().value
    assert beams.largest_beam(mask).minor.value == minors[mask].max().value


@pytest.mark.parametrize(("beams", "majors", "minors", "pas"),
                         [symm_beams_for_tests(), asymm_beams_for_tests()])
def test_smallest_beams(beams, majors, minors, pas):

    assert beams.smallest_beam().major.value == majors.min().value
    assert beams.smallest_beam().minor.value == minors.min().value

    # Slice the object
    mask = np.array([True, False, True, False, True, True], dtype='bool')

    assert beams[mask].smallest_beam().major.value == majors[mask].min().value
    assert beams[mask].smallest_beam().minor.value == minors[mask].min().value

    # Apply a mask only for the operation
    assert beams.smallest_beam(mask).major.value == majors[mask].min().value
    assert beams.smallest_beam(mask).minor.value == minors[mask].min().value


@pytest.mark.parametrize(("beams", "majors", "minors", "pas"),
                         [symm_beams_for_tests(), asymm_beams_for_tests()])
def test_extrema_beams(beams, majors, minors, pas):

    extrema = beams.extrema_beams()
    assert extrema[0].major.value == majors.min().value
    assert extrema[0].minor.value == minors.min().value

    assert extrema[1].major.value == majors.max().value
    assert extrema[1].minor.value == minors.max().value

    # Slice the object
    mask = np.array([True, False, True, False, True, True], dtype='bool')
    extrema = beams[mask].extrema_beams()
    assert extrema[0].major.value == majors[mask].min().value
    assert extrema[0].minor.value == minors[mask].min().value

    assert extrema[1].major.value == majors[mask].max().value
    assert extrema[1].minor.value == minors[mask].max().value

    # Apply a mask only for the operation
    extrema = beams.extrema_beams(mask)
    assert extrema[0].major.value == majors[mask].min().value
    assert extrema[0].minor.value == minors[mask].min().value

    assert extrema[1].major.value == majors[mask].max().value
    assert extrema[1].minor.value == minors[mask].max().value


@pytest.mark.parametrize("majors", [[1, 1, 1, 2, np.NaN, 4],
                                    [0, 1, 1, 2, 3, 4]])
def test_beams_with_invalid(majors):

    majors = np.asarray(majors) * u.arcsec

    beams = Beams(major=majors)

    # Average
    assert beams.average_beam().major.value == np.nanmean(majors[np.nonzero(majors)]).value
    # Largest
    assert beams.largest_beam().major.value == np.nanmax(majors).value
    # Smallest
    assert beams.smallest_beam().major.value == np.nanmin(majors[np.nonzero(majors)]).value
    # Extrema
    extrema = beams.extrema_beams()
    assert extrema[0].major.value == np.nanmin(majors[np.nonzero(majors)]).value
    assert extrema[1].major.value == np.nanmax(majors).value

    # Additional masking
    mask = np.array([True, False, True, False, True, True], dtype='bool')

    if np.isnan(majors).any():
        bad_mask = np.isfinite(majors)
    else:
        bad_mask = majors.value != 0

    combined_mask = np.logical_and(mask, bad_mask)

    # Average
    assert beams[mask].average_beam().major.value == np.nanmean(majors[combined_mask]).value
    # Largest
    assert beams[mask].largest_beam().major.value == np.nanmax(majors[combined_mask]).value
    # Smallest
    assert beams[mask].smallest_beam().major.value == np.nanmin(majors[combined_mask]).value
    # Extrema
    extrema = beams[mask].extrema_beams()
    assert extrema[0].major.value == np.nanmin(majors[combined_mask]).value
    assert extrema[1].major.value == np.nanmax(majors[combined_mask]).value


def test_beams_iter():

    beams, majors = symm_beams_for_tests()[:2]

    # Ensure iterating through yields the same as slicing
    for i, beam in enumerate(beams):
        assert beam == beams[i]


# @pytest.mark.parametrize('comp_vals',
#                          [vals for vals in load_commonbeam_comparisons()])
# def test_commonbeam_casa_compare(comp_vals):

#     # These are the common beam parameters assuming 2 beams:
#     # 1) 3"x3"
#     # 2) 4"x2.5", varying the PA in 1 deg increments from 0 to 179 deg
#     # See data/generate_commonbeam_table.py

#     pa, com_major, com_minor, com_pa = comp_vals

#     beams = Beams(major=[3, 4] * u.arcsec, minor=[3, 2.5] * u.arcsec,
#                   pa=[0, pa] * u.deg)

#     common_beam = beams.common_beam()

#     # npt.assert_almost_equal(common_beam.major.value, com_major)
#     # npt.assert_almost_equal(common_beam.minor.value, com_minor)
#     npt.assert_almost_equal(common_beam.pa.to(u.deg).value, com_pa)
#     npt.assert_almost_equal(common_beam.pa.to(u.deg).value, pa)


# def test_commonbeam_notlargest():

#     beams = Beams(major=[3, 4] * u.arcsec, minor=[3, 2.5] * u.arcsec)

#     target_beam = Beam(major=4 * u.arcsec,
#                        minor=3 * u.arcsec)

#     assert beams.common_beam() == target_beam


# def test_commonbeam_largest():
#     '''
#     commonbeam is the largest in this set.
#     '''

#     beams, majors = symm_beams_for_tests()[:2]

#     assert beams.common_beam() == beams.largest_beam()

#     # With masking
#     mask = np.array([True, False, True, True, True, False], dtype='bool')

#     assert beams[mask].common_beam() == beams[mask].largest_beam()

#     assert beams.common_beam(mask) == beams.largest_beam(mask)


# # Implements the same test suite used in CASA

# def casa_commonbeam_suite():

#     cases = []

#     # https://open-bitbucket.nrao.edu/projects/CASA/repos/casa/browse/code/imageanalysis/ImageAnalysis/test/tCasaImageBeamSet.cc

#     # 1
#     cases.append((Beams(major=[4] * 2 * u.arcsec, minor=[2] * 2 * u.arcsec,
#                         pa=[0, 60] * u.deg),
#                   Beam(major=4.4856 * u.arcsec, minor=3.2916 * u.arcsec,
#                        pa=30.0 * u.deg)))
#     # 2
#     cases.append((Beams(major=[4] * 2 * u.arcsec, minor=[2] * 2 * u.arcsec,
#                         pa=[20, 80] * u.deg),
#                   Beam(major=4.4856 * u.arcsec, minor=3.2916 * u.arcsec,
#                        pa=50.0 * u.deg)))
#     # 3
#     cases.append((Beams(major=[4] * 2 * u.arcsec, minor=[2] * 2 * u.arcsec,
#                         pa=[1, 89] * u.deg),
#                   Beam(major=4.042 * u.arcsec, minor=3.958 * u.arcsec,
#                        pa=45.0 * u.deg)))
#     # 4
#     cases.append((Beams(major=[4] * 2 * u.arcsec, minor=[2] * 2 * u.arcsec,
#                         pa=[0, 90] * u.deg),
#                   Beam(major=4 * u.arcsec, minor=4 * u.arcsec,
#                        pa=0.0 * u.deg)))
#     # 5
#     cases.append((Beams(major=[4, 1.5] * u.arcsec, minor=[2, 1] * u.arcsec,
#                         pa=[0, 90] * u.deg),
#                   Beam(major=4 * u.arcsec, minor=2 * u.arcsec,
#                        pa=0.0 * u.deg)))
#     # 6
#     cases.append((Beams(major=[8, 4] * u.arcsec, minor=[1, 1] * u.arcsec,
#                         pa=[0, 20] * u.deg),
#                   Beam(major=8.377 * u.arcsec, minor=1.628 * u.arcsec,
#                        pa=2.7679 * u.deg)))
#     # 7
#     cases.append((Beams(major=[4, 8] * u.arcsec, minor=[1, 1] * u.arcsec,
#                         pa=[0, 20] * u.deg),
#                   Beam(major=8.369 * u.arcsec, minor=1.626 * u.arcsec,
#                        pa=17.232 * u.deg)))
#     # 8
#     cases.append((Beams(major=[4] * 4 * u.arcsec, minor=[2] * 4 * u.arcsec,
#                         pa=[0, 60, 20, 40] * u.deg),
#                   Beam(major=4.485 * u.arcsec, minor=3.291 * u.arcsec,
#                        pa=30 * u.deg)))
#     # 9
#     # THE COMPARISON BEAM IS NOT CORRECT! There is an ordering issue to
#     # work out! This will fail until it is corrected.
#     cases.append((Beams(major=[4] * 4 * u.arcsec, minor=[2] * 4 * u.arcsec,
#                         pa=[0, 20, 40, 60] * u.deg),
#                   Beam(major=4.485 * u.arcsec, minor=3.291 * u.arcsec,
#                        pa=30 * u.deg)))
#     # 10
#     cases.append((Beams(major=[4, 1] * u.arcsec, minor=[2, 1] * u.arcsec,
#                         pa=[0, 0] * u.deg),
#                   Beam(major=4 * u.arcsec, minor=2 * u.arcsec,
#                        pa=0.0 * u.deg)))

#     return cases


# @pytest.mark.parametrize(("beams", "target_beam"), casa_commonbeam_suite())
# def test_commonbeam_angleoffset(beams, target_beam):

#     # https://open-bitbucket.nrao.edu/projects/CASA/repos/casa/browse/code/imageanalysis/ImageAnalysis/test/tCasaImageBeamSet.cc#447

#     common_beam = beams.common_beam()

#     npt.assert_almost_equal(common_beam.major.value, target_beam.major.value,
#                             decimal=3)
#     npt.assert_almost_equal(common_beam.minor.value, target_beam.minor.value,
#                             decimal=3)
#     npt.assert_almost_equal(common_beam.pa.to(u.deg).value,
#                             target_beam.pa.value, decimal=3)