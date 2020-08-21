import numpy as np
import numpy.testing as npt

from astropy import units as u
from astropy.io import fits

import warnings
import pytest

from ..multiple_beams import Beams
from ..beam import Beam
from ..commonbeam import common_2beams, common_manybeams_mve
from ..utils import InvalidBeamOperationError, BeamError

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


def test_beam_areas():

    beams, majors = symm_beams_for_tests()[:2]

    areas = 2 * np.pi / (8 * np.log(2)) * (majors.to(u.rad)**2).to(u.sr)

    assert np.all(areas.value == beams.sr.value)
    assert np.all(beams.value == beams.sr.value)


def test_beams_from_fits_bintable():

    fname = data_path("m33_beams_bintable.fits.gz")

    bintable = fits.open(fname)[1]

    beams = Beams.from_fits_bintable(bintable)

    assert (beams.major.value == bintable.data['BMAJ']).all()
    assert (beams.minor.value == bintable.data['BMIN']).all()
    assert (beams.pa.value == bintable.data['BPA']).all()


def test_beams_from_list_of_beam():

    beams, majors = symm_beams_for_tests()[:2]

    new_beams = Beams(beams=[beam for beam in beams])

    assert beams == new_beams

    abeams = asymm_beams_for_tests()[0]
    new_abeams = Beams(beams=[beam for beam in abeams])
    assert abeams == new_abeams


def test_beams_equality_beams():

    beams, majors = symm_beams_for_tests()[:2]

    assert beams == beams

    assert not beams != beams

    abeams, amajors = asymm_beams_for_tests()[:2]

    assert not (beams == abeams)

    assert beams != abeams


def test_beams_equality_beam():

    # Test whether all are equal to a single beam
    beams = Beams([1.] * 5 * u.arcsec)

    beam = Beam(1 * u.arcsec)

    assert np.all(beams == beam)

    assert not np.any(beams != beam)


@pytest.mark.xfail(raises=InvalidBeamOperationError, strict=True)
def test_beams_equality_fail():

    # Test whether all are equal to a single beam
    beams = Beams([1.] * 5 * u.arcsec)

    beams == 2


@pytest.mark.xfail(raises=InvalidBeamOperationError, strict=True)
def test_beams_notequality_fail():

    # Test whether all are equal to a single beam
    beams = Beams([1.] * 5 * u.arcsec)

    beams != 2


@pytest.mark.xfail(raises=InvalidBeamOperationError, strict=True)
def test_beams_equality_fail_shape():

    # Test whether all are equal to a single beam
    beams = Beams([1.] * 5 * u.arcsec)

    assert np.all(beams == beams[1:])

@pytest.mark.xfail(raises=InvalidBeamOperationError, strict=True)
def test_beams_add_fail():

    # Test whether all are equal to a single beam
    beams = Beams([1.] * 5 * u.arcsec)

    beams + 2


@pytest.mark.xfail(raises=InvalidBeamOperationError, strict=True)
def test_beams_sub_fail():

    # Test whether all are equal to a single beam
    beams = Beams([1.] * 5 * u.arcsec)

    beams - 2


@pytest.mark.xfail(raises=InvalidBeamOperationError, strict=True)
def test_beams_mult_fail():

    # Test whether all are equal to a single beam
    beams = Beams([1.] * 5 * u.arcsec)

    beams * 2


@pytest.mark.xfail(raises=InvalidBeamOperationError, strict=True)
def test_beams_div_fail():

    # Test whether all are equal to a single beam
    beams = Beams([1.] * 5 * u.arcsec)

    beams / 2


def test_beams_mult_convolution():

    beams, majors = asymm_beams_for_tests()[:2]

    beam = Beam(1 * u.arcsec)

    conv_beams = beams * beam

    individ_conv_beams = [beam_i.convolve(beam) for beam_i in beams]
    new_beams = Beams(beams=individ_conv_beams)

    assert conv_beams == new_beams


def test_beams_div_deconvolution():

    beams, majors = asymm_beams_for_tests()[:2]

    beam = Beam(0.25 * u.arcsec)

    deconv_beams = beams / beam

    individ_deconv_beams = [beam_i.deconvolve(beam) for beam_i in beams]
    new_beams = Beams(beams=individ_deconv_beams)

    assert deconv_beams == new_beams


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

    # Also test int64
    chan = np.int64(3)
    assert hasattr(beams[chan], 'major')
    assert beams[chan].major.value == 2
    assert beams[chan].minor.value == 2
    assert isinstance(beams[chan], Beam)

    mask = np.array([True, False, True, False, True, True], dtype='bool')
    assert hasattr(beams[mask], 'major')
    assert np.all(beams[mask].major.value == majors[mask].value)


def test_average_beams():

    beams, majors = symm_beams_for_tests()[:2]

    assert np.all(beams.average_beam().major.value == majors.mean().value)

    mask = np.array([True, False, True, False, True, True], dtype='bool')

    assert np.all(beams[mask].average_beam().major.value ==
                  majors[mask].mean().value)


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
    assert beams.average_beam().major.value == np.nanmean(
        majors[np.nonzero(majors)]).value
    # Largest
    assert beams.largest_beam().major.value == np.nanmax(majors).value
    # Smallest
    assert beams.smallest_beam().major.value == np.nanmin(
        majors[np.nonzero(majors)]).value
    # Extrema
    extrema = beams.extrema_beams()
    assert extrema[0].major.value == np.nanmin(
        majors[np.nonzero(majors)]).value
    assert extrema[1].major.value == np.nanmax(majors).value

    # Additional masking
    mask = np.array([True, False, True, False, True, True], dtype='bool')

    if np.isnan(majors).any():
        bad_mask = np.isfinite(majors)
    else:
        bad_mask = majors.value != 0

    combined_mask = np.logical_and(mask, bad_mask)

    # Average
    assert beams[mask].average_beam().major.value == np.nanmean(
        majors[combined_mask]).value
    # Largest
    assert beams[mask].largest_beam().major.value == np.nanmax(
        majors[combined_mask]).value
    # Smallest
    assert beams[mask].smallest_beam().major.value == np.nanmin(
        majors[combined_mask]).value
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


def test_common_beam_smallcircular():
    '''
    Simple solution if the smallest beam is circular with a radius larger:
    Major axis is from the largest beam, minor axis is the radius of the
    smaller, and the PA is from the largest beam.
    '''

    for pa in [0., 18., 68., 122.]:
        beams = Beams(major=[3, 4] * u.arcsec,
                      minor=[3, 2.5] * u.arcsec,
                      pa=[0, pa] * u.deg)

        targ_beam = Beam(4 * u.arcsec, 3 * u.arcsec, pa * u.deg)

        assert targ_beam == beams.common_beam()


def test_commonbeam_notlargest():

    beams = Beams(major=[3, 4] * u.arcsec, minor=[3, 2.5] * u.arcsec)

    target_beam = Beam(major=4 * u.arcsec,
                       minor=3 * u.arcsec)

    assert beams.common_beam() == target_beam


def test_commonbeam_largest():
    '''
    commonbeam is the largest in this set.
    '''

    beams, majors = symm_beams_for_tests()[:2]

    assert beams.common_beam() == beams.largest_beam()

    # With masking
    mask = np.array([True, False, True, True, True, False], dtype='bool')

    assert beams[mask].common_beam() == beams[mask].largest_beam()

    assert beams.common_beam(mask) == beams.largest_beam(mask)


# Implements the same test suite used in CASA

def casa_commonbeam_suite():

    cases = []

    # https://open-bitbucket.nrao.edu/projects/CASA/repos/casa/browse/code/imageanalysis/ImageAnalysis/test/tCasaImageBeamSet.cc

    # In some cases, I find smaller common beams than are listed in the CASA
    # tests. The values for the CASA tests are commented out.

    # 1
    cases.append((Beams(major=[4] * 2 * u.arcsec, minor=[2] * 2 * u.arcsec,
                        pa=[0, 60] * u.deg),
                  Beam(major=4.4812 * u.arcsec, minor=3.2883 * u.arcsec,
                       pa=30.0 * u.deg)))
                  # Beam(major=4.4856 * u.arcsec, minor=3.2916 * u.arcsec,
                  #      pa=30.0 * u.deg)))
    # 2
    cases.append((Beams(major=[4] * 2 * u.arcsec, minor=[2] * 2 * u.arcsec,
                        pa=[20, 80] * u.deg),
                  Beam(major=4.4812 * u.arcsec, minor=3.2883 * u.arcsec,
                       pa=50.0 * u.deg)))
                  # Beam(major=4.4856 * u.arcsec, minor=3.2916 * u.arcsec,
                  #      pa=50.0 * u.deg)))
    # 3
    cases.append((Beams(major=[4] * 2 * u.arcsec, minor=[2] * 2 * u.arcsec,
                        pa=[1, 89] * u.deg),
                  Beam(major=4.042 * u.arcsec, minor=3.958 * u.arcsec,
                       pa=45.0 * u.deg)))
    # 4
    cases.append((Beams(major=[4] * 2 * u.arcsec, minor=[2] * 2 * u.arcsec,
                        pa=[0, 90] * u.deg),
                  Beam(major=4 * u.arcsec, minor=4 * u.arcsec,
                       pa=0.0 * u.deg)))
    # 5
    cases.append((Beams(major=[4, 1.5] * u.arcsec, minor=[2, 1] * u.arcsec,
                        pa=[0, 90] * u.deg),
                  Beam(major=4 * u.arcsec, minor=2 * u.arcsec,
                       pa=0.0 * u.deg)))
    # 6
    cases.append((Beams(major=[8, 4] * u.arcsec, minor=[1, 1] * u.arcsec,
                        pa=[0, 20] * u.deg),
                  Beam(major=8.3684 * u.arcsec, minor=1.6253 * u.arcsec,
                       pa=2.7679 * u.deg)))
                  # Beam(major=8.377 * u.arcsec, minor=1.628 * u.arcsec,
                  #      pa=2.7679 * u.deg)))
    # 7
    cases.append((Beams(major=[4, 8] * u.arcsec, minor=[1, 1] * u.arcsec,
                        pa=[0, 20] * u.deg),
                  Beam(major=8.369 * u.arcsec, minor=1.626 * u.arcsec,
                       pa=17.232 * u.deg)))

    # 10
    cases.append((Beams(major=[4, 1] * u.arcsec, minor=[2, 1] * u.arcsec,
                        pa=[0, 0] * u.deg),
                  Beam(major=4 * u.arcsec, minor=2 * u.arcsec,
                       pa=0.0 * u.deg)))

    return cases


@pytest.mark.parametrize(("beams", "target_beam"), casa_commonbeam_suite())
def test_commonbeam_angleoffset(beams, target_beam):

    # https://open-bitbucket.nrao.edu/projects/CASA/repos/casa/browse/code/imageanalysis/ImageAnalysis/test/tCasaImageBeamSet.cc#447

    common_beam = beams.common_beam()

    # Order shouldn't matter
    common_beam_rev = beams[::-1].common_beam()

    assert common_beam == common_beam_rev

    npt.assert_allclose(common_beam.major.value, target_beam.major.value,
                        rtol=1e-3)
    npt.assert_allclose(common_beam.minor.value, target_beam.minor.value,
                        rtol=1e-3)
    npt.assert_allclose(common_beam.pa.to(u.deg).value,
                        target_beam.pa.value, rtol=1e-3)


def casa_commonbeam_suite_multiple():

    cases = []

    # 8
    cases.append((Beams(major=[4] * 4 * u.arcsec, minor=[2] * 4 * u.arcsec,
                        pa=[0, 60, 20, 40] * u.deg),
                  Beam(major=4.48904471492 * u.arcsec,
                       minor=3.28268221138 * u.arcsec,
                       pa=30.0001561178 * u.deg)))

    # This is the beam size in the CASA tests. The MVE method finds a slightly
    # smaller beam area, so that's what is tested against above and below.
                  # Beam(major=4.485 * u.arcsec, minor=3.291 * u.arcsec,
                  #      pa=30 * u.deg)))
    # 9
    cases.append((Beams(major=[4] * 4 * u.arcsec, minor=[2] * 4 * u.arcsec,
                        pa=[0, 20, 40, 60] * u.deg),
                  Beam(major=4.48904471492 * u.arcsec,
                       minor=3.28268221138 * u.arcsec,
                       pa=30.0001561178 * u.deg)))

    return cases


@pytest.mark.parametrize(("beams", "target_beam"),
                         casa_commonbeam_suite_multiple())
def test_commonbeam_multiple(beams, target_beam):

    # https://open-bitbucket.nrao.edu/projects/CASA/repos/casa/browse/code/imageanalysis/ImageAnalysis/test/tCasaImageBeamSet.cc#447

    common_beam = beams.common_beam(epsilon=1e-4)

    # The above should be using the MVE method
    common_beam_check = common_manybeams_mve(beams, epsilon=1e-4)

    assert common_beam == common_beam_check

    npt.assert_almost_equal(common_beam.major.to(u.arcsec).value,
                            target_beam.major.value,
                            decimal=6)
    npt.assert_almost_equal(common_beam.minor.to(u.arcsec).value,
                            target_beam.minor.value,
                            decimal=6)
    npt.assert_allclose(common_beam.pa.to(u.deg).value,
                        target_beam.pa.value, rtol=1e-3)


@pytest.mark.parametrize(("beams", "target_beam"), casa_commonbeam_suite())
def test_commonbeam_methods(beams, target_beam):

    epsilon = 5e-4
    tolerance = 1e-4

    two_beam_method = common_2beams(beams)
    many_beam_method = common_manybeams_mve(beams, epsilon=epsilon,
                                            tolerance=tolerance)

    # Good to ~5x the given epsilon
    npt.assert_allclose(two_beam_method.major.to(u.arcsec).value,
                        many_beam_method.major.to(u.arcsec).value,
                        rtol=3e-3)
    npt.assert_allclose(two_beam_method.minor.to(u.arcsec).value,
                        many_beam_method.minor.to(u.arcsec).value,
                        rtol=3e-3)

    # Only test if the beam is
    circ_check = not two_beam_method.iscircular(rtol=3e-3) or \
        not many_beam_method.iscircular(rtol=3e-3)
    if circ_check:
        # The pa can be sensitive to small changes so give it a larger
        # acceptable tolerance range.
        npt.assert_allclose(two_beam_method.pa.to(u.deg).value,
                            many_beam_method.pa.to(u.deg).value,
                            rtol=5e-3)


def test_catch_common_beam_opt():
    '''
    The optimization method is close to working, but requires more testing.
    Ensure it cannot be used.
    '''

    beams = Beams(major=[4] * 4 * u.arcsec, minor=[2] * 4 * u.arcsec,
                  pa=[0, 20, 40, 60] * u.deg)

    with pytest.raises(NotImplementedError):
        beams.common_beam(method='opt')


def test_major_minor_swap():

    with pytest.raises(ValueError) as exc:
        beams = Beams(minor=[10.,5.] * u.arcsec,
                      major=[5., 5.] * u.arcsec,
                      pa=[30., 60.] * u.deg)

    assert "Minor axis greater than major axis." in exc.value.args[0]


def test_common_beam_mve_auto_increase_epsilon():
    '''
    Here's a case where the default MVE parameters fail.
    By slowly increasing the epsilon* value, we get a common
    beam the can be deconvolved correctly over the set.

    * epsilon is the small factor added to the ellipse perimeter
    radius: radius * (1 + epsilon). The solution is then marginally
    larger than the true optimal solution, but close enough for
    effectively all use cases.
    '''

    major = [8.517199, 8.513563, 8.518497, 8.518434, 8.528561, 8.528236,
             8.530046, 8.530528, 8.530696, 8.533117] * u.arcsec
    minor = [5.7432523, 5.7446027, 5.7407207, 5.740814, 5.7331843,
             5.7356524, 5.7338963, 5.733251, 5.732933, 5.73209] * u.arcsec
    pa = [-32.942623, -32.931957, -33.07815, -33.07532, -33.187653,
          -33.175243, -33.167213, -33.167244, -33.170418, -33.180233] * u.deg

    beams = Beams(major=major, minor=minor, pa=pa)

    err_str = 'Could not find common beam to deconvolve all beams.'
    with pytest.raises(BeamError, match=err_str):

        com_beam = beams.common_beam(method='pts',
                                     epsilon=5e-4,
                                     auto_increase_epsilon=False)

    # Force running into the max iteration of epsilon increases.
    err_str = 'Could not increase epsilon to find common beam.'
    with pytest.raises(BeamError, match=err_str):

        com_beam = beams.common_beam(method='pts',
                                     epsilon=5e-4,
                                     max_iter=2,
                                     max_epsilon=6e-4,
                                     auto_increase_epsilon=True)

    # Should run when epsilon is allowed to increase a bit.
    com_beam = beams.common_beam(method='pts',
                                 epsilon=5e-4,
                                 auto_increase_epsilon=True,
                                 max_epsilon=1e-3)
