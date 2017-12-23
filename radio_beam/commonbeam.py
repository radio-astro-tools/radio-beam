
import numpy as np
import astropy.units as u

try:
    from scipy import optimize as opt
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from .beam import Beam
from .utils import BeamError, transform_ellipse


def commonbeam(beams):
    '''
    Use analytic method if there are only two beams. Otherwise use constrained
    optimization to find the common beam.
    '''

    if beams.size == 1:
        return beams[0]
    if beams.size == 2:
        return common_2beams(beams)
    else:
        return common_manybeams(beams)


def common_2beams(beams):
    '''
    Find a common beam from a `Beams` object with 2 beams. This
    function is based on the CASA implementation `ia.commonbeam`. Note that
    the solution is only valid for 2 beams.

    Parameters
    ----------
    beams : `~radio_beam.Beams`
        Beams object with 2 beams.

    Returns
    -------
    common_beam : `~radio_beam.Beam`
        The smallest common beam in the set of beams.
    '''

    # This code is based on the implementation in CASA:
    # https://open-bitbucket.nrao.edu/projects/CASA/repos/casa/browse/code/imageanalysis/ImageAnalysis/CasaImageBeamSet.cc

    if beams.size != 2:
        raise BeamError("This method is only valid for two beams.")

    if (~beams.isfinite).all():
        raise BeamError("All beams in the object are invalid.")

    large_beam = beams.largest_beam()
    large_major = large_beam.major.to(u.arcsec)
    large_minor = large_beam.minor.to(u.arcsec)

    if beams.argmax() == 0:
        small_beam = beams[1]
    else:
        small_beam = beams[0]
    small_major = small_beam.major.to(u.arcsec)
    small_minor = small_beam.minor.to(u.arcsec)

    # Case where they're already equal
    if small_beam == large_beam:
        return large_beam

    deconv_beam = \
        large_beam.deconvolve(small_beam, failure_returns_pointlike=True)

    # Larger beam can be deconvolved. It is already the smallest common beam
    if deconv_beam.isfinite:
        return large_beam

    # If the smaller beam is a circle, the minor axis is the circle radius
    if small_beam.iscircular():
        common_beam = Beam(large_beam.major, small_beam.major, large_beam.pa)
        return common_beam

    # Wrap angle about 0 to pi.
    pa_diff = ((small_beam.pa.to(u.rad).value - large_beam.pa.to(u.rad).value +
                np.pi / 2. + np.pi) % np.pi - np.pi / 2.) * u.rad

    # If the difference is pi / 2, the larger major is set to the
    # new major and the minor is the other major.
    if np.isclose(np.abs(pa_diff).value, np.pi / 2.):
        larger_major = large_beam.major >= small_beam.major
        major = large_major if larger_major else small_major
        minor = small_major if larger_major else small_major
        pa = large_beam.pa if larger_major else small_beam.pa
        conv_beam = Beam(major=major, minor=minor, pa=pa)

        return conv_beam

    else:
        # Transform to coordinates where large_beam is circular

        major_comb = np.sqrt(large_major * small_major)
        p = major_comb / large_major
        q = major_comb / large_minor

        # Transform beam into the same coordinates, and rotate so its
        # major axis is along the x axis.

        trans_major_sc, trans_minor_sc, trans_pa_sc = \
            transform_ellipse(small_major, small_minor, pa_diff, p, q)

        # The transformed minor axis is major_comb, as defined in CASA
        trans_minor_sc = major_comb

        # Return beam to the original coordinates, still rotated with
        # the major along the x axis
        trans_major_unsc, trans_minor_unsc, trans_pa_unsc = \
            transform_ellipse(trans_major_sc, trans_minor_sc,
                              trans_pa_sc, 1 / p, 1 / q)

        # Lastly, rotate the PA to the enclosing ellipse
        trans_major = trans_major_unsc.to(u.arcsec)
        trans_minor = trans_minor_unsc.to(u.arcsec)
        trans_pa = trans_pa_unsc + large_beam.pa

        # The minor axis becomes an issue when checking against the smaller
        # beam from deconvolution. Adding a tiny fraction makes the deconvolved
        # beam JUST larger than zero (~1e-7).
        epsilon = 100 * np.finfo(trans_major.dtype).eps * trans_major.unit
        trans_beam = Beam(major=trans_major + epsilon,
                          minor=trans_minor + epsilon,
                          pa=trans_pa)

        # Ensure this beam can now be deconvolved
        deconv_large_beam = \
            trans_beam.deconvolve(large_beam,
                                  failure_returns_pointlike=True)
        deconv_prob_beam = \
            trans_beam.deconvolve(small_beam,
                                  failure_returns_pointlike=True)
        if not deconv_large_beam.isfinite or not deconv_prob_beam.isfinite:
            raise BeamError("Failed to find common beam that both beams can "
                            "be deconvolved by.")

        # Taken from CASA implementation, but by adding epsilon, this shouldn't
        # be needed

        # Scale the enclosing ellipse by a small factor until it does.
        # can_deconv = False
        # num = 0
        # while not can_deconv:

        #     deconv_large_beam = \
        #         trans_beam.deconvolve(large_beam,
        #                               failure_returns_pointlike=True)
        #     deconv_prob_beam = \
        #         trans_beam.deconvolve(small_beam,
        #                               failure_returns_pointlike=True)

        #     if (not deconv_large_beam.isfinite or not deconv_prob_beam.isfinite):

        #         scale_factor = 1.001
        #         trans_beam = Beam(major=trans_major * scale_factor,
        #                           minor=trans_minor * scale_factor,
        #                           pa=trans_pa)
        #     else:
        #         can_deconv = True

        #     if num == 10:
        #         break

        #     num += 1

        common_beam = trans_beam
        return common_beam


def boundingcircle(bmaj, bmin, bpa):
    thisone = np.argmax(bmaj)
    return bmaj[thisone], bmaj[thisone], bpa[thisone]


def PtoA(bmaj, bmin, bpa):
    A = np.zeros((2, 2))
    A[0, 0] = np.cos(bpa)**2 / bmaj**2 + np.sin(bpa)**2 / bmin**2
    A[1, 0] = np.cos(bpa) * np.sin(bpa) * (1 / bmaj**2 - 1 / bmin**2)
    A[0, 1] = A[1, 0]
    A[1, 1] = np.sin(bpa)**2 / bmaj**2 + np.cos(bpa)**2 / bmin**2
    return A


def BinsideA(B, A):
    try:
        np.linalg.cholesky(B - A)
        return True
    except np.linalg.LinAlgError:
        return False


def myobjective_regularized(p, bmajvec, bminvec, bpavec):
    # Force bmaj > bmin
    if p[0] < p[1]:
        return 1e30
    # We can safely assume the common major axis is at most the
    # largest major axis in the set
    if (p[0] <= bmajvec).any():
        return 1e30
    A = PtoA(*p)
    test = np.zeros_like(bmajvec)
    for idx, (bmx, bmn, bp) in enumerate(zip(bmajvec, bminvec, bpavec)):
        test[idx] = BinsideA(PtoA(bmx, bmn, bp), A)
    obj = 1 / np.linalg.det(A)
    if np.all(test):
        return obj
    else:
        return obj * 1e30


def common_manybeams(beams, p0=None, optdict=None, verbose=True):

    if not HAS_SCIPY:
        raise ImportError("common_manybeams requires scipy.optimize.")

    # First check if the largest beam is the common beam
    if fits_in_largest(beams):
        return beams.largest_beam()

    bmaj = beams.major.value
    bmin = beams.minor.value
    bpa = beams.pa.to(u.rad).value

    if p0 is None:
        p0 = boundingcircle(bmaj, bmin, bpa)
        # It seems to help to make the initial guess slightly larger
        p0 = (1.1 * p0[0], 1.1 * p0[1], p0[2])

    if optdict is None:
        optdict = {'maxiter': 5000, 'ftol': 1e-11, 'maxfev': 5000}

    result = opt.minimize(myobjective_regularized, p0, method='Nelder-Mead',
                          args=(bmaj, bmin, bpa), options=optdict,
                          tol=1e-12)

    if verbose:
        print(result.viewitems())

    if not result.success:
        raise Warning("Optimization failed")

    com_beam = Beam(result.x[0] * beams.major.unit,
                    result.x[1] * beams.major.unit,
                    result.x[2] * u.rad)

    return com_beam


def fits_in_largest(beams):
    '''
    Test if all beams can be deconvolved by the largest beam
    '''

    large_beam = beams.largest_beam()

    for beam in beams:
        if large_beam == beam:
            continue
        deconv_beam = large_beam.deconvolve(beam, failure_returns_pointlike=True)

        if not deconv_beam.isfinite:
            return False

    return True


def plotellipse(ax, bmaj, bmin, bpa, **kwargs):
    testphi = np.linspace(0, 2 * np.pi, 1001)
    x = bmaj * np.cos(testphi)
    y = bmin * np.sin(testphi)
    xr = x * np.cos(bpa) - y * np.sin(bpa)
    yr = x * np.sin(bpa) + y * np.cos(bpa)
    ax.plot(xr, yr, **kwargs)
