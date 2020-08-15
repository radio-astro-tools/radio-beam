
import numpy as np
import astropy.units as u

try:
    from scipy import optimize as opt
    from scipy.spatial import ConvexHull
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from .beam import Beam
from .utils import BeamError, transform_ellipse

__all__ = ['commonbeam', 'common_2beams', 'getMinVolEllipse',
           'common_manybeams_mve']


def commonbeam(beams, method='pts', **method_kwargs):
    '''
    Use analytic method if there are only two beams. Otherwise use constrained
    optimization to find the common beam.
    '''

    if beams.size == 1:
        return beams[0]
    elif fits_in_largest(beams):
        return beams.largest_beam()
    else:
        if beams.size == 2:
            try:
                return common_2beams(beams)
            # Sometimes this method can fail. Use the many beam solution in
            # this case
            except (ValueError, BeamError):
                pass

        if method == 'pts':
            return common_manybeams_mve(beams, **method_kwargs)
        elif method == 'opt':
            return common_manybeams_opt(beams, **method_kwargs)
        else:
            raise ValueError("method must be 'pts' or 'opt'.")


def common_2beams(beams, check_deconvolution=True):
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

    deconv_beam = large_beam.deconvolve(small_beam, failure_returns_pointlike=True)

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

        if check_deconvolution:
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
    # PA really shouldn't matter here. But the minimization performed better
    # in some cases with a non-zero PA. Presumably this is b/c the PA of the
    # common beam is affected more by the beam with the largest major axis.
    return bmaj[thisone], bmaj[thisone], bpa[thisone]


def PtoA(bmaj, bmin, bpa):
    '''
    Express the ellipse parameters into
    `center-form <https://en.wikipedia.org/wiki/Matrix_representation_of_conic_sections>`_.
    '''
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


def common_manybeams_opt(beams, p0=None, opt_method='Nelder-Mead',
                         optdict={'maxiter': 5000, 'ftol': 1e-14,
                                  'maxfev': 5000},
                         verbose=False,
                         brute=False, brute_steps=40):
    '''
    Optimize the common beam solution by maximizing the determinant of the
    common beam.

    ..note:: This method is experimental and requires further testing.

    Parameters
    ----------
    beams : `~radio_beam.Beams`
        Beams object.
    p0 : tuple, optional
        Initial guess parameters (`major, minor, pa`).
    opt_method : str, optional
        Optimization method to use. See `~scipy.optimize.minimize`.
        The default of Nelder-Mead is the only method we have had
        some success with.
    optdict : dict, optional
        Dictionary parameters passed to `~scipy.optimize.minimize`.
    verbose : bool, optional
        Print the full output from `~scipy.optimize.minimize`.
    brute : bool, optional
        Use `~scipy.optimize.brute` to find the optimal solution.
    brute_steps : int, optional
        Number of positions to sample in each parameter (3).

    Returns
    -------
    com_beam : `~radio_beam.Beam`
        Common beam.
    '''

    raise NotImplementedError("This method is not fully tested. Remove this "
                              "line for testing purposes.")

    if not HAS_SCIPY:
        raise ImportError("common_manybeams_opt requires scipy.optimize.")

    bmaj = beams.major.value
    bmin = beams.minor.value
    bpa = beams.pa.to(u.rad).value

    if p0 is None:
        p0 = boundingcircle(bmaj, bmin, bpa)
        # It seems to help to make the initial guess slightly larger
        p0 = (1.1 * p0[0], 1.1 * p0[1], p0[2])

    if brute:
        maj_range = [beams.major.max(), 1.5 * beams.major.max()]
        maj_step = (maj_range[1] - maj_range[0]) / brute_steps
        min_range = [beams.minor.min(), 1.5 * beams.major.max()]
        min_step = (min_range[1] - min_range[0]) / brute_steps
        rranges = (slice(maj_range[0], maj_range[1], maj_step),
                   slice(min_range[0], min_range[1], min_step),
                   slice(0, 179.9, 180. / brute_steps))
        result = opt.brute(myobjective_regularized, rranges,
                           args=(bmaj, bmin, bpa),
                           full_output=True,
                           finish=opt.fmin)
        params = result[0]

    else:
        result = opt.minimize(myobjective_regularized, p0,
                              method=opt_method,
                              args=(bmaj, bmin, bpa),
                              options=optdict,
                              tol=1e-14)
        params = result.x

        if verbose:
            print(result.viewitems())

        if not result.success:
            raise Warning("Optimization failed")

    com_beam = Beam(params[0] * beams.major.unit,
                    params[1] * beams.major.unit,
                    (params[2] % np.pi) * u.rad)

    # Test if it deconvolves all
    if not fits_in_largest(beams, com_beam):
        raise BeamError("Could not find common beam to deconvolve all beams.")

    return com_beam


def fits_in_largest(beams, large_beam=None):
    '''
    Test if all beams can be deconvolved by the largest beam
    '''

    if large_beam is None:
        large_beam = beams.largest_beam()

    for beam in beams:
        if large_beam == beam:
            continue
        deconv_beam = large_beam.deconvolve(beam,
                                            failure_returns_pointlike=True)

        if not deconv_beam.isfinite:
            return False

    return True


def getMinVolEllipse(P, tolerance=1e-5, maxiter=1e5):
    """
    Use the Khachiyan Algorithm to compute that minimum volume ellipsoid.

    For the purposes of finding a common beam, there is an added check that
    requires the center to be within the tolerance range.

    Adapted code from: https://github.com/minillinim/ellipsoid/blob/master/ellipsoid.py

    That implementation relies on the original work by Nima Moshtagh:
    http://www.mathworks.com/matlabcentral/fileexchange/9542
    and an alternate python version from:
    http://cctbx.sourceforge.net/current/python/scitbx.math.minimum_covering_ellipsoid.html

    Parameters
    ----------
    P : `~numpy.ndarray`
        Points to compute solution.
    tolerance : float, optional
        Allowed error range in the Khachiyan Algorithm. Decreasing the
        tolerance by an order of magnitude requires an order of magnitude
        more iterations to converge.
    maxiter : int, optional
        Maximum iterations.

    Returns
    -------
    center : `~numpy.ndarray`
        Center point of the ellipse. Is required to be smaller than the
        tolerance.
    radii : `~numpy.ndarray`
        Radii of the ellipse.
    rotation : `~numpy.ndarray`
        Rotation matrix of the ellipse.

    """
    N, d = P.shape
    d = float(d)

    # Q will be our working array
    Q = np.vstack([np.copy(P.T), np.ones(N)])
    QT = Q.T

    # initializations
    err = 1.0
    u = np.ones(N) / N

    # Khachiyan Algorithm
    i = 0
    while err > tolerance:
        V = np.dot(Q, np.dot(np.diag(u), QT))
        # M the diagonal vector of an NxN matrix
        M = np.diag(np.dot(QT, np.dot(np.linalg.inv(V), Q)))
        j = np.argmax(M)
        maximum = M[j]
        step_size = (maximum - d - 1.0) / ((d + 1.0) * (maximum - 1.0))
        new_u = (1.0 - step_size) * u
        err = np.linalg.norm(new_u - u)
        if err <= tolerance:
            break
        new_u[j] += step_size
        u = new_u
        i += 1
        if i == maxiter:
            raise ValueError("Reached maximum iterations without converging."
                             " Try increasing the tolerance.")

    # center of the ellipse
    center = np.atleast_2d(np.dot(P.T, u))

    # For our purposes, the centre should always be very small
    center_square = np.outer(center, center)
    if not (center_square < tolerance**2).any():
        raise ValueError("The solved centre ({0}) is larger than the tolerance"
                         " ({1}). Check the input data.".format(center,
                                                                tolerance))

    # the A matrix for the ellipse
    A = np.linalg.inv(np.dot(P.T, np.dot(np.diag(u), P)) -
                      center_square) / d

    # ellip_vals = np.dot(P - center, np.dot(A, (P - center).T))
    # assert (ellip_vals <= 1. + tolerance).all()

    # Get the values we'd like to return
    U, s, rotation = np.linalg.svd(A)
    radii = 1.0 / np.sqrt(s)
    radii *= 1. + tolerance

    return center, radii, rotation


def ellipse_edges(beam, npts=300, epsilon=1e-3):
    '''
    Return the edge points of the beam.

    Parameters
    ----------
    beam : `~radio_beam.Beam`
        Beam object.
    npts : int, optional
        Number of samples.
    epsilon : float
        Increase the radii of the ellipse by 1 + epsilon. This is to ensure
        that `getMinVolEllipse` returns a marginally deconvolvable beam to
        within the error tolerance.

    Returns
    -------
    pts : `~numpy.ndarray`
        The x, y coordinates of the ellipse edge.
    '''

    bpa = beam.pa.to(u.rad).value
    major = beam.major.to(u.deg).value * (1. + epsilon)
    minor = beam.minor.to(u.deg).value * (1. + epsilon)

    phi = np.linspace(0, 2 * np.pi, npts)

    x = major * np.cos(phi)
    y = minor * np.sin(phi)

    xr = x * np.cos(bpa) - y * np.sin(bpa)
    yr = x * np.sin(bpa) + y * np.cos(bpa)

    pts = np.vstack([xr, yr])
    return pts


def common_manybeams_mve(beams, tolerance=1e-4, nsamps=200,
                         epsilon=5e-4,
                         auto_increase_epsilon=True,
                         max_epsilon=1e-3,
                         max_iter=10):
    '''
    Calculate a common beam size using the Khachiyan Algorithm to find the
    minimum enclosing ellipse from all beam edges.

    Parameters
    ----------
    beams : `~radio_beam.Beams`
        Beams object.
    tolerance : float, optional
        Allowed error range in the Khachiyan Algorithm. Decreasing the
        tolerance by an order of magnitude requires an order of magnitude
        more iterations to converge.
    nsamps : int, optional
        Number of edge points to sample from each beam.
    epsilon : float, optional
        Increase the radii of each beam by a factor of 1 + epsilon to ensure
        the common beam can marginally be deconvolved for all beams. Small
        deviations result from the finite sampling of points and the choice
        of the tolerance.
    auto_increase_epsilon : bool, optional
        Re-run the algorithm when the solution cannot quite be deconvolved from
        from all the beams. When `True`, epsilon is slightly increased with
        each iteration until the common beam can be deconvolved from all beams.
        Default is `True`.
    max_epsilon : float, optional
        Maximum epsilon value that is acceptable. Reached with `max_iter`.
        Default is `1e-3`.
    max_iter : int, optional
        Maximum number of times to increase epsilon to try finding a valid
        common beam solution.

    Returns
    -------
    com_beam : `~radio_beam.Beam`
        The common beam for all beams in the set.
    '''

    if not HAS_SCIPY:
        raise ImportError("common_manybeams_mve requires scipy.optimize.")

    step = 1

    while True:
        pts = []

        for beam in beams:
            pts.append(ellipse_edges(beam, nsamps, epsilon=epsilon))

        all_pts = np.hstack(pts).T

        # Now find the outer edges of the convex hull.
        hull = ConvexHull(all_pts)
        edge_pts = all_pts[hull.vertices]

        center, radii, rotation = \
            getMinVolEllipse(edge_pts, tolerance=tolerance)

        # The rotation matrix is coming out as:
        # ((sin theta, cos theta)
        #  (cos theta, - sin theta))
        pa = np.arctan2(- rotation[0, 0], rotation[1, 0]) * u.rad

        if pa.value == -np.pi or pa.value == np.pi:
            pa = 0.0 * u.rad

        com_beam = Beam(major=radii.max() * u.deg, minor=radii.min() * u.deg,
                        pa=pa)

        # If common beam is just slightly smaller than one of the beams,
        # we increase epsilon to encourage a solution marginally larger
        # so all beams can be convolved.
        if auto_increase_epsilon:
            if not fits_in_largest(beams, com_beam):
                # Increase epsilon and run again
                epsilon += (step + 1) * (max_epsilon - epsilon) / max_iter
                step += 1

                if step == max_iter + 1:
                    raise BeamError("Could not increase epsilon to find"
                                    " common beam.")

                continue
            else:
                break
        else:
            break

    if not fits_in_largest(beams, com_beam):
        raise BeamError("Could not find common beam to deconvolve all beams.")

    return com_beam
