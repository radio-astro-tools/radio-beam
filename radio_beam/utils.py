
import math
import numpy as np
import astropy.units as u


DEG2RAD = math.pi / 180.

class BeamError(Exception):
    """docstring for BeamError"""
    pass


class InvalidBeamOperationError(Exception):
    pass


class RadioBeamDeprecationWarning(Warning):
    pass


def deconvolve_optimized(beamprops1, beamprops2, failure_returns_pointlike=False):
    """
    An optimized, non-Quantity version of beam deconvolution.

    Because no unit conversions are handled, the inputs MUST be in degrees for the major, minor,
    and position angle.

    Parameters
    ----------
    beamprops1: dict
        Dictionary with keys 'BMAJ', 'BMIN', and 'BPA' for the beam to deconvolve from. Can be
        produced with `~radio_beam.Beam.to_fits_keywords`.
    beamprops2: dict
        Same as `beamprops1` for the second beam.
    failure_returns_pointlike : bool, optional
        Return a point beam (zero area) when deconvolution fails. If `False`,
        this will instead raise a `~radio_beam.utils.BeamError` when deconvolution fails.


    Returns
    -------
    new_major : float
        Deconvolved major FWHM.
    new_minor : float
        Deconvolved minor FWHM.
    new_pa : float
        Deconvolved position angle.

    """

    # blame: https://github.com/pkgw/carma-miriad/blob/CVSHEAD/src/subs/gaupar.for
    # (githup checkin of MIRIAD, code by Sault)

    maj1 = beamprops1['BMAJ']
    min1 = beamprops1['BMIN']
    pa1 = beamprops1['BPA'] * DEG2RAD

    maj2 = beamprops2['BMAJ']
    min2 = beamprops2['BMIN']
    pa2 = beamprops2['BPA'] * DEG2RAD

    alpha = ((maj1 * math.cos(pa1))**2 +
             (min1 * math.sin(pa1))**2 -
             (maj2 * math.cos(pa2))**2 -
             (min2 * math.sin(pa2))**2)

    beta = ((maj1 * math.sin(pa1))**2 +
            (min1 * math.cos(pa1))**2 -
            (maj2 * math.sin(pa2))**2 -
            (min2 * math.cos(pa2))**2)

    gamma = 2 * ((min1**2 - maj1**2) * math.sin(pa1) * math.cos(pa1) -
                 (min2**2 - maj2**2) * math.sin(pa2) * math.cos(pa2))

    s = alpha + beta
    t = math.sqrt((alpha - beta)**2 + gamma**2)

    # Deal with floating point issues
    # This matches the arcsec**2 check for deconvolve below
    # Difference is we keep things in deg^2 here
    atol_t = np.finfo(np.float64).eps / 3600.**2

    # To deconvolve, the beam must satisfy:
    # alpha < 0
    alpha_cond = alpha + np.finfo(np.float64).eps < 0

    # beta < 0
    beta_cond = beta + np.finfo(np.float64).eps < 0
    # s < t
    st_cond = s < t + atol_t

    if alpha_cond or beta_cond or st_cond:
        if failure_returns_pointlike:
            return 0., 0., 0.
        else:
            raise BeamError("Beam could not be deconvolved")
    else:
        new_major = math.sqrt(0.5 * (s + t))
        new_minor = math.sqrt(0.5 * (s - t))

        # absolute tolerance needs to be <<1 microarcsec
        atol = 1e-7 / 3600.
        if (math.sqrt(abs(gamma) + abs(alpha - beta))) < atol:
            new_pa = 0.0
        else:
            new_pa = 0.5 * math.atan2(-1. * gamma, alpha - beta)

    # In the limiting case, the axes can be zero to within precision
    # Add the precision level onto each axis so a deconvolvable beam
    # is always has beam.isfinite == True
    new_major += np.finfo(np.float64).eps
    new_minor += np.finfo(np.float64).eps

    return new_major, new_minor, new_pa


def deconvolve(beam, other, failure_returns_pointlike=False):
    """
    Deconvolve a beam from another

    Parameters
    ----------
    beam : `Beam`
        The defined beam.
    other : `Beam`
        The beam to deconvolve from this beam
    failure_returns_pointlike : bool
        Option to return a pointlike beam (i.e., one with major=minor=0) if
        the second beam is larger than the first.  Otherwise, a ValueError
        will be raised

    Returns
    -------
    new_beam : `Beam`
        The convolved Beam

    Raises
    ------
    failure : ValueError
        If the second beam is larger than the first, the default behavior
        is to raise an exception.  This can be overridden with
        failure_returns_pointlike
    """


    # The header keywords handle the conversions to degree for BMAJ, BMIN, BPA.
    beamprops1 = beam.to_header_keywords()
    beamprops2 = other.to_header_keywords()

    return deconvolve_optimized(beamprops1, beamprops2,
                                failure_returns_pointlike=failure_returns_pointlike)


def convolve(beam, other):
    """
    Convolve one beam with another.

    Parameters
    ----------
    other : `Beam`
        The beam to convolve with

    Returns
    -------
    new_beam : `Beam`
        The convolved Beam
    """

    # blame: https://github.com/pkgw/carma-miriad/blob/CVSHEAD/src/subs/gaupar.for
    # (github checkin of MIRIAD, code by Sault)

    alpha = ((beam.major * np.cos(beam.pa))**2 +
             (beam.minor * np.sin(beam.pa))**2 +
             (other.major * np.cos(other.pa))**2 +
             (other.minor * np.sin(other.pa))**2)

    beta = ((beam.major * np.sin(beam.pa))**2 +
            (beam.minor * np.cos(beam.pa))**2 +
            (other.major * np.sin(other.pa))**2 +
            (other.minor * np.cos(other.pa))**2)

    gamma = (2 * ((beam.minor**2 - beam.major**2) *
                  np.sin(beam.pa) * np.cos(beam.pa) +
                  (other.minor**2 - other.major**2) *
                  np.sin(other.pa) * np.cos(other.pa)))

    s = alpha + beta
    t = np.sqrt((alpha - beta)**2 + gamma**2)

    new_major = np.sqrt(0.5 * (s + t))
    new_minor = np.sqrt(0.5 * (s - t))
    # absolute tolerance needs to be <<1 microarcsec
    if np.isclose(((abs(gamma) + abs(alpha - beta))**0.5).to(u.arcsec).value, 1e-7):
        new_pa = 0.0 * u.deg
    else:
        new_pa = 0.5 * np.arctan2(-1. * gamma, alpha - beta)

    return new_major, new_minor, new_pa


def transform_ellipse(major, minor, pa, x_scale, y_scale):
    '''
    Transform an ellipse by scaling in the x and y axes.

    Parameters
    ----------
    major : `~astropy.units.Quantity`
        Major axis.
    minor : `~astropy.units.Quantity`
        Minor axis.
    pa : `~astropy.units.Quantity`
        PA of the major axis.
    x_scale : float
        x axis scaling factor.
    y_scale : float
        y axis scaling factor.

    Returns
    -------
    trans_major : `~astropy.units.Quantity`
        Major axis in the transformed frame.
    trans_minor : `~astropy.units.Quantity`
        Minor axis in the transformed frame.
    trans_pa : `~astropy.units.Quantity`
        PA of the major axis in the transformed frame.
    '''

    # This code is based on the implementation in CASA:
    # https://open-bitbucket.nrao.edu/projects/CASA/repos/casa/browse/code/imageanalysis/ImageAnalysis/CasaImageBeamSet.cc

    major = major.to(u.arcsec)
    minor = minor.to(u.arcsec)
    pa = pa.to(u.rad)

    cospa = np.cos(pa)
    sinpa = np.sin(pa)
    cos2pa = cospa**2
    sin2pa = sinpa**2

    major2 = major**2
    minor2 = minor**2

    a = (cos2pa / major2) + (sin2pa / minor2)
    b = -2 * cospa * sinpa * (major2**-1 - minor2**-1)
    c = (sin2pa / major2) + (cos2pa / minor2)

    x2_scale = x_scale**2
    y2_scale = y_scale**2

    r = a / x2_scale
    s = b**2 / (4 * x2_scale * y2_scale)
    t = c / y2_scale

    udiff = r - t
    u2 = udiff**2

    f1 = u2 + 4 * s
    f2 = np.sqrt(f1) * np.abs(udiff)

    j1 = (f2 + f1) / f1 / 2
    j2 = (f1 - f2) / f1 / 2

    k1 = (j1 * r + j1 * t - t) / (2 * j1 - 1)
    k2 = (j2 * r + j2 * t - t) / (2 * j2 - 1)

    c1 = np.sqrt(k1)**-1
    c2 = np.sqrt(k2)**-1

    pa_sign = 1 if pa.value >= 0 else -1

    if c1 == c2:
        # Transformed to a circle
        trans_major = 1 / c1
        trans_minor = trans_major
        trans_pa = 0. * u.rad

    elif c1 > c2:
        # c1 and c2 are the major and minor axes; use j1 to get PA
        trans_major = c1
        trans_minor = c2
        trans_pa = pa_sign * np.arccos(np.sqrt(j1))
    else:
        # Opposite case where the axes are switched; get PA from j2
        trans_major = c2
        trans_minor = c1
        trans_pa = pa_sign * np.arccos(np.sqrt(j2))

    return trans_major, trans_minor, trans_pa
