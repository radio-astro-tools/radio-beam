
import astropy.units as u
from astropy.modeling.models import Gaussian2D

from .beam import Beam
from .utils import deconvolve_optimized


def deconvolve_source(source, mybeam, failure_returns_pointlike=False):
    '''
    Deconvolve the beam size from a 2D Gaussian-shaped source.

    Parameters
    ----------
    source : `~astropy.modeling.models.Gaussian2D`
        A 2D model for the given source.
    mybeam : `~radio_beam.Beam`
        The Beam to deconvolve the source by.
    failure_returns_pointlike : bool, optional
        If the source cannot be deconvolved, return a point beam (i.e., of zero area).
        Default is False, meaning a BeamError is raised if the source cannot be deconvolved
        by the beam.

    Returns
    -------
    deconv_source : `~astropy.modeling.models.Gaussian2D`
        A 2D model with the deconvolved source size.

    '''

    if not isinstance(source, Gaussian2D):
        raise TypeError(f"`source` must be an astropy.modeling.models.Gaussian2D object.")

    if not isinstance(mybeam, Beam):
        raise TypeError(f"`beam` must be a radio_beam.Beam object.")


    # Extract the major, minor, and pa from the source model
    source_major = source.x_stddev.quantity if source.x_stddev > source.y_stddev \
        else source.y_stddev.quantity
    source_minor = source.y_stddev.quantity if source.x_stddev > source.y_stddev \
        else source.x_stddev.quantity
    source_pa = source.theta.value * u.deg if source.theta.quantity is None else source.theta.quantity

    # Use deg everywhere to match the WCS convention for a header
    # (expected by deconvolve_optimized)
    source_props = {"BMAJ": source_major.to(u.deg).value,
                    "BMIN": source_minor.to(u.deg).value,
                    "BPA": source_pa.to(u.deg).value
                    }

    beam_props = mybeam.to_header_keywords()

    deconv_major, deconv_minor, deconv_pa = \
            deconvolve_optimized(source_props,
                                 beam_props,
                                 failure_returns_pointlike=failure_returns_pointlike)

    deconv_major *= u.deg
    deconv_minor *= u.deg
    deconv_pa *= u.deg

    # Re-assemble until a Gaussian2D model
    deconv_source = Gaussian2D(amplitude=source.amplitude,
                               x_mean=source.x_mean,
                               y_mean=source.y_mean,
                               x_stddev=deconv_major,
                               y_stddev=deconv_minor,
                               theta=deconv_pa)

    return deconv_source
