
import numpy as np
import astropy.units as u

from .beam import Beam
from .utils import BeamError, transform_ellipse


def commonbeam(beams):
    '''
    Find a common beam that can be convolved to from a `Beams` object. This
    function is based on the CASA implementation `ia.commonbeam`.

    Parameters
    ----------
    beams : `~radio_beam.Beams`

    Returns
    -------
    common_beam : `~radio_beam.Beam`
        The smallest common beam in the set of beams.
    '''

    # This code is based on the implementation in CASA:
    # https://open-bitbucket.nrao.edu/projects/CASA/repos/casa/browse/code/imageanalysis/ImageAnalysis/CasaImageBeamSet.cc

    if (~beams.isfinite).all():
        raise BeamError("All beams in the object are invalid.")

    largest_beam = beams.largest_beam()
    largest_major = largest_beam.major
    largest_minor = largest_beam.minor

    for beam in beams:

        if beam != largest_beam and beam.isfinite:
            deconv_beam = \
                largest_beam.deconvolve(beam, failure_returns_pointlike=True)
            if not deconv_beam.isfinite:

                pa_diff = (beam.pa - largest_beam.pa).to(u.rad)

                problem_major = beam.major
                problem_minor = beam.minor

                # If the difference is pi / 2, the larger major is set to the
                # new major and the minor is the other major.
                if np.abs(pa_diff).value == np.pi / 2.:
                    larger_major = largest_beam.major >= beam.major
                    major = largest_major if larger_major else problem_major
                    minor = problem_major if larger_major else largest_major
                    pa = largest_beam.pa if larger_major else beam.pa
                    largest_beam = Beam(major=major, minor=minor, pa=pa)
                    continue

                # Transform to coordinates where largest_beam is circular

                major_comb = np.sqrt(largest_major * problem_major)
                p = major_comb / largest_major
                q = major_comb / largest_minor

                # Transform beam into the same coordinates, and rotate so its
                # major axis is along the x axis.

                trans_major_sc, trans_minor_sc, trans_pa_sc = \
                    transform_ellipse(problem_major, problem_minor, pa_diff,
                                      p, q)

                # The transformed minor axis is major_comb, as defined in CASA
                trans_minor_sc = major_comb

                # Return beam to the original coordinates, still rotated with
                # the major along the x axis
                trans_major_unsc, trans_minor_unsc, trans_pa_unsc = \
                    transform_ellipse(trans_major_sc, trans_minor_sc,
                                      trans_pa_sc,
                                      1 / p, 1 / q)

                # Lastly, rotate the PA to the enclosing ellipse

                trans_major = trans_major_unsc
                trans_minor = trans_minor_unsc
                trans_pa = trans_pa_unsc + largest_beam.pa

                trans_beam = Beam(major=trans_major, minor=trans_minor,
                                  pa=trans_pa)

                # Ensure this beam can now be deconvolved
                # Scale the enclosing ellipse by a small factor until it does.
                can_deconv = False
                num = 0
                while not can_deconv:
                    deconv_large_beam = \
                        trans_beam.deconvolve(largest_beam,
                                              failure_returns_pointlike=True)
                    deconv_prob_beam = \
                        trans_beam.deconvolve(beam,
                                              failure_returns_pointlike=True)

                    if (not deconv_large_beam.isfinite or
                        not deconv_prob_beam.isfinite):

                        scale_factor = 1.0 + 1e-8
                        trans_beam = Beam(major=trans_major * scale_factor,
                                          minor=trans_minor * scale_factor,
                                          pa=trans_pa)
                        num += 1
                    else:
                        can_deconv = True

                largest_beam = trans_beam

    common_beam = largest_beam
    return common_beam