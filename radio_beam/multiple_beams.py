from astropy import units as u
from astropy.io import fits
from astropy import constants
import astropy.units as u
from astropy import wcs
from astropy.extern import six
import numpy as np
import warnings

from .beam import Beam, _to_area


class Beams(object):
    """
    An object to handle a set of radio beams for a data cube.
    """
    def __new__(cls, beams=None, majors=None, minors=None, pas=None,
                areas=None, default_unit=u.arcsec, meta=None):
        """
        Create a new set of Gaussian beams

        Parameters
        ----------
        major : :class:`~astropy.units.Quantity` with angular equivalency
        minor : :class:`~astropy.units.Quantity` with angular equivalency
        pa : :class:`~astropy.units.Quantity` with angular equivalency
        area : :class:`~astropy.units.Quantity` with steradian equivalency
        default_unit : :class:`~astropy.units.Unit`
            The unit to impose on major, minor if they are specified as floats
        """

        # improve to some kwargs magic later

        # error checking

        # ... given an area make a round beam assuming it is Gaussian
        if areas is not None:
            rad = np.sqrt(areas / (2 * np.pi)) * u.deg
            majors = rad * SIGMA_TO_FWHM
            minors = rad * SIGMA_TO_FWHM
            pas = np.zeros_like(areas) * u.deg

        # give specified values priority
        if majors is not None:
            if u.deg.is_equivalent(majors):
                majors = majors
            else:
                warnings.warn("Assuming major axes has been specified in degrees")
                majors = majors * u.deg
        if minors is not None:
            if u.deg.is_equivalent(minors):
                minors = minors
            else:
                warnings.warn("Assuming minor axes has been specified in degrees")
                minors = minors * u.deg
        if pas is not None:
            if u.deg.is_equivalent(pas):
                pas = pas
            else:
                warnings.warn("Assuming position angles has been specified in degrees")
                pas = pas * u.deg
        else:
            pas = np.zeros_like(pas) * u.deg

        # some sensible defaults
        if minors is None:
            minors = majors

        if beams is None:
            beams = [Beam(major=major, minor=minor, pa=pa)
                     for major, minor, pa in zip(majors, minors, pas)]


        self = super(Beams, cls).__new__(cls, _to_area(majors, minors).value, u.sr)
        self._majors = majors
        self._minors = minors
        self._pas = pas
        self.default_unit = default_unit

        if meta is None:
            self.meta = {}
        elif isinstance(meta, dict):
            self.meta = meta
        else:
            raise TypeError("metadata must be a dictionary")

        return self

    @property
    def beams(self):
        return self._beams

    @beams.setter
    def beams(self, beam_input):
        for beam in beam_input:
            if isinstance(beam, Beam):
                continue
            raise TypeError("All items in the beam list must be Beam objects.")

        self._beams = beam_input
