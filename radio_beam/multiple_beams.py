from astropy import units as u
from astropy.io import fits
from astropy import constants
import astropy.units as u
from astropy import wcs
from astropy.extern import six
import numpy as np
import warnings

from .beam import Beam, _to_area, SIGMA_TO_FWHM


class Beams(u.Quantity):
    """
    An object to handle a set of radio beams for a data cube.
    """
    def __new__(cls, major=None, minor=None, pa=None,
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
            major = rad * SIGMA_TO_FWHM
            minor = rad * SIGMA_TO_FWHM
            pa = np.zeros_like(areas) * u.deg

        # give specified values priority
        if major is not None:
            if u.deg.is_equivalent(major.unit):
                pass
            else:
                warnings.warn("Assuming major axes has been specified in degrees")
                major = major * u.deg
        if minor is not None:
            if u.deg.is_equivalent(minor.unit):
                pass
            else:
                warnings.warn("Assuming minor axes has been specified in degrees")
                minor = minor * u.deg
        if pa is not None:
            if len(pa) != len(major):
                raise ValueError("Number of position angles must match number of major axis lengths")
            if u.deg.is_equivalent(pa.unit):
                pass
            else:
                warnings.warn("Assuming position angles has been specified in degrees")
                pa = pa * u.deg
        else:
            pa = np.zeros_like(major.value) * u.deg

        # some sensible defaults
        if minor is None:
            minor = major
        elif len(minor) != len(major):
            raise ValueError("Minor and major axes must have same number of values")

        self = super(Beams, cls).__new__(cls, value=_to_area(major, minor).value, unit=u.sr)
        self.major = major
        self.minor = minor
        self.pa = pa
        self.default_unit = default_unit

        if meta is None:
            self.meta = [{}]*len(self)
        else:
            self.meta = meta

        return self

    @property
    def meta(self):
        return self._meta

    @meta.setter
    def meta(self, value):
        if len(value) == len(self):
            self._meta = value
        else:
            raise TypeError("metadata must be a list of dictionaries")

    def __len__(self):
        return len(self.major)


    @property
    def isfinite(self):
        return ((self.major > 0) & (self.minor > 0) & np.isfinite(self.major) &
                np.isfinite(self.minor) & np.isfinite(self.pa))

    def __getitem__(self, view):
        if isinstance(view, int):
            return Beam(major=self.major[view],
                        minor=self.minor[view],
                        pa=self.pa[view],
                        meta=self.meta[view])
        elif isinstance(view, slice):
            return Beams(major=self.major[view],
                         minor=self.minor[view],
                         pa=self.pa[view],
                         meta=self.meta[view])
        elif isinstance(view, np.ndarray):
            if view.dtype.name != 'bool':
                raise ValueError("If using an array to index beams, it must "
                                 "be a boolean array.")
            return Beams(major=self.major[view],
                         minor=self.minor[view],
                         pa=self.pa[view],
                         meta=[x for ii,x in zip(view, self.meta) if ii])


    @classmethod
    def from_fits_bintable(cls, bintable):
        """
        Instantiate a Beams list from a bintable from a CASA-produced image
        HDU.

        Parameters
        ----------
        bintable : fits.BinTableHDU
            The table data containing the beam information

        Returns
        -------
        beams : Beams
            A new Beams object
        """
        major = u.Quantity(bintable.data['BMAJ'], u.arcsec)
        minor = u.Quantity(bintable.data['BMIN'], u.arcsec)
        pa = u.Quantity(bintable.data['BPA'], u.deg)
        meta = [{key: row[key] for key in bintable.columns.names
                 if key not in ('BMAJ', 'BPA', 'BMIN')}
                for row in bintable.data]

        return cls(major=major, minor=minor, pa=pa, meta=meta)
