from astropy import units as u
from astropy.io import fits
from astropy import constants
from astropy import wcs
from astropy.extern import six
import numpy as np
import warnings

from .beam import Beam, _to_area, SIGMA_TO_FWHM
from .commonbeam import commonbeam


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

    def __getslice__(self, start, stop, increment=None):
        return self.__getitem__(slice(start, stop, increment))

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
        else:
            raise ValueError("Invalid slice")

    def __array_finalize__(self, obj):
        # If our unit is not set and obj has a valid one, use it.
        if self._unit is None:
            unit = getattr(obj, '_unit', None)
            if unit is not None:
                self._set_unit(unit)

        if isinstance(obj, Beams):
            self.major = obj.major
            self.minajor = obj.minor
            self.pa = obj.pa
            self.meta = obj.meta

        # Copy info if the original had `info` defined.  Because of the way the
        # DataInfo works, `'info' in obj.__dict__` is False until the
        # `info` attribute is accessed or set.  Note that `obj` can be an
        # ndarray which doesn't have a `__dict__`.
        if 'info' in getattr(obj, '__dict__', ()):
            self.info = obj.info

    @property
    def sr(self):
        return _to_area(self.major, self.minor)

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

    @classmethod
    def from_casa_image(cls, imagename):
        '''
        Instantiate beams from a CASA image. Cannot currently handle beams for
        different polarizations.

        ** Must be run in a CASA environment! **

        Parameters
        ----------
        imagename : str
            Name of CASA image.
        '''

        try:
            import casac
        except ImportError:
            raise ImportError("Could not import CASA (casac) and therefore"
                              " cannot read CASA .image files")

        ia.open(imagename)
        beam_props = ia.restoringbeam()
        ia.close()

        nchans = beam_props['nChannels']

        # Assuming there is always a 0th channel...
        maj_unit = u.Unit(beam_props['beams']['*0']['*0']['major']['unit'])
        min_unit = u.Unit(beam_props['beams']['*0']['*0']['minor']['unit'])
        pa_unit = u.Unit(beam_props['beams']['*0']['*0']['positionangle']['unit'])

        major = np.empty((nchans)) * maj_unit
        minor = np.empty((nchans)) * min_unit
        pa = np.empty((nchans)) * pa_unit

        for chan in range(nchans):

            chan_name = '*{}'.format(chan)
            chanbeam_props = beam_props['beams'][chan_name]['*0']

            # Can CASA have mixes of units between channels? Let's test just
            # in case
            assert maj_unit == u.Unit(chanbeam_props['major']['unit'])
            assert min_unit == u.Unit(chanbeam_props['minor']['unit'])
            assert pa_unit == u.Unit(chanbeam_props['positionangle']['unit'])

            major[chan] = chanbeam_props['major']['value'] * maj_unit
            minor[chan] = chanbeam_props['minor']['value'] * min_unit
            pa[chan] = chanbeam_props['positionangle']['value'] * pa_unit

        return cls(major=major, minor=minor, pa=pa)

    def average_beam(self, includemask=None, raise_for_nan=True):
        """
        Average the beam major, minor, and PA attributes.

        This is usually a dumb thing to do!
        """

        warnings.warn("Do not use the average beam for convolution! Use the"
                      " smallest common beam from `Beams.common_beam()`.")

        from astropy.stats import circmean

        if includemask is None:
            includemask = self.isfinite
        else:
            includemask = np.logical_and(includemask, self.isfinite)

        new_beam = Beam(major=self.major[includemask].mean(),
                        minor=self.minor[includemask].mean(),
                        pa=circmean(self.pa[includemask],
                        weights=(self.major / self.minor)[includemask]))

        if raise_for_nan and np.any(np.isnan(new_beam)):
            raise ValueError("NaNs after averaging.  This is a bug.")

        return new_beam

    def largest_beam(self, includemask=None):
        """
        Returns the largest beam (by area) in a list of beams.
        """

        if includemask is None:
            includemask = self.isfinite
        else:
            includemask = np.logical_and(includemask, self.isfinite)

        largest_idx = (self.major * self.minor)[includemask].argmax()
        new_beam = Beam(major=self.major[includemask][largest_idx],
                        minor=self.minor[includemask][largest_idx],
                        pa=self.pa[includemask][largest_idx])

        return new_beam

    def smallest_beam(self, includemask=None):
        """
        Returns the smallest beam (by area) in a list of beams.
        """

        if includemask is None:
            includemask = self.isfinite
        else:
            includemask = np.logical_and(includemask, self.isfinite)

        largest_idx = (self.major * self.minor)[includemask].argmin()
        new_beam = Beam(major=self.major[includemask][largest_idx],
                        minor=self.minor[includemask][largest_idx],
                        pa=self.pa[includemask][largest_idx])

        return new_beam

    def extrema_beams(self, includemask=None):
        return [self.smallest_beam(includemask),
                self.largest_beam(includemask)]

    def common_beam(self, includemask=None, method='pts', **kwargs):
        '''
        Return the smallest common beam size. For set of two beams,
        the solution is solved analytically. All larger sets solve for the
        minimum volume ellipse using the
        `Khachiyan Algorithm <http://www.mathworks.com/matlabcentral/fileexchange/9542>`_,
        where the convex hull of the set of ellipse edges is used to find the
        boundaries of the set.

        Parameters
        ----------
        includemask : `~numpy.ndarray`, optional
            Boolean mask.
        method : {'pts'}, optional
            Many beam method. Only `pts` is currently available.
        kwargs : Passed to `~radio_beam.commonbeam`.

        '''
        return commonbeam(self if includemask is None else self[includemask],
                          method=method, **kwargs)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
