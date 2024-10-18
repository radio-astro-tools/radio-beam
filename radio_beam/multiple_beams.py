
from astropy import units as u
from astropy.io import fits
from astropy import constants
from astropy import wcs
import numpy as np
import warnings

from .beam import Beam, _to_area, SIGMA_TO_FWHM, _with_default_unit
from .commonbeam import commonbeam
from .utils import InvalidBeamOperationError


class Beams(u.Quantity):
    """
    An object to handle a set of radio beams for a data cube.
    """

    def __new__(cls, major=None, minor=None, pa=None,
                areas=None, default_unit=u.arcsec, meta=None,
                beams=None):
        """
        Create a new set of Gaussian beams

        Parameters
        ----------
        major : :class:`~astropy.units.Quantity` with angular equivalency
            The FWHM major axes
        minor : :class:`~astropy.units.Quantity` with angular equivalency
            The FWHM minor axes
        pa : :class:`~astropy.units.Quantity` with angular equivalency
            The beam position angles
        areas : :class:`~astropy.units.Quantity` with steradian equivalency
            The area of the beams.  This is an alternative to specifying the
            major/minor/PA, and will create those values assuming a circular
            Gaussian beam.
        default_unit : :class:`~astropy.units.Unit`
            The unit to impose on major, minor if they are specified as floats
        meta : dict, optional
            A dictionary of metadata to include in the header.
        beams : List of :class:`~radio_beam.Beam` objects
            List of individual `Beam` objects. The resulting `Beams` object will
            have major and minor axes in degrees.
        """

        # improve to some kwargs magic later

        # error checking

        if beams is not None:
            major = [beam.major.to(u.deg).value for beam in beams] * u.deg
            minor = [beam.minor.to(u.deg).value for beam in beams] * u.deg
            pa = [beam.pa.to(u.deg).value for beam in beams] * u.deg

        # ... given an area make a round beam assuming it is Gaussian
        if areas is not None:
            rad = np.sqrt(areas / (2 * np.pi)) * u.deg
            major = rad * SIGMA_TO_FWHM
            minor = rad * SIGMA_TO_FWHM
            pa = np.zeros_like(areas) * u.deg

        # give specified values priority
        if major is not None:
            major = _with_default_unit("major", major, default_unit)


        if pa is not None:
            if len(pa) != len(major):
                raise ValueError("Number of position angles must match number of major axis lengths")
            pa = _with_default_unit("pa", pa, u.deg)
        else:
            pa = np.zeros(major.shape) * u.deg

        # some sensible defaults
        if minor is None:
            minor = major
        elif len(minor) != len(major):
            raise ValueError("Minor and major axes must have same number of values")
        else:
            minor = _with_default_unit("minor", minor, default_unit)

        if np.any(minor > major):
            raise ValueError("Minor axis greater than major axis.")

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
        if isinstance(view, (int, np.int64)):
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
            # Multiplication and division should change the area,
            # but not the PA or major/minor ratio
            self.major = obj.major
            self.minor = obj.minor
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
        Instantiate a Beams list from a bintable HDU from a CASA-produced image
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

        header = bintable.header

        # Read the bmaj/bmin units from the header
        # (we still assume BPA is degrees because we've never seen an exceptional case)
        # this will crash if there is no appropriate header info
        maj_kw = [kw for kw, val in header.items() if val == 'BMAJ'][0]
        min_kw = [kw for kw, val in header.items() if val == 'BMIN'][0]
        maj_unit = header[maj_kw.replace('TTYPE', 'TUNIT')]
        min_unit = header[min_kw.replace('TTYPE', 'TUNIT')]

        # AIPS uses non-FITS-standard unit names; this catches the
        # only case we've seen so far
        if maj_unit == 'DEGREES':
            maj_unit = 'degree'
        if min_unit == 'DEGREES':
            min_unit = 'degree'

        maj_unit = u.Unit(maj_unit)
        min_unit = u.Unit(min_unit)

        major = u.Quantity(bintable.data['BMAJ'], maj_unit)
        minor = u.Quantity(bintable.data['BMIN'], min_unit)
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
            from casatools import image as iatool

            ia = iatool()

        except ImportError:

            raise ImportError("Could not import CASA and therefore"
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

        Since the minimum ellipse method is approximate, some solutions for
        the common beam will be slightly underestimated and the solution
        cannot be deconvolved from the whole set of beams. To overcome
        this issue, a small `epsilon` correction factor is added to the
        ellipse edges to encourage a valid common beam solution.
        Since `epsilon` is added to all sides, this correction will at most
        increase the common beam size by :math:`2\times(1+\epsilon)`.
        The default values of `epsilon` is :math:`5\times10^{-4}`, so this
        will have a very small effect on the size of the common beam.

        In some cases, `epsilon` must be increased to find a valid common
        beam solution. The algorithm does this by default
        (set by `auto_increase_epsilon=True`), and will incrementally
        increase `epsilon` until the common beam can be deconvolved from
        all beams, or until either(1) `max_iter` is reached (default is 10)
        or (2) `max_epsilon` is reached (default is 1e-3). In practice, we
        find these settings work well for different ALMA or VLA data, but
        these `kwargs` may need to be changed for discrepant cases.


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

    def __mul__(self, other):
        # Other must be a single beam. Assume multiplying is convolving
        # as set of beams with a given beam
        if not isinstance(other, Beam):
            raise InvalidBeamOperationError("Multiplication is defined as a "
                                            "convolution of the set of beams "
                                            "with a given beam. Must be "
                                            "multiplied with a Beam object.")

        return Beams(beams=[beam * other for beam in self])

    def __truediv__(self, other):
        # Other must be a single beam. Assume dividing is deconvolving
        # as set of beams with a given beam
        if not isinstance(other, Beam):
            raise InvalidBeamOperationError("Division is defined as a "
                                            "deconvolution of the set of beams"
                                            " with a given beam. Must be "
                                            "divided by a Beam object.")

        return Beams(beams=[beam / other for beam in self])

    def __add__(self, other):
        raise InvalidBeamOperationError("Addition of a set of Beams "
                                        "is not defined.")

    def __sub__(self, other):
        raise InvalidBeamOperationError("Addition of a set of Beams "
                                        "is not defined.")

    def __eq__(self, other):
        # other should be a single beam, or a another Beams object
        if isinstance(other, Beam):
            return np.array([beam == other for beam in self])
        elif isinstance(other, Beams):
            # These should have the same size.
            if not self.size == other.size:
                raise InvalidBeamOperationError("Beams objects must have the "
                                                "same shape to test "
                                                "equality.")

            return np.all([beam == other_beam for beam, other_beam in
                           zip(self, other)])
        else:
            raise InvalidBeamOperationError("Must test equality with a Beam"
                                            " or Beams object.")

    def __ne__(self, other):
        eq_out = self.__eq__(other)

        # If other is a Beam, will get array back
        if isinstance(eq_out, np.ndarray):
            return ~eq_out
        # If other is a Beams, will get boolean back
        else:
            return not eq_out
