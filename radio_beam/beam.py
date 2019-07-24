import six
from astropy import units as u
from astropy.io import fits
from astropy import constants
from astropy import wcs
import numpy as np
import warnings

# Imports for the custom kernels
from astropy.modeling.models import Ellipse2D, Gaussian2D
from astropy.convolution import Kernel2D
from astropy.convolution.kernels import _round_up_to_odd_integer

from .utils import deconvolve, convolve, RadioBeamDeprecationWarning

# Conversion between a twod Gaussian FWHM**2 and effective area
FWHM_TO_AREA = 2*np.pi/(8*np.log(2))
SIGMA_TO_FWHM = np.sqrt(8*np.log(2))

class NoBeamException(Exception):
    pass

def _to_area(major,minor):
    return (major * minor * FWHM_TO_AREA).to(u.sr)

unit_format = {u.deg: r'\\circ',
               u.arcsec: "''",
               u.arcmin: "'"}


class Beam(u.Quantity):
    """
    An object to handle single radio beams.
    """

    def __new__(cls, major=None, minor=None, pa=None, area=None,
                default_unit=u.arcsec, meta=None):
        """
        Create a new Gaussian beam

        Parameters
        ----------
        major : :class:`~astropy.units.Quantity` with angular equivalency
            The FWHM major axis
        minor : :class:`~astropy.units.Quantity` with angular equivalency
            The FWHM minor axis
        pa : :class:`~astropy.units.Quantity` with angular equivalency
            The beam position angle
        area : :class:`~astropy.units.Quantity` with steradian equivalency
            The area of the beam.  This is an alternative to specifying the
            major/minor/PA, and will create those values assuming a circular
            Gaussian beam.
        default_unit : :class:`~astropy.units.Unit`
            The unit to impose on major, minor if they are specified as floats
        """

        # improve to some kwargs magic later

        # error checking

        # ... given an area make a round beam assuming it is Gaussian
        if area is not None:
            if major is not None:
                raise ValueError("Can only specify one of {major,minor,pa} "
                                 "and {area}")
            rad = np.sqrt(area/(2*np.pi)) * u.deg
            major = rad * SIGMA_TO_FWHM
            minor = rad * SIGMA_TO_FWHM
            pa = 0.0 * u.deg

        # give specified values priority
        if major is not None:
            if u.deg.is_equivalent(major):
                major = major
            else:
                warnings.warn("Assuming major axis has been specified in degrees")
                major = major * u.deg
        if minor is not None:
            if u.deg.is_equivalent(minor):
                minor = minor
            else:
                warnings.warn("Assuming minor axis has been specified in degrees")
                minor = minor * u.deg
        if pa is not None:
            if u.deg.is_equivalent(pa):
                pa = pa
            else:
                warnings.warn("Assuming position angle has been specified in degrees")
                pa = pa * u.deg
        else:
            pa = 0.0 * u.deg

        # some sensible defaults
        if minor is None:
            minor = major

        if minor > major:
            raise ValueError("Minor axis greater than major axis.")

        self = super(Beam, cls).__new__(cls, _to_area(major,minor).value, u.sr)
        self._major = major
        self._minor = minor
        self._pa = pa
        self.default_unit = default_unit

        if meta is None:
            self.meta = {}
        elif isinstance(meta, dict):
            self.meta = meta
        else:
            raise TypeError("metadata must be a dictionary")

        return self

    @classmethod
    def from_fits_bintable(cls, bintable, tolerance=0.01):
        """
        Instantiate a single beam from a bintable from a CASA-produced image
        HDU.  The beams in the BinTableHDU will be averaged to form a single
        beam.

        Parameters
        ----------
        bintable : fits.BinTableHDU
            The table data containing the beam information
        tolerance : float
            The fractional tolerance on the beam size to include when averaging
            to a single beam

        Returns
        -------
        beam : Beam
            A new beam object that is the average of the table beams
        """
        from astropy.stats import circmean

        bmaj = bintable.data['BMAJ']
        bmin = bintable.data['BMIN']
        bpa = bintable.data['BPA']
        if np.any(np.isnan(bmaj) | np.isnan(bmin) | np.isnan(bpa)):
            raise ValueError("NaN beam encountered.")
        for par in (bmin,bmaj):
            par_mean = par.mean()
            if (par.max() > par_mean*(1+tolerance)) or (par.min()<par_mean*(1-tolerance)):
                raise ValueError("Beams are not within specified tolerance")

        meta = {key: bintable.data[key].mean() for key in bintable.data.names if
                key not in ('BMAJ','BPA', 'BMIN')}
        if meta:
            warnings.warn("Metadata was averaged for keywords "
                          "{0}".format(",".join([key for key in meta])))

        return cls(major=bmaj.mean()*u.arcsec, minor=bmin.mean()*u.arcsec,
                   pa=circmean(bpa*u.deg, weights=bmaj/bmin))

    @classmethod
    def from_fits_header(cls, hdr):
        """
        Instantiate the beam from a header. Attempts to extract the
        beam from standard keywords. Failing that, it looks for an
        AIPS-style HISTORY entry.
        """
        # ... given a file try to make a fits header
        # assume a string refers to a filename on disk
        if not isinstance(hdr,fits.Header):
            if isinstance(hdr, six.string_types):
                if hdr.lower().endswith(('.fits', '.fits.gz', '.fit',
                                         '.fit.gz', '.fits.Z', '.fit.Z')):
                    hdr = fits.getheader(hdr)
                else:
                    raise TypeError("Unrecognized extension.")
            else:
                raise TypeError("Header is not a FITS header or a filename")


        # If we find a major axis keyword then we are in keyword
        # mode. Else look to see if there is an AIPS header.
        if "BMAJ" in hdr:
            major = hdr["BMAJ"] * u.deg
        else:
            hist_beam = cls.from_fits_history(hdr)
            if hist_beam is not None:
                return hist_beam
            else:
                raise NoBeamException("No BMAJ found and does not appear to be a CASA/AIPS header.")

        # Fill out the minor axis and position angle if they are
        # present. Else they will default .
        if "BMIN" in hdr:
            minor = hdr["BMIN"] * u.deg
        else:
            minor = None
        if "BPA" in hdr:
            pa = hdr["BPA"] * u.deg
        else:
            pa = None

        return cls(major=major, minor=minor, pa=pa)


    @classmethod
    def from_fits_history(cls, hdr):
        """
        Instantiate the beam from an AIPS header. AIPS holds the beam
        in history. This method of initializing uses the last such
        entry.
        """
        # a line looks like
        # HISTORY AIPS   CLEAN BMAJ=  1.7599E-03 BMIN=  1.5740E-03 BPA=   2.61
        if 'HISTORY' not in hdr:
            return None

        aipsline = None
        for line in hdr['HISTORY']:
            if 'BMAJ' in line:
                aipsline = line

        # a line looks like
        # HISTORY Sat May 10 20:53:11 2014
        # HISTORY imager::clean() [] Fitted beam used in
        # HISTORY > restoration: 1.34841 by 0.830715 (arcsec)
        #        at pa 82.8827 (deg)

        casaline = None
        for line in hdr['HISTORY']:
            if ('restoration' in line) and ('arcsec' in line):
                casaline = line
        #assert precedence for CASA style over AIPS
        #        this is a dubious choice

        if casaline is not None:
            bmaj = float(casaline.split()[2]) * u.arcsec
            bmin = float(casaline.split()[4]) * u.arcsec
            bpa = float(casaline.split()[8]) * u.deg
            return cls(major=bmaj, minor=bmin, pa=bpa)

        elif aipsline is not None:
            bmaj = float(aipsline.split()[3]) * u.deg
            bmin = float(aipsline.split()[5]) * u.deg
            bpa = float(aipsline.split()[7]) * u.deg
            return cls(major=bmaj, minor=bmin, pa=bpa)

        else:
            return None

    @classmethod
    def from_casa_image(cls, imagename):
        '''
        Instantiate beam from a CASA image.

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

        beam_keys = ["major", "minor", "positionangle"]
        if not all([True for key in beam_keys if key in beam_props]):
            raise ValueError("The image does not contain complete beam "
                             "information. Check the output of "
                             "ia.restoringbeam().")

        major = beam_props["major"]["value"] * \
            u.Unit(beam_props["major"]["unit"])
        minor = beam_props["minor"]["value"] * \
            u.Unit(beam_props["minor"]["unit"])
        pa = beam_props["positionangle"]["value"] * \
            u.Unit(beam_props["positionangle"]["unit"])

        return cls(major=major, minor=minor, pa=pa)

    def attach_to_header(self, header, copy=True):
        '''
        Attach the beam information to the provided header.

        Parameters
        ----------
        header : astropy.io.fits.header.Header
            Header to add/update beam info.
        copy : bool, optional
            Returns a copy of the inputted header with the beam information.

        Returns
        -------
        copy_header : astropy.io.fits.header.Header
            Copy of the input header with the updated beam info when
            `copy=True`.
        '''

        if copy:
            header = header.copy()

        header.update(self.to_header_keywords())

        return header

    def __repr__(self):
        return "Beam: BMAJ={0} BMIN={1} BPA={2}".format(self.major.to(self.default_unit),self.minor.to(self.default_unit),self.pa.to(u.deg))

    def __repr_html__(self):
        return "Beam: BMAJ={0} BMIN={1} BPA={2}".format(self.major.to(self.default_unit),self.minor.to(self.default_unit),self.pa.to(u.deg))

    def _repr_latex_(self):
        return "Beam: BMAJ=${0}^{{{fmt}}}$ BMIN=${1}^{{{fmt}}}$ BPA=${2}^\\circ$".format(self.major.to(self.default_unit).value,
                                                                                         self.minor.to(self.default_unit).value,
                                                                                         self.pa.to(u.deg).value,
                                                                                         fmt = unit_format[self.default_unit])

    def __str__(self):
        return self.__repr__()


    def convolve(self, other):
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

        new_major, new_minor, new_pa = convolve(self, other)

        return Beam(major=new_major,
                    minor=new_minor,
                    pa=new_pa)

    def __mul__(self, other):
        return self.convolve(other)

    # Does division do the same? Or what? Doesn't have to be defined.
    def __sub__(self, other):
        warnings.warn("Subtraction-as-deconvolution is deprecated. "
                      "Use division instead.",
                      RadioBeamDeprecationWarning)
        return self.deconvolve(other)

    def __truediv__(self, other):
        return self.deconvolve(other)

    def deconvolve(self, other, failure_returns_pointlike=False):
        """
        Deconvolve a beam from another

        Parameters
        ----------
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

        new_major, new_minor, new_pa = \
            deconvolve(self, other,
                       failure_returns_pointlike=failure_returns_pointlike)
        return Beam(major=new_major, minor=new_minor, pa=new_pa)

    def __eq__(self, other):

        # Catch floating point issues
        atol_deg = 1e-12 * u.deg

        this_pa = self.pa.to(u.deg) % (180.0 * u.deg)
        other_pa = other.pa.to(u.deg) % (180.0 * u.deg)

        if self.iscircular():
            equal_pa = True
        else:
            equal_pa = True if np.abs(this_pa - other_pa) < atol_deg else False

        equal_maj = np.abs(self.major - other.major) < atol_deg
        equal_min = np.abs(self.minor - other.minor) < atol_deg

        if equal_maj and equal_min and equal_pa:
            return True
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    # Is it astropy convention to access properties through methods?
    @property
    def sr(self):
        return _to_area(self.major,self.minor)

    @property
    def major(self):
        """ Beam FWHM Major Axis """
        return self._major

    @property
    def minor(self):
        """ Beam FWHM Minor Axis """
        return self._minor

    @property
    def pa(self):
        return self._pa

    @property
    def isfinite(self):
        return ((self.major > 0) & (self.minor > 0) & np.isfinite(self.major) &
                np.isfinite(self.minor) & np.isfinite(self.pa))

    def iscircular(self, rtol=1e-6):

        frac_diff = (self.major - self.minor).to(u.deg) / self.major.to(u.deg)

        return frac_diff <= rtol

    def beam_projected_area(self, distance):
        """
        Return the beam area in pc^2 (or equivalent) given a distance
        """
        return self.sr*(distance**2)/u.sr

    def jtok_equiv(self, freq):
        '''
        Return conversion function between Jy/beam to K at the specified
        frequency.

        The function can be used with the usual astropy.units conversion:
        >>> beam = Beam.from_fits_header("header.fits") # doctest: +SKIP
        >>> (1.0*u.Jy).to(u.K, beam.jtok_equiv(1.4*u.GHz)) # doctest: +SKIP

        Parameters
        ----------
        freq : astropy.units.quantity.Quantity
            Frequency to calculate conversion.

        Returns
        -------
        u.brightness_temperature
        '''

        if not isinstance(freq, u.quantity.Quantity):
            raise TypeError("freq must be a Quantity object. "
                            "Try 'freq*u.Hz' or another equivalent unit.")

        try:
            return u.brightness_temperature(beam_area=self.sr, frequency=freq)
        except TypeError:
            # old astropy used ordered arguments
            return u.brightness_temperature(self.sr, freq)

    def jtok(self, freq, value=1.0*u.Jy):
        """
        Return the conversion for the given value between Jy/beam to K at
        the specified frequency.

        Unlike :meth:`jtok_equiv`, the output is the numerical value that
        converts the units, without any attached unit.

        Parameters
        ----------
        freq : astropy.units.quantity.Quantity
            Frequency to calculate conversion.
        value : astropy.units.quantity.Quantity
            Value (in Jy or an equivalent unit) to convert to K.

        Returns
        -------
        value : float
            Value converted to K.
        """

        return value.to(u.K, self.jtok_equiv(freq))

    def ellipse_to_plot(self, xcen, ycen, pixscale):
        """
        Return a matplotlib ellipse for plotting

        Parameters
        ----------
        xcen : int
            Center pixel in the x-direction.
        ycen : int
            Center pixel in the y-direction.
        pixscale : `~astropy.units.Quantity`
            Conversion from degrees to pixels.

        Returns
        -------
        ~matplotlib.patches.Ellipse
            Ellipse patch object centered on the given pixel coordinates.
        """
        from matplotlib.patches import Ellipse
        return Ellipse((xcen, ycen),
                       width=(self.major.to(u.deg) / pixscale).to(u.dimensionless_unscaled).value,
                       height=(self.minor.to(u.deg) / pixscale).to(u.dimensionless_unscaled).value,
                       # PA is 90 deg offset from x-y axes by convention
                       # (it is angle from NCP)
                       angle=(self.pa+90*u.deg).to(u.deg).value)

    def as_kernel(self, pixscale, **kwargs):
        """
        Returns an elliptical Gaussian kernel of the beam.

        .. warning::
            This method is not aware of any misalignment between pixel
            and world coordinates.

        Parameters
        ----------
        pixscale : `~astropy.units.Quantity`
            Conversion from angular to pixel size.
        kwargs : passed to EllipticalGaussian2DKernel
        """
        # do something here involving matrices
        # need to rotate the kernel into the wcs pixel space, kinda...
        # at the least, need to rescale the kernel axes into pixels

        stddev_maj = (self.major.to(u.deg)/(pixscale.to(u.deg) *
                                            SIGMA_TO_FWHM)).decompose()
        stddev_min = (self.minor.to(u.deg)/(pixscale.to(u.deg) *
                                            SIGMA_TO_FWHM)).decompose()

        # position angle is defined as CCW from north
        # "angle" is conventionally defined as CCW from "west".
        # Therefore, add 90 degrees
        angle = (90*u.deg+self.pa).to(u.radian).value,

        return EllipticalGaussian2DKernel(stddev_maj.value,
                                          stddev_min.value,
                                          angle,
                                          **kwargs)

    def as_tophat_kernel(self, pixscale, **kwargs):
        '''
        Returns an elliptical Tophat kernel of the beam. The area has
        been scaled to match the 2D Gaussian area:

        .. math::
            \\begin{array}{ll}
            A_{\\mathrm{Gauss}} = 2\\pi\\sigma_{\\mathrm{Gauss}}^{2}
            A_{\\mathrm{Tophat}} = \\pi\\sigma_{\\mathrm{Tophat}}^{2}
            \\sigma_{\\mathrm{Tophat}} = \\sqrt{2}\\sigma_{\\mathrm{Gauss}}
            \\end{array}

        .. warning::
            This method is not aware of any misalignment between pixel
            and world coordinates.

        Parameters
        ----------
        pixscale : float
            deg -> pixels
        **kwargs : passed to EllipticalTophat2DKernel
        '''

        # Based on Gaussian to Tophat area conversion
        # A_gaussian = 2 * pi * sigma^2 / (sqrt(8*log(2))^2
        # A_tophat = pi * r^2
        # pi r^2 = 2 * pi * sigma^2 / (sqrt(8*log(2))^2
        # r = sqrt(2)/sqrt(8*log(2)) * sigma

        gauss_to_top = np.sqrt(2)

        maj_eff = gauss_to_top * self.major.to(u.deg) / \
            (pixscale * SIGMA_TO_FWHM)
        min_eff = gauss_to_top * self.minor.to(u.deg) / \
            (pixscale * SIGMA_TO_FWHM)

        # position angle is defined as CCW from north
        # "angle" is conventionally defined as CCW from "west".
        # Therefore, add 90 degrees
        angle = (90*u.deg+self.pa).to(u.radian).value,

        return EllipticalTophat2DKernel(maj_eff.value, min_eff.value,
                                        angle, **kwargs)

    def to_header_keywords(self):
        return {'BMAJ': self.major.to(u.deg).value,
                'BMIN': self.minor.to(u.deg).value,
                'BPA':  self.pa.to(u.deg).value,
                }

# Beam.__doc__ = Beam.__doc__ + Beam.__new__.__doc__

def mywcs_to_platescale(mywcs):
    pix_area = wcs.utils.proj_plane_pixel_area(mywcs)
    return pix_area**0.5


class EllipticalGaussian2DKernel(Kernel2D):
    """
    2D Elliptical Gaussian filter kernel.

    The Gaussian filter is a filter with great smoothing properties. It is
    isotropic and does not produce artifacts.

    Parameters
    ----------
    stddev_maj : float
        Standard deviation of the Gaussian kernel in direction 1
    stddev_min : float
        Standard deviation of the Gaussian kernel in direction 1
    position_angle : float
        Position angle of the elliptical gaussian
    x_size : odd int, optional
        Size in x direction of the kernel array. Default = support_scaling *
        stddev.
    y_size : odd int, optional
        Size in y direction of the kernel array. Default = support_scaling *
        stddev.
    support_scaling : int
        The amount to scale the stddev to determine the size of the kernel
    mode : str, optional
        One of the following discretization modes:
            * 'center' (default)
                Discretize model by taking the value
                at the center of the bin.
            * 'linear_interp'
                Discretize model by performing a bilinear interpolation
                between the values at the corners of the bin.
            * 'oversample'
                Discretize model by taking the average
                on an oversampled grid.
            * 'integrate'
                Discretize model by integrating the
                model over the bin.
    factor : number, optional
        Factor of oversampling. Default factor = 10.


    See Also
    --------
    Box2DKernel, Tophat2DKernel, MexicanHat2DKernel, Ring2DKernel,
    TrapezoidDisk2DKernel, AiryDisk2DKernel, Gaussian2DKernel,
    EllipticalTophat2DKernel

    Examples
    --------
    Kernel response:

     .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from radio_beam import EllipticalGaussian2DKernel
        gaussian_2D_kernel = EllipticalGaussian2DKernel(10, 5, np.pi/4)
        plt.imshow(gaussian_2D_kernel, interpolation='none', origin='lower')
        plt.xlabel('x [pixels]')
        plt.ylabel('y [pixels]')
        plt.colorbar()
        plt.show()

    """
    _separable = True
    _is_bool = False

    def __init__(self, stddev_maj, stddev_min, position_angle,
                 support_scaling=8, **kwargs):
        self._model = Gaussian2D(1. / (2 * np.pi * stddev_maj * stddev_min), 0,
                                 0, x_stddev=stddev_maj, y_stddev=stddev_min,
                                 theta=position_angle)

        try:
            from astropy.modeling.utils import ellipse_extent
        except ImportError:
            raise NotImplementedError("EllipticalGaussian2DKernel requires"
                                      " astropy 1.1b1 or greater.")

        max_extent = \
            np.max(ellipse_extent(stddev_maj, stddev_min, position_angle))
        self._default_size = \
            _round_up_to_odd_integer(support_scaling * 2 * max_extent)
        super(EllipticalGaussian2DKernel, self).__init__(**kwargs)
        self._truncation = np.abs(1. - 1 / self._array.sum())


class EllipticalTophat2DKernel(Kernel2D):
    """
    2D Elliptical Tophat filter kernel.

    The Tophat filter can produce artifacts when applied
    repeatedly on the same data.

    Parameters
    ----------
    stddev_maj : float
        Standard deviation of the Gaussian kernel in direction 1
    stddev_min : float
        Standard deviation of the Gaussian kernel in direction 1
    position_angle : float
        Position angle of the elliptical gaussian
    x_size : odd int, optional
        Size in x direction of the kernel array. Default = support_scaling *
        stddev.
    y_size : odd int, optional
        Size in y direction of the kernel array. Default = support_scaling *
        stddev.
    support_scaling : int
        The amount to scale the stddev to determine the size of the kernel
    mode : str, optional
        One of the following discretization modes:
            * 'center' (default)
                Discretize model by taking the value
                at the center of the bin.
            * 'linear_interp'
                Discretize model by performing a bilinear interpolation
                between the values at the corners of the bin.
            * 'oversample'
                Discretize model by taking the average
                on an oversampled grid.
            * 'integrate'
                Discretize model by integrating the
                model over the bin.
    factor : number, optional
        Factor of oversampling. Default factor = 10.


    See Also
    --------
    Box2DKernel, Tophat2DKernel, MexicanHat2DKernel, Ring2DKernel,
    TrapezoidDisk2DKernel, AiryDisk2DKernel, Gaussian2DKernel,
    EllipticalGaussian2DKernel

    Examples
    --------
    Kernel response:

     .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from radio_beam import EllipticalTophat2DKernel
        tophat_2D_kernel = EllipticalTophat2DKernel(10, 5, np.pi/4)
        plt.imshow(tophat_2D_kernel, interpolation='none', origin='lower')
        plt.xlabel('x [pixels]')
        plt.ylabel('y [pixels]')
        plt.colorbar()
        plt.show()

    """

    _is_bool = True

    def __init__(self, stddev_maj, stddev_min, position_angle, support_scaling=1,
                 **kwargs):

        self._model = Ellipse2D(1. / (np.pi * stddev_maj * stddev_min), 0, 0,
                                stddev_maj, stddev_min, position_angle)

        try:
            from astropy.modeling.utils import ellipse_extent
        except ImportError:
            raise NotImplementedError("EllipticalTophat2DKernel requires"
                                      " astropy 1.1b1 or greater.")

        max_extent = \
            np.max(ellipse_extent(stddev_maj, stddev_min, position_angle))
        self._default_size = \
            _round_up_to_odd_integer(support_scaling * 2 * max_extent)
        super(EllipticalTophat2DKernel, self).__init__(**kwargs)
        self._truncation = 0
