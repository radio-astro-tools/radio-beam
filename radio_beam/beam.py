from astropy import units as u
from astropy.io import fits
from astropy import constants
import astropy.units as u
#from astropy import wcs
from astropy.extern import six
import numpy as np
import warnings

# Conversion between a twod Gaussian FWHM**2 and effective area
FWHM_TO_AREA = 2*np.pi/(8*np.log(2))

def _to_area(major,minor):
    return (major * minor * FWHM_TO_AREA).to(u.sr)

unit_format = {u.deg: '\\circ',
               u.arcsec: "''",
               u.arcmin: "'"}

class Beam(u.Quantity):
    """
    An object to handle radio beams.
    """

    def __new__(cls, major=None, minor=None, pa=None, area=None,
                default_unit=u.arcsec):
        """

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

        # ... given an area make a round beam
        if area is not None:
            rad = np.sqrt(area/np.pi) * u.deg
            major = rad
            minor = rad
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

        self = super(Beam, cls).__new__(cls, _to_area(major,minor).value, u.sr)
        self._major = major
        self._minor = minor
        self._pa = pa
        self.default_unit = default_unit

        return self


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
            aips_beam = cls.from_aips_header(hdr)
            if aips_beam is None:
                raise TypeError("No BMAJ found and does not appear to be an AIPS header.")
            else:
                return aips_beam

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
    def from_aips_header(cls, hdr):
        """
        Instantiate the beam from an AIPS header. AIPS holds the beam
        in history. This method of initializing uses the last such
        entry.
        """
        # a line looks like
        # HISTORY AIPS   CLEAN BMAJ=  1.7599E-03 BMIN=  1.5740E-03 BPA=   2.61
        aipsline = None
        for line in hdr['HISTORY']:
            if 'BMAJ' in line:
                aipsline = line

        if aipsline is not None:
            bmaj = float(aipsline.split()[3]) * u.deg
            bmin = float(aipsline.split()[5]) * u.deg
            bpa = float(aipsline.split()[7]) * u.deg
            return cls(major=bmaj, minor=bmin, pa=bpa)
        else:
            return None

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

        # blame: https://github.com/pkgw/carma-miriad/blob/CVSHEAD/src/subs/gaupar.for
        # (githup checkin of MIRIAD, code by Sault)

        alpha = ((self.major*np.cos(self.pa))**2 +
                 (self.minor*np.sin(self.pa))**2 +
                 (other.major*np.cos(other.pa))**2 +
                 (other.minor*np.sin(other.pa))**2)

        beta = ((self.major*np.sin(self.pa))**2 +
                (self.minor*np.cos(self.pa))**2 +
                (other.major*np.sin(other.pa))**2 +
                (other.minor*np.cos(other.pa))**2)

        gamma = (2*((self.minor**2-self.major**2) *
                    np.sin(self.pa)*np.cos(self.pa) +
                    (other.minor**2-other.major**2) *
                    np.sin(other.pa)*np.cos(other.pa)))

        s = alpha + beta
        t = np.sqrt((alpha-beta)**2 + gamma**2)

        new_major = np.sqrt(0.5*(s+t))
        new_minor = np.sqrt(0.5*(s-t))
        if (abs(gamma)+abs(alpha-beta)) == 0:
            new_pa = 0.0 * u.deg
        else:
            new_pa = 0.5*np.arctan2(-1.*gamma, alpha-beta)

        return Beam(major=new_major,
                    minor=new_minor,
                    pa=new_pa)

    def __mult__(self, other):
        return self.convolve(other)

    # Does division do the same? Or what? Doesn't have to be defined.
    def __sub__(self, other):
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

        # blame: https://github.com/pkgw/carma-miriad/blob/CVSHEAD/src/subs/gaupar.for
        # (githup checkin of MIRIAD, code by Sault)

        # rename to shorter variables for readability
        maj1,min1,pa1 = self.major,self.minor,self.pa
        maj2,min2,pa2 = other.major,other.minor,other.pa
        cos,sin = np.cos,np.sin

        alpha = ((maj1*cos(pa1))**2 +
                 (min1*sin(pa1))**2 -
                 (maj2*cos(pa2))**2 -
                 (min2*sin(pa2))**2)

        beta = ((maj1*sin(pa1))**2 +
                (min1*cos(pa1))**2 -
                (maj2*sin(pa2))**2 -
                (min2*cos(pa2))**2)

        gamma = 2 * ((min1**2 - maj1**2) * sin(pa1)*cos(pa1) -
                     (min2**2 - maj2**2) * sin(pa2)*cos(pa2))

        s = alpha + beta
        t = np.sqrt((alpha-beta)**2 + gamma**2)

        # identify the smallest resolution
        axes = np.array([maj1.to(u.deg).value,
                         min1.to(u.deg).value,
                         maj2.to(u.deg).value,
                         min2.to(u.deg).value])*u.deg
        limit = np.min(axes)
        limit = 0.1*limit*limit

        if (alpha < 0) or (beta < 0) or (s < t):
            if failure_returns_pointlike:
                return Beam(major=0, minor=0, pa=0)
            else:
                raise ValueError("Beam could not be deconvolved")
        else:
            new_major = np.sqrt(0.5*(s+t))
            new_minor = np.sqrt(0.5*(s-t))

            if (abs(gamma)+abs(alpha-beta)) == 0:
                new_pa = 0.0
            else:
                new_pa = 0.5*np.arctan2(-1.*gamma, alpha-beta)

        return Beam(major=new_major,
                    minor=new_minor,
                    pa=new_pa)

    def __eq__(self, other):
        if ((self.major == other.major) and
            (self.minor == other.minor) and
            (self.pa == other.pa)):
            return True
        else:
            return False

    # Is it astropy convention to access properties through methods?
    @property
    def sr(self):
        return _to_area(self.major,self.minor)

    @property
    def major(self):
        return self._major

    @property
    def minor(self):
        return self._minor

    @property
    def pa(self):
        return self._pa

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
        >>> (1.0*u.Jy).to(u.K, self.jtok_equiv(1.4*u.GHz))

        Parameters
        ----------
        freq : astropy.units.quantity.Quantity
            Frequency to calculate conversion.

        Returns
        -------

        u.brightness_temperature

        '''
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

        c = (constants.c.cgs).value
        kb = (constants.k_B.cgs).value

        if u.hertz.is_equivalent(freq):
            freq = freq
        else:
            warnings.warn("Assuming frequency has been specified in Hz")
            freq = freq * u.hertz

        return c**2/self.sr.value/1e23/(2*kb*(freq.to(u.hertz).value)**2)

    def ellipse_to_plot(self, xcen, ycen, units=u.deg, wcs=None):
        """
        Return a matplotlib ellipse for plotting

        .. todo::
            Implement this!
        """
        import matplotlib
        return matplotlib.patches.Ellipse((xcen,ycen),
                                          width=self.major.to(u.deg).value/pixscale,
                                          height=self.minor.to(u.deg).value/pixscale,
                                          angle=self.pa.to(u.deg).value)

    def as_kernel(self, pixscale):
        """
        Parameters
        ----------
        pixscale : float
            deg -> pixels

        """
        # do something here involving matrices
        # need to rotate the kernel into the wcs pixel space, kinda...
        # at the least, need to rescale the kernel axes into pixels
        warnings.warn("as_kernel is not aware of any misaligment between pixel "
                      "and world coordinates")

        return EllipticalGaussian2DKernel(self.major.to(u.deg).value/pixscale,
                                          self.minor.to(u.deg).value/pixscale,
                                          self.pa.to(u.radian).value)

    def to_header_keywords(self):
        return {'BMAJ': self.major.to(u.deg).value,
                'BMIN': self.major.to(u.deg).value,
                'BPA':  self.pa.to(u.deg).value,
               }


def wcs_to_platescale(wcs):
    cdelt = np.matrix(wcs.get_cdelt())
    pc = np.matrix(wcs.get_pc())
    scale = np.array(cdelt * pc)[0,:]
    # this may be wrong in perverse cases
    pixscale = np.abs(scale[0])
    return pixscale

from astropy.modeling import models
from astropy.convolution import Kernel2D
from astropy.convolution.kernels import _round_up_to_odd_integer

class EllipticalGaussian2DKernel(Kernel2D):
    """
    2D Elliptical Gaussian filter kernel.

    The Gaussian filter is a filter with great smoothing properties. It is
    isotropic and does not produce artifacts.

    Parameters
    ----------
    width : float
        Standard deviation of the Gaussian kernel in direction 1
    height : float
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
    TrapezoidDisk2DKernel, AiryDisk2DKernel, Gaussian2DKernel

    Examples
    --------
    Kernel response:

     .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from beam import EllipticalGaussian2DKernel
        gaussian_2D_kernel = EllipticalGaussian2DKernel(10)
        plt.imshow(gaussian_2D_kernel, interpolation='none', origin='lower')
        plt.xlabel('x [pixels]')
        plt.ylabel('y [pixels]')
        plt.colorbar()
        plt.show()

    """
    _separable = True
    _is_bool = False

    def __init__(self, width, height, position_angle, support_scaling=8, **kwargs):
        self._model = models.Gaussian2D(1. / (2 * np.pi * width * height), 0,
                                        0, x_stddev=width, y_stddev=height,
                                        theta=position_angle)
        self._default_size = _round_up_to_odd_integer(support_scaling *
                                                      np.max([width,height]))
        super(EllipticalGaussian2DKernel, self).__init__(**kwargs)
        self._truncation = np.abs(1. - 1 / self._normalization)
