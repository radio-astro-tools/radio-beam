from astropy import units as u
from astropy.io import fits
from astropy import constants
import astropy.wcs
import numpy as np
import warnings

# Conversion between a twod Gaussian FWHM**2 and effective area
FWHM_TO_AREA = 2*np.pi/(8*np.log(2))

def _to_area(major,minor):
    return (major * minor * FWHM_TO_AREA).to(u.sr)

class Beam(u.Quantity):
    """
    An object to handle radio beams.
    """
 
    # Attributes
    major = None
    minor = None
    pa = None

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Constructor
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    def __new__(cls, major=None, minor=None, pa=None, area=None, hdr=None,):
        """

        Parameters
        ----------
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
            if type(hdr) == type("hello"):
                if (hdr[-4:]).upper() == "FITS":
                    hdr = fits.getheader(hdr)

        if not isinstance(hdr,fits.Header):
            # right type of error?
            raise TypeError("Header does not appear to be a valid header or a file holding a header.")

        if hdr is not None:

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
            if "BPA" in hdr:
                pa = hdr["BPA"] * u.deg

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

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Operators
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    def __repr__(self):
        return "Beam: BMAJ={0} BMIN={1} BPA={2}".format(self.major,self.minor,self.pa)

    def __repr_html__(self):
        return "Beam: BMAJ={0} BMIN={1} BPA={2}".format(self.major,self.minor,self.pa)

    def _repr_latex_(self):
        return "Beam: BMAJ=${0}^\\circ$ BMIN=${1}^\\circ$ BPA=${2}^\\circ$".format(self.major.to(u.deg).value,
                                                                                   self.minor.to(u.deg).value,
                                                                                   self.pa.to(u.deg).value)


    def convolve(self, other):
        """
        Convolve one beam with another. Returns a new beam
        object. This new beam would be appropriate for 
        """
        
        # Units crap - we're storing PA in degrees, do we need to go to radians?

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
            # units!
        
        # Make a new beam and return it
        return Beam(major=new_major,
                    minor=new_minor,
                    pa=new_pa)

    # Does multiplication do the same? Or what?
    
    def deconvolve(self, other):
        """
        Subtraction deconvolves.
        """
        # math.

        # Units crap - we're storing PA in degrees, do we need to go to radians?

        # blame: https://github.com/pkgw/carma-miriad/blob/CVSHEAD/src/subs/gaupar.for
        # (githup checkin of MIRIAD, code by Sault)

        alpha = ((self.major*np.cos(self.pa))**2 +
                 (self.minor*np.sin(self.pa))**2 -
                 (other.major*np.cos(other.pa))**2 -
                 (other.minor*np.sin(other.pa))**2)

        beta = ((self.major*np.sin(self.pa))**2 +
                (self.minor*np.cos(self.pa))**2 -
                (other.major*np.sin(other.pa))**2 -
                (other.minor*np.cos(other.pa))**2)

        gamma = (2 * (self.minor**2 - self.major**2) *
                 np.sin(self.pa)*np.cos(self.pa) -
                 (other.minor**2 - other.major**2) *
                 np.sin(other.pa)*np.cos(other.pa))

        s = alpha + beta
        t = np.sqrt((alpha-beta)**2 + gamma**2)

        # identify the smallest resolution

        # ... MECHANICAL: How do I do this right?
        axes = np.array([self.major.to(u.deg).value,
                         self.minor.to(u.deg).value,
                         other.major.to(u.deg).value,
                         other.minor.to(u.deg).value])*u.deg
        limit = np.min(axes)
        limit = 0.1*limit*limit
        
        # two cases...

        # ... failure
        if (alpha < 0) or (beta < 0) or (s < t):

            # Note failure as an empty beam
            new_major = 0.0
            new_minor = 0.0
            new_pa = 0.0

            # Record that things failed
            failed = True
            
            # Check if it is close to a point source
            if ((0.5*(s-t) < limit) and
                (alpha > -1*limit) and
                (beta > -1*limit)):
                pointlike = True
            else:
                pointlike = False
        # ... success
        else:
            # save
            new_major = np.sqrt(0.5*(s+t))
            new_minor = np.sqrt(0.5*(s-t))

            if (abs(gamma)+abs(alpha-beta)) == 0:
                new_pa = 0.0
            else:
                new_pa = 0.5*np.arctan2(-1.*gamma, alpha-beta)

            failed = False
            pointlike = False

        # Make a new beam and return it
        return Beam(major=new_major,
                    minor=new_minor,
                    pa=new_pa)

    # Does division do the same? Or what? Doesn't have to be defined.

    def __eq__(self, other):
        """
        Equality operator.
        """

        # Right now it's loose, just check major, minor, pa.

        if self.major != other.major:
            return False

        if self.minor != other.minor:
            return False

        if self.pa != other.pa:
            return False

        return True

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Property Access
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

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

    def jtok(self, freq):
        """
        Return the conversion between janskies per beam and kelvin (in
        Rayleigh Jeans brightness temperature) given a frequency.
        """

        c = (constants.c.cgs).value
        h = (constants.h.cgs).value
        kb = (constants.k_B.cgs).value

        if u.hertz.is_equivalent(freq):
            freq = freq
        else:
            warnings.warn("Assuming frequency has been specified in Hz")
            freq = freq * u.hertz
            
        return c**2/self.sr.value/1e23/(2*kb*(freq.to(u.hertz).value)**2)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Methods
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    def ellipse_to_plot(self, xcen, ycen):
        """
        Return a matplotlib ellipse
        """
        import matplotlib
        raise NotImplementedError("Let's finish this later, k?")
        return matplotlib.Patches.Ellipse(self.major, self.minor, self.pa)

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

        return EllipticalGaussian2DKernel(self.major.to(u.deg).value/pixscale,
                                          self.minor.to(u.deg).value/pixscale,
                                          self.pa.to(u.radian).value)

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
