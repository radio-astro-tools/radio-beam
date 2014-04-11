# Astropy required or not?
from astropy import units as u
from astropy.io import fits
from math import sqrt, pi, cos, sin, abs, atan2, log
import numpy as np
import warnings

fwhm_to_area = 2*pi*(8*log(2))

class Beam(object):
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

    def __init__(self, major=None, minor=None, pa=None, area=None, hdr=None,):
        """

        Parameters
        ----------
        """
        
        # improve to some kwargs magic later

        # error checking

        # ... given an area make a round beam
        if area is not None:
            rad = sqrt(area/pi) * u.deg
            self.major = rad
            self.minor = rad
            self.pa = 0.0
            
                
        # give specified values priority
        if major is not None:
            if u.deg.is_equivalent(major):
                self.major = major
            else:
                warnings.warn("Assuming major axis has been specified in degrees")
                self.major = major * u.deg
        if minor is not None:
            if u.deg.is_equivalent(minor):
                self.minor = minor
            else:
                warnings.warn("Assuming minor axis has been specified in degrees")
                self.minor = minor * u.deg
        if pa is not None:
            if u.deg.is_equivalent(pa):
                self.pa = pa
            else:
                self.pa = pa * u.deg
        else:
            self.pa = 0.0 * u.deg

        # some sensible defaults
        if self.minor is None:
            self.minor = self.major

    @classmethod
    def from_fits_header(cls, hdr):
        # ... given a file try to make a fits header
        # assume a string refers to a filename on disk
        if not isinstance(hdr,fits.Header):
            hdr = fits.getheader(hdr)

        if hdr is not None:
            if "BMAJ" in hdr:
                major = hdr["BMAJ"] * u.deg
            else:
                raise TypeError("No BMAJ found; could be an AIPS header... TODO: look that up")

            if "BMIN" in hdr:
                minor = hdr["BMIN"] * u.deg
            if "BPA" in hdr:
                pa = hdr["BPA"] * u.deg

        return cls(major=major, minor=minor, pa=pa)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Operators
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    def __add__(self, other):
        """
        Addition convolves.
        """
        
        # Units crap - expect in radians

        # blame: https://github.com/pkgw/carma-miriad/blob/CVSHEAD/src/subs/gaupar.for

        alpha = ((self.major*cos(self.pa))**2 +
                 (self.minor*sin(self.pa))**2 +
                 (other.major*cos(other.pa))**2 +
                 (other.minor*sin(other.pa))**2)

        beta = ((self.major*sin(self.pa))**2 +
                (self.minor*cos(self.pa))**2 +
                (other.major*sin(other.pa))**2 +
                (other.minor*cos(other.pa))**2)
        
        gamma = (2*((self.minor**2-self.major**2) *
                    sin(self.pa)*cos(self.pa) +
                    (other.minor**2-other.major**2) *
                    sin(other.pa)*cos(other.pa)))

        s = alpha + beta
        t = sqrt((alpha-beta)**2 + gamma**2)

        new_major = sqrt(0.5*(s+t))
        new_minor = sqrt(0.5*(s-t))
        if (abs(gamma)+abs(alpha-beta)) == 0:
            new_pa = 0.0
        else:
            new_pa = 0.5*atan2(-1.*gamma, alpha-beta)
            # units!
        
        # Make a new beam and return it
        return Beam(major=new_major,
                    minor=new_minor,
                    pa=new_pa)

    # Does multiplication do the same? Or what?
    
    def __sub__(self, other):
        """
        Subtraction deconvolves.
        """
        # math.
        pass

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
        return self.major * self.minor * fwhm_to_area

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Methods
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    
