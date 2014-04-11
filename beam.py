# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# IMPORTS
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# Astropy required or not?
from astropy.io import fits
from math import sqrt, pi, cos, sin, abs, atan2
import numpy as np

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# OBJECT
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class Beam(object):
    """ 
    An object to handle radio beams.
    """

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Attributes
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=    

    major = None
    minor = None
    pa = None    

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Constructor
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    def __init__(
        self,
        major=None,
        minor=None,
        pa = None,
        area = None,
        fname = None,
        hdr = None,
        ):
        """
        """
        
        # improve to some kwargs magic later

        # error checking

        # ... given an area make a round beam        
        if area != None:
            rad = sqrt(area/pi)
            self.major = rad
            self.minor = rad
            self.pa = 0.0
            return
            
        # ... given a file try to make a fits header
        if fname != None:
            hdr = fits.getheader(fname)

        if hdr != None:
            if hdr.has_key("BMAJ"):
                self.major = hdr["BMAJ"]
            else:
                if self.from_aips_header(hdr) == False:
                    print "No keyword BMAJ or AIPS convention."
                    # ... exit with blank object
                    return

            if hdr.has_key("BMIN"):
                self.minor = hdr["BMIN"]
            if hdr.has_key("BPA"):
                self.pa = hdr["BPA"]
            # ... logic for case of no keyword
            # ... get AIPS?
                
        # give specified values priority
        if major != None:
            self.major = major
        if minor != None:
            self.minor = minor
        if pa != None:
            self.pa = pa

        # some sensible defaults
        if self.minor == None:
            self.minor = self.major

        if self.pa == None:
            self.pa = 0.0

        pass

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Functions to get the beam
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    def from_aips_header(
        self,
        hdr):
        """
        Extract the beam from an old AIPS header. Returns true if
        successful?
        """

        # logic to crawl the history here

        # a line looks like 
        # HISTORY AIPS   CLEAN BMAJ=  1.7599E-03 BMIN=  1.5740E-03 BPA=   2.61

        # multiple cleans can be in there, so you need to work
        # BACKWARDS to get the last line that has this.

        return False

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Operators
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    def __add__(
        self,
        other):
        """
        Addition convolves.
        """
        
        # Units crap - expect in radians

        # blame: https://github.com/pkgw/carma-miriad/blob/CVSHEAD/src/subs/gaupar.for

        alpha = (self.major*cos(self.pa))**2 + \
            (self.minor*sin(self.pa))**2 + \
            (other.major*cos(other.pa))**2 + \
            (other.minor*sin(other.pa))**2

        beta = (self.major*sin(self.pa))**2 + \
            (self.minor*cos(self.pa))**2 + \
            (other.major*sin(other.pa))**2 + \
            (other.minor*cos(other.pa))**2
	
	gamma = 2*((self.minor**2-self.major**2)* \
                       sin(self.pa)*cos(self.pa) + \
                       (other.minor**2-other.major**2)* \
                       sin(other.pa)*cos(other.pa))

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
    
    def __sub__(
        self,
        other):
        """
        Addition deconvolves.
        """
        # math.
        pass

    # Does division do the same? Or what? Doesn't have to be defined.

    def __eq__(
        self,
        other):
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

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Methods
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    
