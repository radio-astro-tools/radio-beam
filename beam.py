# Astropy required or not?
from astropy import units as u
from astropy.io import fits
import numpy as np
import warnings

FWHM_TO_AREA = 2*np.pi*(8*np.log(2))

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


    @classmethod
    def from_aips_header(cls, hdr):
        """
        Extract the beam from an old AIPS header. Returns true if
        successful?
        """
        # a line looks like
        # HISTORY AIPS   CLEAN BMAJ=  1.7599E-03 BMIN=  1.5740E-03 BPA=   2.61
        for line in hdr['HISTORY']:
            if 'BMAJ' in line:
                aipsline = line

        bmaj = float(aipsline.split()[3]) * u.deg
        bmin = float(aipsline.split()[5]) * u.deg
        bpa = float(aipsline.split()[7]) * u.deg

        return cls(major=bmaj, minor=bmin, pa=bpa)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Operators
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    def __repr__(self):
        return "Beam: BMAJ={0} BMIN={1} BPA={2}".format(self.major,self.minor,self.pa)

    def __repr_html__(self):
        return "Beam: BMAJ={0} BMIN={1} BPA={2}".format(self.major,self.minor,self.pa)


    def convolve(self, other):
        """
        Addition convolves.
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
            new_pa = 0.5*np.arctan2(-1.*gamma, alpha-beta) * u.rad
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
        pass

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
