Radio Beam
==========

A tool for manipulating and utilizing two dimensional gaussian beams within the
`astropy <http://www.astropy.org>`__ framework.


Examples
--------

Read a beam from a fits header::

    >>> from radio_beam import Beam
    >>> from astropy.io import fits
    >>> header = fits.getheader('file.fits')  # doctest: +SKIP
    >>> my_beam = Beam.from_fits_header(header)  # doctest: +SKIP
    >>> print(my_beam)  # doctest: +SKIP
    Beam: BMAJ=0.038652855902928 arcsec BMIN=0.032841067761183604 arcsec BPA=32.29655838013 deg


Create a beam from scratch::

    >>> from astropy import units as u
    >>> my_beam = Beam(0.5*u.arcsec)


Use a beam for Jy -> K conversion::

    >>> (1*u.Jy).to(u.K, u.brightness_temperature(my_beam, 25*u.GHz))
    <Quantity 7821.571333052632 K>

Convolve with another beam::

    >>> my_asymmetric_beam = Beam(0.75*u.arcsec, 0.25*u.arcsec, 0*u.deg)
    >>> my_other_asymmetric_beam = Beam(0.75*u.arcsec, 0.25*u.arcsec, 90*u.deg)
    >>> my_asymmetric_beam.convolve(my_other_asymmetric_beam)  # doctest: +SKIP
    Beam: BMAJ=0.790569415042 arcsec BMIN=0.790569415042 arcsec BPA=45.0 deg

Deconvolve another beam::

    >>> my_big_beam = Beam(1.0*u.arcsec, 1.0*u.arcsec, 0*u.deg)
    >>> my_little_beam = Beam(0.5*u.arcsec, 0.5*u.arcsec, 0*u.deg)
    >>> my_big_beam.deconvolve(my_little_beam)  # doctest: +SKIP
    Beam: BMAJ=0.866025403784 arcsec BMIN=0.866025403784 arcsec BPA=0.0 deg