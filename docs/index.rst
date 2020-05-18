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

    >>> (1*u.Jy).to(u.K, u.brightness_temperature(25*u.GHz, my_beam)) # doctest: +FLOAT_CMP
    <Quantity 7821.572919292681 K>

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

Read a table of beams::

    >>> from radio_beam import Beams
    >>> from astropy.io import fits
    >>> bin_hdu = fits.open('file.fits')[1]  # doctest: +SKIP
    >>> beams = Beams.from_fits_bintable(bin_hdu)  # doctest: +SKIP

Create a table of beams::

    >>> my_beams = Beams([1.5, 1.3] * u.arcsec, [1., 1.2] * u.arcsec, [0, 50] * u.deg)

Find the largest beam in the set::

    >>> my_beams.largest_beam()
    Beam: BMAJ=1.3 arcsec BMIN=1.2 arcsec BPA=50.0 deg

Find the smallest common beam for the set (see :ref:`here <com_beam>` for more on common beams)::

    >>> my_beams.common_beam()  # doctest: +SKIP
    Beam: BMAJ=1.50671729431 arcsec BMIN=1.25695643792 arcsec BPA=6.69089813778 deg

Getting started
^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 2

   install.rst
   commonbeam.rst
   convolution_kernels.rst
   api.rst
