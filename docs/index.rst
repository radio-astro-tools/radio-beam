Radio Beam
==========

radio-beam provides a tools for manipulating and utilizing two-dimensional Gaussian beams
within the `astropy <http://www.astropy.org>`__ framework. It is primarily built for handling
radio astronomy data and is integrated into the `spectral-cube <https://spectral-cube.readthedocs.io>_`
package, amongst others.

radio-beam also handles operations on sets of beams, for example from a spectral cube with
varying resolution in spectral channels. Of note are the algorithms for identifying the
smallest common beam in a set (i.e., the minimum enclosing ellipse area).

Getting started
^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 2

   install.rst
   commonbeam.rst
   plotting_beams.rst
   convolution_kernels.rst
   api.rst


Basic Examples
^^^^^^^^^^^^^^

Handling a single beam
----------------------

`~radio_beam.Beam` handles operations on individual beams.

Read a beam from a FITS header::

    >>> from radio_beam import Beam
    >>> from astropy.io import fits
    >>> header = fits.getheader('file.fits')  # doctest: +SKIP
    >>> my_beam = Beam.from_fits_header(header)  # doctest: +SKIP
    >>> print(my_beam)  # doctest: +SKIP
    Beam: BMAJ=0.038652855902928 arcsec BMIN=0.032841067761183604 arcsec BPA=32.29655838013 deg

To add the beam parameters to a FITS header::

    >>> header.update(my_beam.to_header_keywords())  # doctest: +SKIP

This will return new or add values for the `BMAJ`, `BMIN`, and `BPA` keywords.

Create a new circular beam::

    >>> from astropy import units as u
    >>> my_beam = Beam(0.5*u.arcsec)
    >>> my_beam
    Beam: BMAJ=0.5 arcsec BMIN=0.5 arcsec BPA=0.0 deg

`~radio_beam.Beam` assumes a circular beam when a minor full-width-half-max (FWHM) is not given.
To create an elliptical beam, the minor FWHM and position angle need to be given::

    >>> my_beam_ellip = Beam(major=0.5*u.arcsec, minor=0.25*u.arcsec, pa=30*u.deg)
    >>> my_beam_ellip
    Beam: BMAJ=0.5 arcsec BMIN=0.25 arcsec BPA=30.0 deg

The beam area in steradians is::

    >>> my_beam_ellip.sr # doctest: +FLOAT_CMP
    <Quantity 3.3290795e-12 sr>

Or projected into physical units given a distance::

    >>> my_beam_ellip.beam_projected_area(840*u.kpc).to(u.pc**2) # doctest: +FLOAT_CMP
    <Quantity 2.3489985 pc2>

A common unit conversion in radio astronomy is Jy/beam to K, which depends on the beam area.
A `~radio_beam.Beam` object can be used for this unit conversion with `~astropy.units`::

    >>> (1*u.Jy).to(u.K, u.brightness_temperature(25*u.GHz, my_beam)) # doctest: +FLOAT_CMP
    <Quantity 7821.572919292681 K>

Or alternatively with::

    >>> (1*u.Jy).to(u.K, my_beam.jtok_equiv(25*u.GHz)) # doctest: +FLOAT_CMP
    <Quantity 7821.572919292681 K>

To get the value of 1 Jy in K for a given beam::

    >>> my_beam.jtok(25*u.GHz) # doctest: +FLOAT_CMP
    <Quantity 7821.572919292681 K>

Two beams can be convolved::

    >>> my_asymmetric_beam = Beam(0.75*u.arcsec, 0.25*u.arcsec, 0*u.deg)
    >>> my_other_asymmetric_beam = Beam(0.75*u.arcsec, 0.25*u.arcsec, 90*u.deg)
    >>> my_asymmetric_beam.convolve(my_other_asymmetric_beam) # doctest: +FLOAT_CMP
    Beam: BMAJ=0.790569415042 arcsec BMIN=0.790569415042 arcsec BPA=45.0 deg

And also deconvolved::

    >>> my_big_beam = Beam(1.0*u.arcsec, 1.0*u.arcsec, 0*u.deg)
    >>> my_little_beam = Beam(0.5*u.arcsec, 0.5*u.arcsec, 0*u.deg)
    >>> my_big_beam.deconvolve(my_little_beam) # doctest: +FLOAT_CMP
    Beam: BMAJ=0.866025403784 arcsec BMIN=0.866025403784 arcsec BPA=0.0 deg

An error is returned if the beam area is too small to deconvolve from the other::

    >>> my_little_beam.deconvolve(my_big_beam)  # doctest: +SKIP

To find the smallest common beam between any two beams::

    >>> my_asymmetric_beam.commonbeam_with(my_other_asymmetric_beam) # doctest: +FLOAT_CMP
    Beam: BMAJ=0.75 arcsec BMIN=0.75 arcsec BPA=90.0 deg

Handling a sets of beams
------------------------

`~radio_beam.Beams` handles operations on sets of beams.

To read a table of beams from a FITS table::

    >>> from radio_beam import Beams
    >>> from astropy.io import fits
    >>> bin_hdu = fits.open('file.fits')[1]  # doctest: +SKIP
    >>> beams = Beams.from_fits_bintable(bin_hdu)  # doctest: +SKIP

In the above example, the second FITS extension contains the beam tables, while the
first would have the spectral cube data.

To read a table of beams from a CASA image (must be run inside a CASA environment!)::

    >>> beams = Beams.from_casa_image('file.image')  # doctest: +SKIP

Create a table of beams::

    >>> my_beams = Beams([1.5, 1.3] * u.arcsec, [1., 1.2] * u.arcsec, [0, 50] * u.deg)

`~radio_beam.Beams` acts like a numpy array and can be sliced in the same way::

    >>> my_beams[0]
    Beam: BMAJ=1.5 arcsec BMIN=1.0 arcsec BPA=0.0 deg
    >>> my_beams[1]
    Beam: BMAJ=1.3 arcsec BMIN=1.2 arcsec BPA=50.0 deg

Find the largest beam in the set::

    >>> my_beams.largest_beam()
    Beam: BMAJ=1.3 arcsec BMIN=1.2 arcsec BPA=50.0 deg

Find the smallest common beam for the set (see :ref:`here <com_beam>` for more on common beams)::

    >>> my_beams.common_beam() # doctest: +FLOAT_CMP
    Beam: BMAJ=1.50671729431 arcsec BMIN=1.25695643792 arcsec BPA=6.69089813778 deg

Return the smallest and largest beams in a set (by beam area)::

    >>> smallest_beam, largest_beam = my_beams.extrema_beams()
    >>> smallest_beam
    Beam: BMAJ=1.5 arcsec BMIN=1.0 arcsec BPA=0.0 deg
    >>> largest_beam
    Beam: BMAJ=1.3 arcsec BMIN=1.2 arcsec BPA=50.0 deg

Optionally mask out a beam (to exclude from the calculation)::

    >>> import numpy as np
    >>> beam_mask = np.array([True, False])
    >>> smallest_beam, largest_beam = my_beams.extrema_beams(includemask=beam_mask)
    >>> smallest_beam
    Beam: BMAJ=1.5 arcsec BMIN=1.0 arcsec BPA=0.0 deg
    >>> largest_beam
    Beam: BMAJ=1.5 arcsec BMIN=1.0 arcsec BPA=0.0 deg

This masking can be applied to most operations, including `~radio_beam.Beams.common_beam` to
exclude large outliers in the set.
One useful example is if a channel is blanked in a spectral-cube, and the beam is a `NaN`.
To make a mask to select only finite beams::

    >>> my_beams.isfinite
    array([ True,  True])

