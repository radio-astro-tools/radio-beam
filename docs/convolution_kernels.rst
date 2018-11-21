.. _convkernels:

Making convolution kernels
==========================

`~radio_beam.Beam` can produce two types of kernels: a Gaussian (`~radio_beam.Beam.as_kernel`) and a top-hat (`~radio_beam.Beam.as_tophat_kernel`).

As an example, consider the elliptical beam::

    >>> import astropy.units as u
    >>> from radio_beam import Beam
    >>> my_beam = Beam(3*u.arcsec, 1.5*u.arcsec, 60*u.deg)


Gaussian
^^^^^^^^

`~radio_beam.Beam.as_kernel` will return an elliptical Gaussian kernel given the angular size of a pixel::

    >>> pix_scale = 0.5 * u.arcsec
    >>> gauss_kern = my_beam.as_kernel(pix_scale)

`gauss_kern` will be a `~radio_beam.beam.EllipticalGaussian2DKernel` object and has the same methods, attributes and keyword arguments as `Kernel2D <http://docs.astropy.org/en/stable/api/astropy.convolution.Kernel2D.html#astropy.convolution.Kernel2D>`__ in astropy's convolution package. These keyword arguments can be passed to `~radio_beam.Beam.as_kernel`.  See the `astropy documentation <http://docs.astropy.org/en/stable/convolution/kernels.html>`_ for more information on convolution kernels.

Top-Hat
^^^^^^^

`~radio_beam.Beam.as_tophat_kernel` returns an elliptical top-hat kernel scales to have the same area as a Gaussian kernel within the FWHM.  Similar to the Gaussian kernel, only the pixel scale needs to be given::

    >>> tophat_kern = my_beam.as_tophat_kernel(pix_scale)

`tophat_kern` is a `~radio_beam.beam.EllipticalTophat2DKernel` object, also derived from `Kernel2D <http://docs.astropy.org/en/stable/api/astropy.convolution.Kernel2D.html#astropy.convolution.Kernel2D>`__ in astropy's convolution package. Keyword arguments can be passed to `~radio_beam.Beam.as_tophat_kernel`.

The values in the kernel are normalized to unity, and it is suitable for convolution.  However, the top-hat kernel is also useful for masking purposes, in which case a boolean version of the kernel is useful.  To make a boolean version, we need to access the array in the kernel object and look for non-zero values::

    >>> tophat_kern_bool = tophat_kern.array > 0

`tophat_kern_bool` is suitable for use with morphological operations, such as those in `scipy.ndimage <https://docs.scipy.org/doc/scipy/reference/ndimage.html>`_.

Convolution kernels from multiple beams
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

From a `~radio_beam.Beams` object, a convolution kernel can be made for each beam in the set by slicing::

    >>> from radio_beam import Beams
    >>> from astropy.io import fits
    >>> bin_hdu = fits.open('file.fits')[1]  # doctest: +SKIP
    >>> beams = Beams.from_fits_bintable(bin_hdu)  # doctest: +SKIP
    >>> beams[0].as_kernel(pix_scale)  # doctest: +SKIP

