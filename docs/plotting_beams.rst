.. _plotting:

Add a beam to a matplotlib plot
===============================

To show the beam on an image in matplotlib, use `~radio_beam.Beam.ellipse_to_plot`::

    >>> from radio_beam import Beam
    >>> import astropy.units as u
    >>> import matplotlib.pyplot as plt
    >>> my_beam = Beam(5*u.arcsec, 3*u.arcsec, 30*u.deg)
    >>> ycen_pix, xcen_pix = 15, 15
    >>> pixscale = 1 * u.arcsec
    >>> ellipse_artist = my_beam.ellipse_to_plot(xcen_pix, ycen_pix, pixscale)
    >>> ax = plt.imshow(image)  # doctest: +SKIP
    >>> _ = ax.add_artist(ellipse_artist)  # doctest: +SKIP

The three inputs you need for adding to an arbitrary image are the x and y coordinates
to center the beam at in the image, and the pixel scale of the image as defined in the WCS
information.
