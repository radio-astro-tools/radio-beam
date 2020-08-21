.. _com_beam:

Finding the smallest common beam
================================

radio-beam implements an exact solution for sets of 2 beams and an approximate method---`the Khachiyan algorithm <https://en.wikipedia.org/wiki/Ellipsoid_method>`_---for larger sets. The former case is straightforward to compute as the beams can be transformed into a space where the larger beam is circular to find the overlap area (`~radio_beam.commonbeam.common_2beams`). Our implementation borrows from the implementation in `CASA <https://casa.nrao.edu/>`_ (see `here <https://open-bitbucket.nrao.edu/projects/CASA/repos/casa/browse/code/imageanalysis/ImageAnalysis/CasaImageBeamSet.cc>`__). Note that CASA uses this method for sets of beams larger than 2 by iterating through the beams and comparing each beam to the largest beam from the previous iterations.  However, this approach is not guaranteed to find the minimum enclosing beam.


For sets of more than two beams, finding the smallest common beam is a convex optimization problem equivalent to finding the minimum enclosed ellipse for a set of ellipses centered on the origin (`Boyd & Vandenberghe <http://web.stanford.edu/~boyd/cvxbook/>`_, see `example <http://web.cvxr.com/cvx/examples/cvxbook/Ch08_geometric_probs/html/min_vol_elp_finite_set.html>`_ in Sec 8.4.1). To avoid having radio-beam depend on convex optimization libraries, we implement the Khachiyan algorithm as an approximate method for finding the minimum ellipse.  This algorithm finds the minimum ellipse that encloses the convex hull of a set of points (`Khachiyan & Todd 1993 <https://link.springer.com/article/10.1007/BF01582144>`_, `Todd & Yildirim 2005 <https://people.orie.cornell.edu/miketodd/TYKhach.pdf>`_). By sampling a points on the boundaries of the beams in the set, we create a set of points whose convex hull is used to find the common beam.

Since the minimum ellipse method is approximate, some solutions for
the common beam will be slightly underestimated and the solution
cannot be deconvolved from the whole set of beams. To overcome
this issue, a small `epsilon` correction factor is added to the
ellipse edges to encourage a valid common beam solution.
Since `epsilon` is added to all sides, this correction will at most
increase the common beam area by :math:`(1+\epsilon)^2`.
The default values of `epsilon` is :math:`5\times10^{-4}`, so this
will have a very small effect on the size of the common beam.

The implementation in radio-beam is adapted from `a generalized python implementation <https://github.com/minillinim/ellipsoid/blob/master/ellipsoid.py>`_ and `the original matlab version <http://www.mathworks.com/matlabcentral/fileexchange/9542>`_ written by Nima Moshtagh (see accompanying paper `here <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.116.7691&rep=rep1&type=pdf>`__).


Could not find common beam to deconvolve all beams
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You may encounter the error "Could not find common beam to deconvolve all
beams." This occurs because the Khachiyan algorithm has converged to
within the allowed tolerance with a solution marginally *smaller* than the
enclosing ellipse.

To mitigate this issue, the default settings now enable the keyword `auto_increase_epsilon=True`, which allows for small increases in `epsilon` until the common beam solution can be deconvolved by all beams in the set. The solution for the common beam will be iterated until this point, or until (1)  `max_iter` is reached (default is 10) or (2) `max_epsilon` is reached (default is 1e-3). These values appear to work well with different ALMA and VLA data, but may need to be changed for specific data cubes. **If you notice these default parameters do not work for your data, please raise an issue** `here <https://github.com/radio-astro-tools/radio-beam/issues>`_.

In the case this issue persists, there are a few ways it can be fixed:

1. **Changing the tolerance.** - The default tolerance for convergence of the Khachiyan algorithm (`~radio_beam.commonbeam.getMinVolEllipse`) is `tolerance=1e-5`. This tolerance can be changed in `~radio_beam.Beams.common_beam` by specifying a new tolerance. Convergence may be met by either increasing or decreasing the tolerance; it depends on having the algorithm not step within the minimum enclosing ellipse, leading to the error. Note that decreasing the tolerance by an order of magnitude will require an order of magnitude more iterations for the algorithm to converge. It will typically be faster to change `epsilon` (see below).

2. **Changing epsilon** - A second parameter `epsilon` controls the points sampled at the edges of the beams in the set (`~radio_beam.commonbeam.ellipse_edges`), which are used in the Khachiyan algorithm. `epsilon` is the fraction beyond the true edge of the ellipse that points will be sampled at. For example, the default value of `epsilon=1e-3` will sample points 0.1% larger than the edge of the ellipse. Increasing `epsilon` ensures that a valid common beam can be found, avoiding the tolerance issue, but will result in overestimating the common beam area. For most radio data sets, where the beam is oversampled by :math:`\sim 3--5` pixels, moderate increases in `epsilon` will increase the common beam area far less than a pixel area, making the overestimation negligible.

3. **Changing the `auto_increase_epsilon` keywords** - To avoid the manual guess-and-check, the `auto_increase_epsilon` can be made more lenient to encourage a valid solution. This can be achieved by (i) increasing the intial values of `epsilon` (equivalent to #2), (ii) decreasing the number of iterations (forces larger incremental steps in `epsilon`, or (iii) increasing `max_epsilon`. (i) and (ii) will both reduce the number of iterations making it quicker to test different keyword values. (iii) allows for the common beam solution to be moderately larger. As noted above, increasing `epsilon` allows for the common beam area to be overestimated *up to* :math:`(1+\epsilon)^2`.


We recommend testing different values of tolerance to find convergence, and if the error persists, to then slowly increase epsilon until a valid common beam is found.
