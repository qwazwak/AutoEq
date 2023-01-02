using System.Collections.Generic;
using System.Linq;

namespace AutoEQ.Core;

/// <summary>
/// 1-D smoothing spline fit to a given set of data points.
/// Fits a spline y = spl(x) of degree `k` to the provided `x`, `y` data.  `s`
/// specifies the number of knots by specifying a smoothing condition.
/// </summary>
public abstract class UnivariateSpline<T>
{
    public enum Extrapolation
    {
        extrapolate,
        zeros,
        raise,
        @const,
    }
    /// <summary>
    /// 1-D array of independent input data. Must be increasing; must be strictly increasing if `s` is 0
    /// </summary>
    public IList<T> x { get; set; }
    /// <summary>
    /// 1-D array of dependent input data, of the same length as `x`.
    /// </summary>
    public IList<T> y { get; set; }
    /// <summary>
    /// Weights for spline fitting.  Must be positive.  If `w` is None, weights are all equal. Default is None.
    /// </summary>
    public IList<T>? w { get; set; }
    /// <summary>
    /// Controls the value returned for elements of `x` not in the
    /// interval defined by the knot sequence.
    /// * if ext=0 or 'extrapolate', return the extrapolated value.
    /// * if ext=1 or 'zeros', return 0
    /// * if ext=2 or 'raise', raise a ValueError
    /// * if ext=3 or 'const', return the boundary value.
    /// The default value is 0, passed from the initialization of
    /// UnivariateSpline.
    /// </summary>
    public Extrapolation ext { get; set; } = Extrapolation.extrapolate;



    /*         
 bbox : (2,) array_like, optional
     2-sequence specifying the boundary of the approximation interval. If
     `bbox` is None, ``bbox=[x[0], x[-1]]``. Default is None.
 k : int, optional
     Degree of the smoothing spline.  Must be 1 <= `k` <= 5.
     ``k = 3`` is a cubic spline. Default is 3.
 s : float or None, optional
     Positive smoothing factor used to choose the number of knots.  Number
     of knots will be increased until the smoothing condition is satisfied::
         sum((w[i] * (y[i]-spl(x[i])))**2, axis=0) <= s
     If `s` is None, ``s = len(w)`` which should be a good value if
     ``1/w[i]`` is an estimate of the standard deviation of ``y[i]``.
     If 0, spline will interpolate through all data points. Default is None.
 ext : int or str, optional
     Controls the extrapolation mode for elements
     not in the interval defined by the knot sequence.
     * if ext=0 or 'extrapolate', return the extrapolated value.
     * if ext=1 or 'zeros', return 0
     * if ext=2 or 'raise', raise a ValueError
     * if ext=3 of 'const', return the boundary value.
     Default is 0.
 check_finite : bool, optional
     Whether to check that the input arrays contain only finite numbers.
     Disabling may give a performance gain, but may result in problems
     (crashes, non-termination or non-sensical results) if the inputs
     do contain infinities or NaNs.
     Default is False.
 See Also
 --------
 BivariateSpline :
     a base class for bivariate splines.
 SmoothBivariateSpline :
     a smoothing bivariate spline through the given points
 LSQBivariateSpline :
     a bivariate spline using weighted least-squares fitting
 RectSphereBivariateSpline :
     a bivariate spline over a rectangular mesh on a sphere
 SmoothSphereBivariateSpline :
     a smoothing bivariate spline in spherical coordinates
 LSQSphereBivariateSpline :
     a bivariate spline in spherical coordinates using weighted
     least-squares fitting
 RectBivariateSpline :
     a bivariate spline over a rectangular mesh
 InterpolatedUnivariateSpline :
     a interpolating univariate spline for a given set of data points.
 bisplrep :
     a function to find a bivariate B-spline representation of a surface
 bisplev :
     a function to evaluate a bivariate B-spline and its derivatives
 splrep :
     a function to find the B-spline representation of a 1-D curve
 splev :
     a function to evaluate a B-spline or its derivatives
 sproot :
     a function to find the roots of a cubic B-spline
 splint :
     a function to evaluate the definite integral of a B-spline between two
     given points
 spalde :
     a function to evaluate all derivatives of a B-spline
 Notes
 -----
 The number of data points must be larger than the spline degree `k`.
 **NaN handling**: If the input arrays contain ``nan`` values, the result
 is not useful, since the underlying spline fitting routines cannot deal
 with ``nan``. A workaround is to use zero weights for not-a-number
 data points:
 >>> from scipy.interpolate import UnivariateSpline
 >>> x, y = np.array([1, 2, 3, 4]), np.array([1, np.nan, 3, 4])
 >>> w = np.isnan(y)
 >>> y[w] = 0.
 >>> spl = UnivariateSpline(x, y, w=~w)
 Notice the need to replace a ``nan`` by a numerical value (precise value
 does not matter as long as the corresponding weight is zero.)
 Examples
 --------
 >>> import matplotlib.pyplot as plt
 >>> from scipy.interpolate import UnivariateSpline
 >>> rng = np.random.default_rng()
 >>> x = np.linspace(-3, 3, 50)
 >>> y = np.exp(-x**2) + 0.1 * rng.standard_normal(50)
 >>> plt.plot(x, y, 'ro', ms=5)
 Use the default value for the smoothing parameter:
 >>> spl = UnivariateSpline(x, y)
 >>> xs = np.linspace(-3, 3, 1000)
 >>> plt.plot(xs, spl(xs), 'g', lw=3)
 Manually change the amount of smoothing:
 >>> spl.set_smoothing_factor(0.5)
 >>> plt.plot(xs, spl(xs), 'b', lw=3)
 >>> plt.show()
 */
    /*
 def __init__(self, x, y, w=None, bbox=[None]*2, k=3, s=None,
              ext=0, check_finite=False):

     x, y, w, bbox, self.ext = self.validate_input(x, y, w, bbox, k, s, ext,
                                                   check_finite)

     # _data == x,y,w,xb,xe,k,s,n,t,c,fp,fpint,nrdata,ier
     data = dfitpack.fpcurf0(x, y, k, w=w, xb=bbox[0],
                             xe=bbox[1], s=s)
     if data[-1] == 1:
         # nest too small, setting to maximum bound
         data = self._reset_nest(data)
     self._data = data
     self._reset_class()

 @staticmethod
 def validate_input(x, y, w, bbox, k, s, ext, check_finite):
     x, y, bbox = np.asarray(x), np.asarray(y), np.asarray(bbox)
     if w is not None:
         w = np.asarray(w)
     if check_finite:
         w_finite = np.isfinite(w).all() if w is not None else True
         if (not np.isfinite(x).all() or not np.isfinite(y).all() or
                 not w_finite):
             raise ValueError("x and y array must not contain "
                              "NaNs or infs.")
     if s is None or s > 0:
         if not np.all(diff(x) >= 0.0):
             raise ValueError("x must be increasing if s > 0")
     else:
         if not np.all(diff(x) > 0.0):
             raise ValueError("x must be strictly increasing if s = 0")
     if x.size != y.size:
         raise ValueError("x and y should have a same length")
     elif w is not None and not x.size == y.size == w.size:
         raise ValueError("x, y, and w should have a same length")
     elif bbox.shape != (2,):
         raise ValueError("bbox shape should be (2,)")
     elif not (1 <= k <= 5):
         raise ValueError("k should be 1 <= k <= 5")
     elif s is not None and not s >= 0.0:
         raise ValueError("s should be s >= 0.0")

     try:
         ext = _extrap_modes[ext]
     except KeyError as e:
         raise ValueError("Unknown extrapolation mode %s." % ext) from e

     return x, y, w, bbox, ext

 @classmethod
 def _from_tck(cls, tck, ext=0):
     """Construct a spline object from given tck"""
     self = cls.__new__(cls)
     t, c, k = tck
     self._eval_args = tck
     # _data == x,y,w,xb,xe,k,s,n,t,c,fp,fpint,nrdata,ier
     self._data = (None, None, None, None, None, k, None, len(t), t,
                   c, None, None, None, None)
     self.ext = ext
     return self

 def _reset_class(self):
     data = self._data
     n, t, c, k, ier = data[7], data[8], data[9], data[5], data[-1]
     self._eval_args = t[:n], c[:n], k
     if ier == 0:
         # the spline returned has a residual sum of squares fp
         # such that abs(fp-s)/s <= tol with tol a relative
         # tolerance set to 0.001 by the program
         pass
     elif ier == -1:
         # the spline returned is an interpolating spline
         self._set_class(InterpolatedUnivariateSpline)
     elif ier == -2:
         # the spline returned is the weighted least-squares
         # polynomial of degree k. In this extreme case fp gives
         # the upper bound fp0 for the smoothing factor s.
         self._set_class(LSQUnivariateSpline)
     else:
         # error
         if ier == 1:
             self._set_class(LSQUnivariateSpline)
         message = _curfit_messages.get(ier, 'ier=%s' % (ier))
         warnings.warn(message)

 def _set_class(self, cls):
     self._spline_class = cls
     if self.__class__ in (UnivariateSpline, InterpolatedUnivariateSpline,
                           LSQUnivariateSpline):
         self.__class__ = cls
     else:
         # It's an unknown subclass -- don't change class. cf. #731
         pass

 def _reset_nest(self, data, nest=None):
     n = data[10]
     if nest is None:
         k, m = data[5], len(data[0])
         nest = m+k+1  # this is the maximum bound for nest
     else:
         if not n <= nest:
             raise ValueError("`nest` can only be increased")
     t, c, fpint, nrdata = [np.resize(data[j], nest) for j in
                            [8, 9, 11, 12]]

     args = data[:8] + (t, c, n, fpint, nrdata, data[13])
     data = dfitpack.fpcurf1(*args)
     return data

 def set_smoothing_factor(self, s):
     """ Continue spline computation with the given smoothing
     factor s and with the knots found at the last call.
     This routine modifies the spline in place.
     """
     data = self._data
     if data[6] == -1:
         warnings.warn('smoothing factor unchanged for'
                       'LSQ spline with fixed knots')
         return
     args = data[:6] + (s,) + data[7:]
     data = dfitpack.fpcurf1(*args)
     if data[-1] == 1:
         # nest too small, setting to maximum bound
         data = self._reset_nest(data)
     self._data = data
     self._reset_class()
    */
    /// <summary>
    /// Evaluate spline (or its nu-th derivative) at positions x.
    /// </summary>
    /// <param name="x">A 1-D array of points at which to return the value of the smoothed
    /// spline or its derivatives.Note: `x` can be unordered but the
    /// evaluation is more efficient if `x` is (partially) ordered.</param>
    /// <param name="nu">The order of derivative of the spline to compute.</param>
    /// <param name="ext">Controls the value returned for elements of `x` not in the
    /// interval defined by the knot sequence.
    /// * if ext=0 or 'extrapolate', return the extrapolated value.
    /// * if ext=1 or 'zeros', return 0
    /// * if ext=2 or 'raise', raise a ValueError
    /// * if ext=3 or 'const', return the boundary value.
    /// The default value is 0, passed from the initialization of
    /// UnivariateSpline.</param>
    /// <returns></returns>
    public IEnumerable<double> __call__(IEnumerable<double> x, int nu = 0, Extrapolation? ext =null)
    {
        if (!x.Any())
        {
            yield break;
        }
        ext ??= this.ext;
        return _fitpack_py.splev(x, self._eval_args, der = nu, ext = ext
    }
    /// <summary>
    /// Evaluate a B-spline or its derivatives.
    /// Given the knots and coefficients of a B-spline representation, evaluate
    /// the value of the smoothing polynomial and its derivatives.This is a
    /// wrapper around the FORTRAN routines splev and splder of FITPACK.
    /// </summary>
    /// <param name="x"></param>
    /// <param name=""></param>
    /// <param name=""></param>
    /// <param name=""></param>
    /// <returns></returns>
    private static IEnumerable<double>  splev(IEnumerable<double> x, tck, der=0, ext= 0) {

    """
    
    Parameters
    ----------
    x : array_like
        An array of points at which to return the value of the smoothed
        spline or its derivatives. If `tck` was returned from `splprep`,
        then the parameter values, u should be given.
    tck : 3-tuple or a BSpline object
        If a tuple, then it should be a sequence of length 3 returned by
        `splrep` or `splprep` containing the knots, coefficients, and degree
        of the spline. (Also see Notes.)
    der : int, optional
        The order of derivative of the spline to compute (must be less than
        or equal to k, the degree of the spline).
    ext : int, optional
        Controls the value returned for elements of ``x`` not in the
        interval defined by the knot sequence.
        * if ext=0, return the extrapolated value.
        * if ext=1, return 0
        * if ext=2, raise a ValueError
        * if ext=3, return the boundary value.
        The default value is 0.
    Returns
    -------
    y : ndarray or list of ndarrays
        An array of values representing the spline function evaluated at
        the points in `x`.  If `tck` was returned from `splprep`, then this
        is a list of arrays representing the curve in an N-D space.
    Notes
    -----
    Manipulating the tck-tuples directly is not recommended. In new code,
    prefer using `BSpline` objects.
    See Also
    --------
    splprep, splrep, sproot, spalde, splint
    bisplrep, bisplev
    BSpline
    References
    ----------
    .. [1] C. de Boor, "On calculating with b-splines", J. Approximation
        Theory, 6, p.50-62, 1972.
    .. [2] M. G. Cox, "The numerical evaluation of b-splines", J. Inst. Maths
        Applics, 10, p.134-149, 1972.
    .. [3] P. Dierckx, "Curve and surface fitting with splines", Monographs
        on Numerical Analysis, Oxford University Press, 1993.
    Examples
    --------
    Examples are given :ref:`in the tutorial <tutorial-interpolate_splXXX>`.
    """
    if isinstance(tck, BSpline):
        if tck.c.ndim > 1:
            mesg = ("Calling splev() with BSpline objects with c.ndim > 1 is "
                    "not allowed. Use BSpline.__call__(x) instead.")
            raise ValueError(mesg)

        # remap the out-of-bounds behavior
        try:
            extrapolate = {0: True, }[ext]
        except KeyError as e:
            raise ValueError("Extrapolation mode %s is not supported "
                             "by BSpline." % ext) from e

        return tck(x, der, extrapolate=extrapolate)
    else:
        return _impl.splev(x, tck, der, ext)
    }
       /*
    def get_knots(self):
        """ Return positions of interior knots of the spline.
        Internally, the knot vector contains ``2*k`` additional boundary knots.
        """
        data = self._data
        k, n = data[5], data[7]
        return data[8][k:n-k]

    def get_coeffs(self):
        """Return spline coefficients."""
        data = self._data
        k, n = data[5], data[7]
        return data[9][:n-k-1]

    def get_residual(self):
        """Return weighted sum of squared residuals of the spline approximation.
           This is equivalent to::
                sum((w[i] * (y[i]-spl(x[i])))**2, axis=0)
        """
        return self._data[10]

    def integral(self, a, b):
        """ Return definite integral of the spline between two given points.
        Parameters
        ----------
        a : float
            Lower limit of integration.
        b : float
            Upper limit of integration.
        Returns
        -------
        integral : float
            The value of the definite integral of the spline between limits.
        Examples
        --------
        >>> from scipy.interpolate import UnivariateSpline
        >>> x = np.linspace(0, 3, 11)
        >>> y = x**2
        >>> spl = UnivariateSpline(x, y)
        >>> spl.integral(0, 3)
        9.0
        which agrees with :math:`\\int x^2 dx = x^3 / 3` between the limits
        of 0 and 3.
        A caveat is that this routine assumes the spline to be zero outside of
        the data limits:
        >>> spl.integral(-1, 4)
        9.0
        >>> spl.integral(-1, 0)
        0.0
        """
        return dfitpack.splint(*(self._eval_args+(a, b)))

    def derivatives(self, x):
        """ Return all derivatives of the spline at the point x.
        Parameters
        ----------
        x : float
            The point to evaluate the derivatives at.
        Returns
        -------
        der : ndarray, shape(k+1,)
            Derivatives of the orders 0 to k.
        Examples
        --------
        >>> from scipy.interpolate import UnivariateSpline
        >>> x = np.linspace(0, 3, 11)
        >>> y = x**2
        >>> spl = UnivariateSpline(x, y)
        >>> spl.derivatives(1.5)
        array([2.25, 3.0, 2.0, 0])
        """
        d, ier = dfitpack.spalde(*(self._eval_args+(x,)))
        if not ier == 0:
            raise ValueError("Error code returned by spalde: %s" % ier)
        return d

    def roots(self):
        """ Return the zeros of the spline.
        Restriction: only cubic splines are supported by fitpack.
        """
        k = self._data[5]
        if k == 3:
            z, m, ier = dfitpack.sproot(*self._eval_args[:2])
            if not ier == 0:
                raise ValueError("Error code returned by spalde: %s" % ier)
            return z[:m]
        raise NotImplementedError('finding roots unsupported for '
                                  'non-cubic splines')

    def derivative(self, n=1):
        """
        Construct a new spline representing the derivative of this spline.
        Parameters
        ----------
        n : int, optional
            Order of derivative to evaluate. Default: 1
        Returns
        -------
        spline : UnivariateSpline
            Spline of order k2=k-n representing the derivative of this
            spline.
        See Also
        --------
        splder, antiderivative
        Notes
        -----
        .. versionadded:: 0.13.0
        Examples
        --------
        This can be used for finding maxima of a curve:
        >>> from scipy.interpolate import UnivariateSpline
        >>> x = np.linspace(0, 10, 70)
        >>> y = np.sin(x)
        >>> spl = UnivariateSpline(x, y, k=4, s=0)
        Now, differentiate the spline and find the zeros of the
        derivative. (NB: `sproot` only works for order 3 splines, so we
        fit an order 4 spline):
        >>> spl.derivative().roots() / np.pi
        array([ 0.50000001,  1.5       ,  2.49999998])
        This agrees well with roots :math:`\\pi/2 + n\\pi` of
        :math:`\\cos(x) = \\sin'(x)`.
        """
        tck = _fitpack_py.splder(self._eval_args, n)
        # if self.ext is 'const', derivative.ext will be 'zeros'
        ext = 1 if self.ext == 3 else self.ext
        return UnivariateSpline._from_tck(tck, ext=ext)

    def antiderivative(self, n=1):
        """
        Construct a new spline representing the antiderivative of this spline.
        Parameters
        ----------
        n : int, optional
            Order of antiderivative to evaluate. Default: 1
        Returns
        -------
        spline : UnivariateSpline
            Spline of order k2=k+n representing the antiderivative of this
            spline.
        Notes
        -----
        .. versionadded:: 0.13.0
        See Also
        --------
        splantider, derivative
        Examples
        --------
        >>> from scipy.interpolate import UnivariateSpline
        >>> x = np.linspace(0, np.pi/2, 70)
        >>> y = 1 / np.sqrt(1 - 0.8*np.sin(x)**2)
        >>> spl = UnivariateSpline(x, y, s=0)
        The derivative is the inverse operation of the antiderivative,
        although some floating point error accumulates:
        >>> spl(1.7), spl.antiderivative().derivative()(1.7)
        (array(2.1565429877197317), array(2.1565429877201865))
        Antiderivative can be used to evaluate definite integrals:
        >>> ispl = spl.antiderivative()
        >>> ispl(np.pi/2) - ispl(0)
        2.2572053588768486
        This is indeed an approximation to the complete elliptic integral
        :math:`K(m) = \\int_0^{\\pi/2} [1 - m\\sin^2 x]^{-1/2} dx`:
        >>> from scipy.special import ellipk
        >>> ellipk(0.8)
        2.2572053268208538
        """
        tck = _fitpack_py.splantider(self._eval_args, n)
        return UnivariateSpline._from_tck(tck, self.ext)

    */
}
/*
class InterpolatedUnivariateSpline(UnivariateSpline):
    """
    1-D interpolating spline for a given set of data points.
    Fits a spline y = spl(x) of degree `k` to the provided `x`, `y` data.
    Spline function passes through all provided points. Equivalent to
    `UnivariateSpline` with  s=0.
    Parameters
    ----------
    x : (N,) array_like
        Input dimension of data points -- must be strictly increasing
    y : (N,) array_like
        input dimension of data points
    w : (N,) array_like, optional
        Weights for spline fitting.  Must be positive.  If None (default),
        weights are all equal.
    bbox : (2,) array_like, optional
        2-sequence specifying the boundary of the approximation interval. If
        None (default), ``bbox=[x[0], x[-1]]``.
    k : int, optional
        Degree of the smoothing spline.  Must be 1 <= `k` <= 5.
    ext : int or str, optional
        Controls the extrapolation mode for elements
        not in the interval defined by the knot sequence.
        * if ext=0 or 'extrapolate', return the extrapolated value.
        * if ext=1 or 'zeros', return 0
        * if ext=2 or 'raise', raise a ValueError
        * if ext=3 of 'const', return the boundary value.
        The default value is 0.
    check_finite : bool, optional
        Whether to check that the input arrays contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination or non-sensical results) if the inputs
        do contain infinities or NaNs.
        Default is False.
    See Also
    --------
    UnivariateSpline :
        a smooth univariate spline to fit a given set of data points.
    LSQUnivariateSpline :
        a spline for which knots are user-selected
    SmoothBivariateSpline :
        a smoothing bivariate spline through the given points
    LSQBivariateSpline :
        a bivariate spline using weighted least-squares fitting
    splrep :
        a function to find the B-spline representation of a 1-D curve
    splev :
        a function to evaluate a B-spline or its derivatives
    sproot :
        a function to find the roots of a cubic B-spline
    splint :
        a function to evaluate the definite integral of a B-spline between two
        given points
    spalde :
        a function to evaluate all derivatives of a B-spline
    Notes
    -----
    The number of data points must be larger than the spline degree `k`.
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from scipy.interpolate import InterpolatedUnivariateSpline
    >>> rng = np.random.default_rng()
    >>> x = np.linspace(-3, 3, 50)
    >>> y = np.exp(-x**2) + 0.1 * rng.standard_normal(50)
    >>> spl = InterpolatedUnivariateSpline(x, y)
    >>> plt.plot(x, y, 'ro', ms=5)
    >>> xs = np.linspace(-3, 3, 1000)
    >>> plt.plot(xs, spl(xs), 'g', lw=3, alpha=0.7)
    >>> plt.show()
    Notice that the ``spl(x)`` interpolates `y`:
    >>> spl.get_residual()
    0.0
    """
    def __init__(self, x, y, w=None, bbox=[None]*2, k=3,
                 ext=0, check_finite=False):

        x, y, w, bbox, self.ext = self.validate_input(x, y, w, bbox, k, None,
                                            ext, check_finite)
        if not np.all(diff(x) > 0.0):
            raise ValueError('x must be strictly increasing')

        # _data == x,y,w,xb,xe,k,s,n,t,c,fp,fpint,nrdata,ier
        self._data = dfitpack.fpcurf0(x, y, k, w=w, xb=bbox[0],
                                      xe=bbox[1], s=0)
        self._reset_class()*/