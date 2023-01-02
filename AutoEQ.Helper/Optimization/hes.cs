using AutoEQ.Helper;
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;

namespace AutoEQ.Core;
/*
public static class RandomPython
{

def get_blas_funcs(names, arrays=(), dtype=None, ilp64=False):
    """Return available BLAS function objects from names.
    Arrays are used to determine the optimal prefix of BLAS routines.
    Parameters
    ----------
    names : str or sequence of str
        Name(s) of BLAS functions without type prefix.
    arrays : sequence of ndarrays, optional
        Arrays can be given to determine optimal prefix of BLAS
        routines. If not given, double-precision routines will be
        used, otherwise the most generic type in arrays will be used.
    dtype : str or dtype, optional
        Data-type specifier. Not used if `arrays` is non-empty.
    ilp64 : {True, False, "preferred"}, optional
        Whether to return ILP64 routine variant.
        Choosing "preferred" returns ILP64 routine if available,
        and otherwise the 32-bit routine. Default: False
    Returns
    -------
    funcs : list
        List containing the found function(s).
    Notes
    -----
    This routine automatically chooses between Fortran/C
    interfaces. Fortran code is used whenever possible for arrays with
    column major order. In all other cases, C code is preferred.
    In BLAS, the naming convention is that all functions start with a
    type prefix, which depends on the type of the principal
    matrix. These can be one of {"s', "d", "c", "z'} for the NumPy
    types {float32, float64, complex64, complex128} respectively.
    The code and the dtype are stored in attributes `typecode` and `dtype`
    of the returned functions.
    Examples
    --------
    >>> import numpy as np
    >>> import scipy.linalg as LA
    >>> rng = np.random.default_rng()
    >>> a = rng.random((3,2))
    >>> x_gemv = LA.get_blas_funcs("gemv", (a,))
    >>> x_gemv.typecode
    'd'
    >>> x_gemv = LA.get_blas_funcs("gemv",(a*1j,))
    >>> x_gemv.typecode
    'z'
    """
    if isinstance(ilp64, str):
        if ilp64 == "preferred":
            ilp64 = HAS_ILP64
        else:
            raise ValueError("Invalid value for "ilp64"")

    if not ilp64:
        return _get_funcs(names, arrays, dtype,
                          "BLAS", _fblas, _cblas, "fblas", "cblas",
                          _blas_alias, ilp64=False)
    else:
        if not HAS_ILP64:
            raise RuntimeError("BLAS ILP64 routine requested, but Scipy "
                               "compiled only with 32-bit BLAS")
        return _get_funcs(names, arrays, dtype,
                          "BLAS", _fblas_64, None, "fblas_64", None,
                          _blas_alias, ilp64= True)
}*/
/// <summary>
/// Interface for implementing Hessian update strategies.
/// </summary>
/// <remarks>
/// Many optimization methods make use of Hessian (or inverse Hessian)
/// approximations, such as the quasi-Newton methods BFGS, SR1, L-BFGS.
/// Some of these  approximations, however, do not actually need to store
/// the entire matrix or can compute the internal matrix product with a
/// given vector in a very efficiently manner.This class serves as an
/// abstract interface between the optimization algorithm and the
/// quasi-Newton update strategies, giving freedom of implementation
/// to store and update the internal matrix as efficiently as possible.
/// Different choices of initialization and update procedure will result
/// in different quasi-Newton strategies.
/// Four methods should be implemented in derived classes: ``initialize``,
/// ``update``, ``dot`` and ``get_matrix``.
/// Notes
/// -----
/// Any instance of a class that implements this interface,
/// can be accepted by the method ``minimize`` and used by
/// the compatible solvers to approximate the Hessian(or
/// inverse Hessian) used by the optimization algorithms.
/// </remarks>
public interface HessianUpdateStrategy
{
    /// <summary>
    /// Initialize internal matrix.  Allocate internal memory for storing and updating the Hessian or its inverse.
    /// </summary>
    /// <param name="n">Problem dimension.</param>
    /// <param name="approx_type">
    /// { "hess", "inv_hess"}
    /// Selects either the Hessian or the inverse Hessian.
    /// When set to "hess" the Hessian will be stored and updated.
    /// When set to "inv_hess" its inverse will be used instead.
    /// </param>
    public void initialize(int n, string approx_type);

    /// <summary>
    /// Update internal matrix.
    /// Update Hessian matrix or its inverse(depending on how "approx_type"
    /// is defined) using information about the last evaluated points.
    /// </summary>
    /// <param name="delta_x">he difference between two points the gradient function have been evaluated at: ``delta_x = x2 - x1</param>
    /// <param name="delta_grad">The difference between the gradients: ``delta_grad = grad(x2) - grad(x1)``. </param>
    public void update(IList<double> delta_x, IList<double> delta_grad);

    /// <summary>
    /// Compute the product of the internal matrix with the given vector.
    /// </summary>
    /// <param name="p">1-D array representing a vector.</param>
    /// <returns>1-D represents the result of multiplying the approximation matrix by vector p.</returns>
    public double dot(IList<double> p);
    /// <summary>
    /// Return current internal matrix.
    /// </summary>
    /// <returns>
    /// Dense matrix containing either the Hessian or its inverse(depending on how "approx_type" is defined).
    /// </returns>
    public double[,] get_matrix();
}
/// <summary>
/// Hessian update strategy with full dimensional internal representation.
/// </summary>
public abstract class FullHessianUpdateStrategy : HessianUpdateStrategy
{
    //_syr = get_blas_funcs("syr", dtype= 'd')  // Symmetric rank 1 update
    //_syr2 = get_blas_funcs("syr2", dtype= 'd')  // Symmetric rank 2 update
    // Symmetric matrix-vector product
    //_symv = get_blas_funcs("symv", dtype= 'd')

    private unsafe static double[,] symv(bool UseUpper = true)
    {
        sbyte UPLO = UseUpper ? (sbyte)OpenBLAS.CBLAS_UPLO.CblasUpper : (sbyte)OpenBLAS.CBLAS_UPLO.CblasLower;

        double[,] Result = new double[6,6];
        double[] Y_RESULT = Result.;
        return symv(UPLO, N, ALPHA, A, LDA, X, INCX, BETA, ref Y_RESULT, INCY);
    }

    /// <summary>
    /// DSYMV  performs the matrix-vector  operation
    ///     y := alpha*A*x + beta*y,
    /// where alpha and beta are scalars, x and y are n element vectors and A is an n by n symmetric matrix.
    /// </summary>
    /// <param name="UPLO">
    /// On entry, UPLO specifies whether the upper or lower
    /// triangular part of the array A is to be referenced as
    /// follows:
    /// 
    /// UPLO = "U' or "u'   Only the upper triangular part of A is to be referenced.
    /// 
    /// UPLO = "L' or "l'   Only the lower triangular part of A is to be referenced.
    /// </param>
    /// <param name="N">On entry, N specifies the order of the matrix A. N must be at least zero.</param>
    /// <param name="ALPHA">ALPHA is DOUBLE PRECISION. On entry, ALPHA specifies the scalar alpha.</param>
    /// <param name="A">
    ///     A is DOUBLE PRECISION array, dimension(LDA, N )
    ///     Before entry with UPLO = "U' or "u', the leading n by n
    ///     upper triangular part of the array A must contain the upper
    ///     triangular part of the symmetric matrix and the strictly
    ///     lower triangular part of A is not referenced.
    ///     Before entry with UPLO = "L' or "l', the leading n by n
    ///     lower triangular part of the array A must contain the lower
    ///     triangular part of the symmetric matrix and the strictly
    ///     upper triangular part of A is not referenced.
    ///     </param>
    /// <param name="LDA">On entry, LDA specifies the first dimension of A as declared in the calling (sub) program. LDA must be at least max( 1, n ).</param>
    /// <param name="X">X is DOUBLE PRECISION array, dimension at least ( 1 + ( n - 1 )*abs( INCX ) ). Before entry, the incremented array X must contain the n element vector x.</param>
    /// <param name="INCX">On entry, INCX specifies the increment for the elements of X. INCX must not be zero.</param>
    /// <param name="BETA">On entry, BETA specifies the scalar beta. When BETA is supplied as zero then Y need not be set on input.</param>
    /// <param name="Y">Y is DOUBLE PRECISION array, dimension at least ( 1 + ( n - 1 )*abs( INCY ) ). Before entry, the incremented array Y must contain the n element vector y. On exit, Y is overwritten by the updated vector </param>
    /// <param name="INCY">On entry, INCY specifies the increment for the elements of Y. INCY must not be zero.</param>
    /// <returns></returns>
    private unsafe static void symv(sbyte UPLO, int N, double ALPHA, double[] A, int LDA, double[] X, int INCX, double BETA, ref double[] Y, int INCY)
    {
        //UPLO,N,ALPHA,A,LDA,X,INCX,BETA,Y,INCY)
        OpenBLAS.BLAS.Dsymv(&UPLO, ref N, ref ALPHA, ref A, ref LDA, ref X, ref INCX, ref BETA, ref Y, ref INCY);
    //    OpenBLAS.BLAS.Dsymv(&nta, &nta, ref m, ref n, ref k, ref alpha, ref a[0, 0], ref lda, ref b[0, 0], ref ldb, ref beta, ref c[0, 0], ref ldc);
    }

    /// <summary>
    /// When null, uses automatic scaling
    /// </summary>
    public double? init_scale { get; init; } = null;
    public bool first_iteration { get; private set; } = null!;
    public bool StdHess { get; private set; }
    public bool InvHess { get => !StdHess; private set => StdHess = !value; }
    public int Length { get; private set; }
    public double[]? Matrix { get; protected set; } = null;
    //public double[]? H { get; private set; } = null;
    //public double[]? B { get; private set; } = null;

    // Until initialize is called we can't really use the class, so it makes sense to set everything to None.
    protected FullHessianUpdateStrategy(double? init_scale) : this() => this.init_scale = init_scale;
    protected FullHessianUpdateStrategy() { }
    /// <inheritdoc>/>
    /// <summary>
    /// Initialize internal matrix.
    /// Allocate internal memory for storing and updating
    /// the Hessian or its inverse.
    /// </summary>
    public void initialize(int n, string approx_type)
    {
        first_iteration = true;

        this.n = n;

        // Create matrix
        if (approx_type == "hess")
            StdHess = true;
        else if (approx_type == "invhess")
            InvHess = true;
        else
            throw new ArgumentOutOfRangeException(nameof(approx_type), $"{nameof(approx_type)} must be \"hess\" or \"inv_hess\"");
        //if (this.approx_type == "hess")
        //    B = Identity2D(n);
        //else
        //    H = Identity2D(n);
        Matrix = Identity(n);
        Length = n;
    }

    private static double[,] Identity2D(int n)
    {
        double[,] arr = new double[n,n];
        for (int i = 0; i < n; i++)
            arr[i, i] = 1;
        return arr;
    }
    private protected double[] Identity(int n, double val = 1)
    {
        int NSqr = n * n;
        double[] arr = new double[NSqr];
        for (int i = 0; i < NSqr; i++)
            arr[(i * n) + i] = val;
        return arr;
    }

    protected double _auto_scale(IEnumerable<double> delta_x, IEnumerable<double> delta_grad)
    {
        // Heuristic to scale matrix at first iteration.
        // Described in Nocedal and Wright "Numerical Optimization"
        // p.143 formula (6.20).
        double s_norm2 = delta_x.DotProductSingle();
        if (s_norm2 == 0.0) return 1;

        double y_norm2 = delta_grad.DotProductSingle();
        if (y_norm2 == 0.0) return 1;

        double ys = Math.Abs(delta_grad.DotProduct(delta_x));
        if(ys == 0.0) return 1;

        if (StdHess)
            return y_norm2 / ys;
        else
            return ys / y_norm2;
    }

    protected abstract void _update_implementation(IList<double> delta_x, IList<double> delta_grad);

    public void update(IList<double> delta_x, IList<double> delta_grad)
    {
        if (delta_x.All(dx => dx == 0))
            return;
        if (delta_grad.All(dg => dg == 0))
            warn("delta_grad == 0.0. Check if the approximated  function is linear. If the function is linear  better results can be obtained by defining the  Hessian as zero instead of using quasi-Newton approximations.");

        if (first_iteration)
        {
            double scale;
            // Get user specific scale
            if (init_scale.HasValue)
                scale = init_scale.Value;
            else
                scale = _auto_scale(delta_x, delta_grad);
            // Scale initial matrix with ``scale * np.eye(n)``
            //(approx_type == "hess" ? B : H).MultEach(scale)
            Matrix.MultEach(scale);
            first_iteration = false;
        }
        _update_implementation(delta_x, delta_grad);
    }
    public IList<double> dot(IList<double> p)
    {
        return symv(1, Matrix, p);
    }

    public dynamic get_matrix()
    {
        double[,] Result = new double[Length, Length];
        foreach ((int i, int j) in tril(Length, -1))
        {
            Result[i, j] = Matrix[(i * Length) + j];
        }
        return Result;

        static IEnumerable<(int, int)> tril(int size, int k = 0)
        {
            int RowCount = 0;
            for (int i = 0 - k; i < size; i++)
            {
                foreach (int j in Enumerable.Range(0, ++RowCount))
                    yield return (i, j);
            }
        }
    }
}

public enum ExceptionStrategies
{
    skip_update,
    damp_update
}
/// <summary>
/// Broyden-Fletcher-Goldfarb-Shanno (BFGS) Hessian update strategy.
/// </summary>
public class BFGS : FullHessianUpdateStrategy
{
    public ExceptionStrategies exception_strategy { get; init; } = ExceptionStrategies.skip_update;
    public double min_curvature { get; init; }
    /*
    exception_strategy : {"skip_update", "damp_update"}, optional
        Define how to proceed when the curvature condition is violated.
        Set it to "skip_update" to just skip the update. Or, alternatively,
        set it to "damp_update" to interpolate between the actual BFGS
        result and the unmodified matrix. Both exceptions strategies
        are explained  in [1] _, p .536 - 537.
      min_curvature : float
          This number, scaled by a normalization factor, defines the
        minimum curvature ``dot(delta_grad, delta_x)`` allowed to go
        unaffected by the exception strategy. By default is equal to
        1e-8 when ``exception_strategy = "skip_update"`` and equal
        to 0.2 when ``exception_strategy = "damp_update"``.
    init_scale : { float, "auto"}
Matrix scale at first iteration. At the first
        iteration the Hessian matrix or its inverse will be initialized
        with ``init_scale*np.eye(n)``, where ``n`` is the problem dimension.
        Set it to "auto" in order to use an automatic heuristic for choosing
        the initial scale. The heuristic is described in [1] _, p .143.
          By default uses "auto".
      Notes
      ---- -
      The update is based on the description in [1] _, p .140.
        References
        ----------.. [1] Nocedal, Jorge, and Stephen J. Wright. "Numerical optimization"
           Second Edition (2006).
    */
    /*
    public BFGS(exception_strategy= "skip_update", min_curvature= None,
                 init_scale= "auto")
    {
        if exception_strategy == "skip_update":
            if min_curvature is not None:
                self.min_curvature = min_curvature
            else:
                self.min_curvature = 1e-8
        elif exception_strategy == "damp_update":
            if min_curvature is not None:
                self.min_curvature = min_curvature
            else:
                self.min_curvature = 0.2
        else:
            raise ValueError("`exception_strategy` must be \"skip_update\" "
                             "or \"damp_update\".")

        super().__init__(init_scale)
        self.exception_strategy = exception_strategy
    }
    
    */
    /// <summary>
    /// Update the inverse Hessian matrix.
    /// </summary>
    /// <remarks>
    ///     BFGS update using the formula:
    ///             ``H< -H + ((H* y).T* y + s.T* y) / (s.T* y) ^ 2 * (s* s.T)
    ///                      - 1 / (s.T* y)* ((H* y)* s.T + s* (H* y).T)``
    ///     where ``s = delta_x`` and ``y = delta_grad``. This formula is
    ///     equivalent to(6.17) in [1] _ written in a more efficient way
    ///         for implementation.
    ///     References
    ///         ----------.. [1] Nocedal, Jorge, and Stephen J.Wright. "Numerical optimization"
    ///     Second Edition(2006).
    /// </remarks>
    private void _update_inverse_hessian(double ys, IList<double> Hy, double yHy, IList<double> s)
    {
        Matrix = syr2(-1.0 / ys, s, Hy, a: Matrix);
        Matrix = syr((ys + yHy) / Math.Pow(ys, 2), s, a: Matrix);
    }

    /// <summary>
    /// Update the Hessian matrix.
    /// </summary>
    /// <remarks>
    /// 
    ///         BFGS update using the formula:
    ///             ``B < -B - (B * s) * (B * s).T / s.T * (B * s) + y * y ^ T / s.T * y``
    ///     where ``s`` is short for ``delta_x`` and ``y`` is short
    ///         for ``delta_grad``. Formula(6.19) in [1]
    ///  ----------..
    ///     References
    ///         [1] Nocedal, Jorge, and Stephen J.Wright. "Numerical optimization"
    ///    Second Edition(2006).</remarks>
    private void _update_hessian(double ys, IList<double> Bs, double sBs, IList<double> y)
    {
        Matrix = syr(1.0 / ys, y, a: Matrix);
        Matrix = syr(-1.0 / sBs, Bs, a: Matrix);
    }
    protected override void _update_implementation(IList<double> delta_x, IList<double> delta_grad)
    {
        // Auxiliary variables w and z
        (IList<double> w, IList<double> z) = StdHess ? (delta_x, delta_grad) : (delta_grad, delta_x);

        // Do some common operations
        double wz = w.DotProduct(z);
        IList<double> Mw = dot(w);
       double wMw = Mw.DotProduct(w);
        // Guarantee that wMw > 0 by reinitializing matrix.
        // While this is always true in exact arithmetics,
        // indefinite matrix may appear due to roundoff errors.
        if (wMw <= 0.0)
        {

            double scale = _auto_scale(delta_x, delta_grad);
            // Reinitialize matrix
            Matrix = Identity(Length, scale);
            // Do common operations for new matrix
            Mw = dot(w);
            wMw = Mw.DotProduct(w);
        }
        // Check if curvature condition is violated
        if (wz <= min_curvature * wMw)
        {
            // If the option "skip_update" is set
            // we just skip the update when the condion
            // is violated.
            if (exception_strategy == ExceptionStrategies.skip_update)
                return;
            // If the option "damp_update" is set we
            // interpolate between the actual BFGS
            // result and the unmodified matrix.
            else if (exception_strategy == ExceptionStrategies.damp_update)
            {
                double update_factor = (1 - min_curvature) / (1 - (wz / wMw));

                z = z.Select(z => update_factor * z).Zip(Mw.Select(Mw => (1 - update_factor) * Mw), (a, b) => a + b).ToList();
                wz = MathEx.DotProduct(w, z);
            }
        }
        // Update matrix
        if (StdHess)
            _update_hessian(wz, Mw, wMw, z);
        else
            _update_inverse_hessian(wz, Mw, wMw, z);
    }
}