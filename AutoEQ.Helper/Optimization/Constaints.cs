using System;
using System.Collections.Generic;

namespace AutoEQ.Core;
public class Constraint : Constraint<double>
{

}

/// <summary>
/// Nonlinear constraint on the variables.
/// The constraint has the general inequality form::
///      lb <= fun(x) <= ub
/// Here the vector of independent variables x is passed as ndarray of shape
/// (n,) and ``fun`` returns a vector with m components.
/// It is possible to use equal bounds to represent an equality constraint or
/// infinite bounds to represent a one-sided constraint.
/// </summary>
public class Constraint<TDouble>
{
    public class Feasability
    {
        public bool UsingOverall { get; private set; }

        private bool? _Overall;
        private List<bool>? _Each;

        public bool? Overall
        {
            get => UsingOverall ? _Overall : null;
            set
            {
                if (value == null)
                {
                    UsingOverall = false;
                    _Overall = null;
                    _Each = new();
                }
                else
                {
                    UsingOverall = true;
                    _Overall = value;
                    _Each = null;
                }
            }
        }

        public bool this[int index]
        {
            get => UsingOverall ? Overall!.Value : _Each[index];
            set
            {
                UsingOverall = false;
                _Each[index] = value;
                if (value == null)
                {
                    _Overall = null;
                }
                else
                {
                    UsingOverall = true;
                    _Overall = value;
                    _Each = null;
                }
            }
        }
    }

    /// <summary>
    /// Lower and upper bounds on the constraint. Each array must have the
    /// shape(m,) or be a scalar, in the latter case a bound will be the same
    /// for all components of the constraint.Use ``np.inf`` with an
    /// appropriate sign to specify a one-sided constraint.
    /// Set components of `lb` and `ub` equal to represent an equality
    /// constraint. Note that you can mix constraints of different types:
    /// interval, one-sided or equality, by setting different components of
    /// `lb` and `ub` as  necessary.
    /// </summary>
    public IList<TDouble> LowerBound { get; init; }
    /// <inheritdoc cref="LowerBound"/>
    public IList<TDouble> UpperBound { get; init; }
    public IList<TDouble> lb { get => LowerBound; init => LowerBound = value; }
    public IList<TDouble> ub { get => UpperBound; init => UpperBound = value; }

}
/// <summary>
/// Nonlinear constraint on the variables.
/// The constraint has the general inequality form::
///      lb <= fun(x) <= ub
/// Here the vector of independent variables x is passed as ndarray of shape
/// (n,) and ``fun`` returns a vector with m components.
/// It is possible to use equal bounds to represent an equality constraint or
/// infinite bounds to represent a one-sided constraint.
/// </summary>
public class NonlinearConstraint<TDouble> : Constraint<TDouble>
{
    /// <summary>
    /// The function defining the constraint. The signature is ``fun(x) -> array_like, shape(m,)
    /// </summary>
    public Func<TDouble, IList<TDouble>> fun { get; init; }
    /// <summary>
    /// {callable,  '2-point', '3-point', 'cs'}, optional
    /// Method of computing the Jacobian matrix(an m-by-n matrix,
    /// where element (i, j) is the partial derivative of f[i] with respect to x[j]).  The keywords
    /// {'2-point', '3-point', 'cs'} select a finite difference scheme for the numerical estimation
    /// A callable must have the following signature:
    /// ``jac(x) -> {ndarray, sparse matrix}, shape(m, n)``.
    /// Default is '2-point'.
    /// </summary>
    public dynamic jac { get; init; } = "2-point";

    /// <summary>
    /// hess : {callable, '2-point', '3-point', 'cs', HessianUpdateStrategy, None}, optional
    ///     Method for computing the Hessian matrix. The keywords
    ///     {'2-point', '3-point', 'cs'} select a finite difference scheme for
    ///     numerical  estimation.  Alternatively, objects implementing
    ///     `HessianUpdateStrategy` interface can be used to approximate the
    ///     Hessian. Currently available implementations are:
    ///         - `BFGS` (default option)
    ///         - `SR1`
    ///     A callable must return the Hessian matrix of ``dot(fun, v)`` and
    ///     must have the following signature:
    ///     ``hess(x, v) -> {LinearOperator, sparse matrix, array_like}, shape (n, n)``.
    ///     Here ``v`` is ndarray with shape (m,) containing Lagrange multipliers.
    /// </summary>
    public dynamic hess { get; init; }
    /// <summary>
    /// keep_feasible : array_like of bool, optional
    ///     Whether to keep the constraint components feasible throughout
    ///     iterations. A single value set this property for all components.
    ///     Default is False. Has no effect for equality constraints.
    /// </summary>
    public IList<bool> keep_feasible { get; init; } = null;
    public bool? keep_feasible_all { get; init; } = false;

    /// <summary>
    /// finite_diff_rel_step: None or array_like, optional
    ///     Relative step size for the finite difference approximation. Default is
    ///     None, which will select a reasonable value automatically depending
    ///     on a finite difference scheme.
    /// </summary>
    public IList<TDouble>? finite_diff_rel_step { get; init; }
    /// <summary>
    /// finite_diff_jac_sparsity: {None, array_like, sparse matrix}, optional
    ///     Defines the sparsity structure of the Jacobian matrix for finite
    ///     difference estimation, its shape must be (m, n). If the Jacobian has
    ///     only few non-zero elements in *each* row, providing the sparsity
    ///     structure will greatly speed up the computations. A zero entry means
    ///     that a corresponding element in the Jacobian is identically zero.
    ///     If provided, forces the use of 'lsmr' trust-region solver.
    ///     If None (default) then dense differencing will be used.
    /// </summary>
    public dynamic finite_diff_jac_sparsity { get; init; }
    /// <summary>
    /// finite_diff_rel_step: None or array_like, optional
    ///     Relative step size for the finite difference approximation. Default is
    ///     None, which will select a reasonable value automatically depending
    ///     on a finite difference scheme.
    /// </summary>
    public IList<TDouble>? hfinite_diff_rel_stepess { get; init; }

    public NonlinearConstraint(Func<TDouble, IList<TDouble>> fun, IList<TDouble> lb, IList<TDouble> ub, hess= BFGS(),
                 finite_diff_jac_sparsity= None)
    {
        this.fun = fun;
        this.lb = lb;
        this.ub = ub;
        this.finite_diff_rel_step = finite_diff_rel_step;
        this.finite_diff_jac_sparsity = finite_diff_jac_sparsity;
        this.jac = jac;
        this.hess = hess;
        this.keep_feasible = keep_feasible;
    }

}
class LinearConstraint :
    """Linear constraint on the variables.
    The constraint has the general inequality form::
        lb <= A.dot(x) <= ub
    Here the vector of independent variables x is passed as ndarray of shape
    (n,) and the matrix A has shape (m, n).
    It is possible to use equal bounds to represent an equality constraint or
    infinite bounds to represent a one-sided constraint.
    Parameters
    ----------
    A : { array_like, sparse matrix}, shape(m, n)
        Matrix defining the constraint.
    lb, ub : array_like, optional
        Lower and upper limits on the constraint. Each array must have the
        shape (m,) or be a scalar, in the latter case a bound will be the same
        for all components of the constraint. Use ``np.inf`` with an
        appropriate sign to specify a one-sided constraint.
        Set components of `lb` and `ub` equal to represent an equality
        constraint. Note that you can mix constraints of different types:
        interval, one - sided or equality, by setting different components of
        `lb` and `ub` as  necessary. Defaults to ``lb = -np.inf``
        and ``ub = np.inf`` (no limits).
    keep_feasible: array_like of bool, optional
       Whether to keep the constraint components feasible throughout
        iterations. A single value set this property for all components.
        Default is False. Has no effect for equality constraints.
    """
    def _input_validation(self):
        if self.A.ndim != 2:
            message = "`A` must have exactly two dimensions."
            raise ValueError(message)

        try:
            shape = self.A.shape[0:1]
            self.lb = np.broadcast_to(self.lb, shape)
            self.ub = np.broadcast_to(self.ub, shape)
            self.keep_feasible = np.broadcast_to(self.keep_feasible, shape)
        except ValueError:
            message = ("`lb`, `ub`, and `keep_feasible` must be broadcastable "
                       "to shape `A.shape[0:1]`")
            raise ValueError(message)

    def __init__(self, A, lb= -np.inf, ub= np.inf, keep_feasible= False):
        if not issparse(A):
            # In some cases, if the constraint is not valid, this emits a
            # VisibleDeprecationWarning about ragged nested sequences
            # before eventually causing an error. `scipy.optimize.milp` would
            # prefer that this just error out immediately so it can handle it
            # rather than concerning the user.
            with catch_warnings():
                simplefilter("error")
                self.A = np.atleast_2d(A).astype(np.float64)
        else:
            self.A = A
        self.lb = np.atleast_1d(lb).astype(np.float64)
        self.ub = np.atleast_1d(ub).astype(np.float64)
        self.keep_feasible = np.atleast_1d(keep_feasible).astype(bool)
        self._input_validation()

    def residual(self, x):
        """
        Calculate the residual between the constraint function and the limits
        For a linear constraint of the form::
            lb <= A@x <= ub
        the lower and upper residuals between ``A @x`` and the limits are values
        ``sl`` and ``sb`` such that::
            lb + sl == A@x == ub - sb
        When all elements of ``sl`` and ``sb`` are positive, all elements of
        the constraint are satisfied; a negative element in ``sl`` or ``sb``
        indicates that the corresponding element of the constraint is not
        satisfied.
        Parameters
        ----------
        x: array_like
            Vector of independent variables
        Returns
        -------
        sl, sb: array - like
            The lower and upper residuals
        """
        return self.A@x - self.lb, self.ub - self.A@x

