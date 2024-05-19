from typing import Callable, ClassVar, TypeAlias
from math import sqrt

import jax
import jax.numpy as jnp

from diffrax._custom_types import Args, BoolScalarLike, DenseInfo, RealScalarLike, VF, Y
from diffrax._solver.base import AbstractAdaptiveSolver
from diffrax._local_interpolation import LocalLinearInterpolation
from diffrax._term import AbstractTerm
from diffrax._solution import RESULTS



_ErrorEstimate: TypeAlias = None
_SolverState: TypeAlias = None

# TODO: This code should use the butcher Tableu class and be a more general framework
#       for fully implicit RK methods as described in
#       https://www.math.uni-frankfurt.de/~harrach/lehre/Numerik_von_Differentialgleichungen.pdf


sqrt2 = sqrt(2)
a  = 1/(2 + sqrt2)
d31 = - (4 + sqrt2) / (2 + sqrt2)
d32 = (6 + sqrt2) / (2 + sqrt2)


class Rosenbrock23(AbstractAdaptiveSolver):
    r"""Rosenbrock 2(3) method.

    The Rosenbrock-Wanner methods are a class of implicit Runge-Kutta methods 
    that are linearly implicit, i.e. the implicit stage equations can be solved 
    with a matrix inversion. A good overview can be found near page 59 of 
    https://www.math.uni-frankfurt.de/~harrach/lehre/Numerik_von_Differentialgleichungen.pdf.

    A commonly used linear-implicit method is the combination of a two-stage $y^{(2)}$ 
    and three-stage $y^{(3)}$ Rosenbrock-Wanner method allowing for embedded error control. 
    This method is known as `ODE23s` in MATLAB. The method is a good choice for stiff problems, 
    and is often used as a default solver in many software packages. 

    """

    term_structure: ClassVar = AbstractTerm
    interpolation_cls: ClassVar[
        Callable[..., LocalLinearInterpolation]
    ] = LocalLinearInterpolation

    def order(self, terms):
        return 2

    def strong_order(self, terms):
        return None

    def init(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> _SolverState:
        return None

    def step(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump

        n  = y0.shape[0]


        f = lambda y: terms.vf(t=t0, y=y, args=args)
        h = terms.contr(t0, t1)

        # linear inverse
        I  = jnp.eye(n)
        J  = jax.jacfwd(f)(y0)

        A  = I - a * h * J
        LU_and_piv = jax.scipy.linalg.lu_factor(A, overwrite_a=True, check_finite=False)

        b1 = f( y0 )
        k1 = jax.scipy.linalg.lu_solve( LU_and_piv, b1 )
        # stage 2
        hJk1 = h * J@k1
        b2 = f( y0 + 0.5 * h * k1 )  - a * hJk1
        k2 = jax.scipy.linalg.lu_solve( LU_and_piv, b2 )
        # stage 3
        hJk2 = h * J@k2
        b3 = f( y0 + h * k2 ) - d31 * hJk1 - d32 * hJk2
        k3 = jax.scipy.linalg.lu_solve( LU_and_piv, b3 )
        # Advance solution
        yip3 = y0 + h/6 * (k1 + 4*k2 + k3)
        yip2 = y0 + h * k2

        dense_info = dict(y0=y0, y1=yip3)
        return yip3, yip3 - yip2, dense_info, None, RESULTS.successful

    def func(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> VF:
        return terms.vf(t0, y0, args)