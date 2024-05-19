from typing import Callable, ClassVar, TypeAlias
from math import sqrt

import jax
import jax.numpy as jnp

from diffrax._custom_types import Args, BoolScalarLike, DenseInfo, RealScalarLike, VF, Y
from diffrax._solver.base import AbstractAdaptiveSolver
from diffrax._local_interpolation import LocalLinearInterpolation
from diffrax._term import AbstractTerm
from diffrax._solution import RESULTS

gamma = 0.19
a21 = 2.0
a31 = 3.040894194418781
a32 = 1.041747909077569
a41 = 2.576417536461461
a42 = 1.622083060776640
a43 = -0.9089668560264532
a51 = 2.760842080225597
a52 = 1.446624659844071
a53 = -0.3036980084553738
a54 = 0.2877498600325443
a61 = -14.09640773051259
a62 = 6.925207756232704
a63 = -41.47510893210728
a64 = 2.343771018586405
a65 = 24.13215229196062
C21 = -10.31323885133993
C31 = -21.04823117650003
C32 = -7.234992135176716
C41 = 32.22751541853323
C42 = -4.943732386540191
C43 = 19.44922031041879
C51 = -20.69865579590063
C52 = -8.816374604402768
C53 = 1.260436877740897
C54 = -0.7495647613787146
C61 = -46.22004352711257
C62 = -17.49534862857472
C63 = -289.6389582892057
C64 = 93.60855400400906
C65 = 318.3822534212147
C71 = 34.20013733472935
C72 = -14.15535402717690
C73 = 57.82335640988400
C74 = 25.83362985412365
C75 = 1.408950972071624
C76 = -6.551835421242162
C81 = 42.57076742291101
C82 = -13.80770672017997
C83 = 93.98938432427124
C84 = 18.77919633714503
C85 = -31.58359187223370
C86 = -6.685968952921985
C87 = -5.810979938412932
c2 = 20.38
c3 = 20.3878509998321533
c4 = 20.4839718937873840
c5 = 20.4570477008819580
d1 = gamma
d2 = -0.1823079225333714636
d3 = -0.319231832186874912
d4 = 0.3449828624725343
d5 = -0.377417564392089818

h21 = 27.354592673333357
h22 = -6.925207756232857
h23 = 26.40037733258859
h24 = 0.5635230501052979
h25 = -4.699151156849391
h26 = -1.6008677469422725
h27 = -1.5306074446748028
h28 = -1.3929872940716344

h31 = 44.19024239501722
h32 = 1.3677947663381929e-13
h33 = 202.93261852171622
h34 = -35.5669339789154
h35 = -181.91095152160645
h36 = 3.4116351403665033
h37 = 2.5793540257308067
h38 = 2.2435122582734066

h41 = -44.0988150021747
h42 = -5.755396159656812e-13
h43 = -181.26175034586677
h44 = 56.99302194811676
h45 = 183.21182741427398
h46 = -7.480257918273637
h47 = -5.792426076169686
h48 = -5.32503859794143



_ErrorEstimate: TypeAlias = None
_SolverState: TypeAlias = None


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
        return None

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

        A  = I - gamma * h * J
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