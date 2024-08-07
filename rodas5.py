from typing import ClassVar, Union, Any, Callable
from typing_extensions import TypeAlias

import equinox.internal as eqxi
import jax.numpy as jnp
import jax

from jaxtyping import (
    Array,
    ArrayLike,
    PyTree,
    Shaped,
)

from diffrax import AbstractTerm
from diffrax import LocalLinearInterpolation
from diffrax import RESULTS
from diffrax import AbstractAdaptiveSolver

BoolScalarLike = Union[bool, Array, jnp.ndarray]
FloatScalarLike = Union[float, Array, jnp.ndarray]
IntScalarLike = Union[int, Array, jnp.ndarray]


RealScalarLike = Union[FloatScalarLike, IntScalarLike]

Y = PyTree[Shaped[ArrayLike, "?*y"], "Y"]
VF = PyTree[Shaped[ArrayLike, "?*vf"], "VF"]
Control = PyTree[Shaped[ArrayLike, "?*control"], "C"]
Args = PyTree[Any]

DenseInfo = dict[str, PyTree[Array]]
DenseInfos = dict[str, PyTree[Shaped[Array, "times-1 ..."]]]
BufferDenseInfos = dict[str, PyTree[eqxi.MaybeBuffer[Shaped[Array, "times ..."]]]]
sentinel: Any = eqxi.doc_repr(object(), "sentinel")



_SolverState: TypeAlias = None

# Efficient implementation with no matrix multiplication, as outlined in Hairer & Wanner

gamma = .19
a21 = 2.0
a31 = 3.040894194418781 
a32 = 1.041747909077569 
a41 = 2.576417536461461 
a42 = 1.622083060776640 
a43 = -.9089668560264532
a51 = 2.760842080225597 
a52 = 1.446624659844071 
a53 = -.3036980084553738
a54 = .2877498600325443 
a61 = -14.09640773051259
a62 = 6.925207756232704 
a63 = -41.47510893210728
a64 = 2.343771018586405 
a65 = 24.13215229196062 
a71 = -14.09640773051259
a72 = 6.925207756232704
a73 = -41.47510893210728
a74 = 2.343771018586405
a75 = 24.13215229196062
a76 = 1.0
a81 = -14.09640773051259
a82 = 6.925207756232704
a83 = -41.47510893210728
a84 = 2.343771018586405
a85 = 24.13215229196062
a86 = 1.0
a87 = 1.0
C21 = -10.31323885133993
C31 = -21.04823117650003
C32 = -7.234992135176716
C41 = 32.22751541853323 
C42 = -4.943732386540191
C43 = 19.44922031041879 
C51 = -20.69865579590063
C52 = -8.816374604402768
C53 = 1.260436877740897 
C54 = -.7495647613787146
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
c2 = 0.38              
c3 = 0.3878509998321533
c4 = 0.4839718937873840
c5 = 0.4570477008819580
d1 = gamma                 
d2 = -0.1823079225333714636
d3 = -0.319231832186874912 
d4 =  0.3449828624725343   
d5 = -0.377417564392089818 




_ErrorEstimate: TypeAlias = None
_SolverState: TypeAlias = None


class Rodas5(AbstractAdaptiveSolver):
    r"""Rodas5 method.
    """

    term_structure: ClassVar = AbstractTerm
    interpolation_cls: ClassVar[
         Callable[..., LocalLinearInterpolation]
    ] = LocalLinearInterpolation

    def order(self, terms):
        return 5

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
        dt = terms.contr(t0, t1)

        # Precalculations
        dtC21 = C21/dt
        dtC31 = C31/dt
        dtC32 = C32/dt
        dtC41 = C41/dt
        dtC42 = C42/dt
        dtC43 = C43/dt
        dtC51 = C51/dt
        dtC52 = C52/dt
        dtC53 = C53/dt
        dtC54 = C54/dt
        dtC61 = C61/dt
        dtC62 = C62/dt
        dtC63 = C63/dt
        dtC64 = C64/dt
        dtC65 = C65/dt
        dtC71 = C71/dt
        dtC72 = C72/dt
        dtC73 = C73/dt
        dtC74 = C74/dt
        dtC75 = C75/dt
        dtC76 = C76/dt
        dtC81 = C81/dt
        dtC82 = C82/dt
        dtC83 = C83/dt
        dtC84 = C84/dt
        dtC85 = C85/dt
        dtC86 = C86/dt
        dtC87 = C87/dt

        dtd1 = dt*d1
        dtd2 = dt*d2
        dtd3 = dt*d3
        dtd4 = dt*d4
        dtd5 = dt*d5
        dtgamma = dt*gamma

        # calculate and invert W
        I  = jnp.eye(n)
        J  = jax.jacfwd(lambda y: terms.vf(t=t0, y=y, args=args))(y0)

        W  = I/dtgamma - J

        LU, piv = jax.scipy.linalg.lu_factor(W)

        # time derivative of vector field
        dT = jax.jacfwd(lambda t: terms.vf(t=t, y=y0, args=args))(t0)

        dy1 = terms.vf(t=t0, y=y0, args=args)
        rhs = dy1 + dtd1*dT
        k1 = jax.scipy.linalg.lu_solve( (LU, piv), rhs )

        u = y0 + a21*k1
        du = terms.vf(t=t0 + c2*dt, y=u, args=args)
        rhs = du + dtd2*dT + dtC21*k1
        k2 = jax.scipy.linalg.lu_solve( (LU, piv), rhs )

        u = y0 + a31*k1 + a32*k2
        du = terms.vf(t=t0 + c3*dt, y=u, args=args)
        rhs = du + dtd3*dT + (dtC31*k1 + dtC32*k2)
        k3 = jax.scipy.linalg.lu_solve( (LU, piv), rhs )

        u = y0 + a41*k1 + a42*k2 + a43*k3
        du = terms.vf(t=t0 + c4*dt, y=u, args=args)
        rhs = du + dtd4*dT + (dtC41*k1 + dtC42*k2 + dtC43*k3)
        k4 = jax.scipy.linalg.lu_solve( (LU, piv), rhs )

        u = y0 + a51*k1 + a52*k2 + a53*k3 + a54*k4
        du = terms.vf(t=t0 + c5*dt, y=u, args=args)
        rhs = du + dtd5*dT + (dtC51*k1 + dtC52*k2 + dtC53*k3 + dtC54*k4)
        k5 = jax.scipy.linalg.lu_solve( (LU, piv), rhs )

        u = y0 + a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5
        du = terms.vf(t=t0 + dt, y=u, args=args)
        rhs = du + (dtC61*k1 + dtC62*k2 + dtC63*k3 + dtC64*k4 + dtC65*k5)
        k6 = jax.scipy.linalg.lu_solve( (LU, piv), rhs )

        u = u + k6 
        du = terms.vf(t=t0 + dt, y=u, args=args)
        rhs = du + (dtC71*k1 + dtC72*k2 + dtC73*k3 + dtC74*k4 + dtC75*k5 + dtC76*k6)
        k7 = jax.scipy.linalg.lu_solve( (LU, piv), rhs)

        u = u + k7 
        du = terms.vf(t=t0 + dt, y=u, args=args)
        rhs = du + (dtC81*k1 + dtC82*k2 + dtC83*k3 + dtC84*k4 + dtC85*k5 + dtC86*k6 + dtC87*k7)
        k8 = jax.scipy.linalg.lu_solve( (LU, piv), rhs)

        y1 = u + k8
        du = terms.vf(t=t0 + dt, y=u, args=args)



        dense_info = dict(y0=y0, y1=y1 ) # for cubic spine inetrpolator:, k0=dy1, k1=du)
        return y1, k8, dense_info, None, RESULTS.successful

    def func(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> VF:
        return terms.vf(t0, y0, args)