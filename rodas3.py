from typing import Callable, ClassVar, TypeAlias

import numpy as np

import jax
import jax.numpy as jnp

from diffrax._custom_types import Args, BoolScalarLike, DenseInfo, RealScalarLike, VF, Y
from diffrax._solver.base import AbstractAdaptiveSolver
from diffrax._local_interpolation import LocalLinearInterpolation
from diffrax._term import AbstractTerm
from diffrax._solution import RESULTS
from diffrax._solver.runge_kutta import ButcherTableau

import equinox.internal as eqxi


gamma = 1 / 2
a21 = 0
a31 = 2
a32 = 0
a41 = 2
a42 = 0
a43 = 1
C21 = 4
C31 = 1
C32 = -1
C41 = 1
C42 = -1
C43 = -8 / 3
b1 = 2
b2 = 0
b3 = 1
b4 = 1
btilde1 = 0.0
btilde2 = 0.0
btilde3 = 0.0
btilde4 = 1.0
c2 = 0.0
c3 = 1.0
c4 = 1.0
d1 = 1 / 2
d2 = 3 / 2
d3 = 0
d4 = 0

a_lower=(
        np.array([a21]),
        np.array([a31, a32]),
        np.array([a41, a42, a43]),
    )

C_lower = (
        np.array([C21]),
        np.array([C31, C32]),
        np.array([C41, C42, C43]),
    )

b = np.array([b1, b2, b3, b4])

b_tilde = np.array([btilde1, btilde2, btilde3, btilde4])

c = np.array([c2, c3, c4])

d = np.array([d1, d2, d3, d4])


# _rodas3_tableau = ButcherTableau(
#     a_lower=(
#         np.array([a21]),
#         np.array([a31, a32]),
#         np.array([a41, a42, a43]),
#     ),
#     a_diagonal=np.array([0, γ, γ, γ, γ, γ, γ]),

#     b_sol=np.array([a71, a72, a73, a74, a75, a76, γ]),
#     b_error=np.array(
#         [a71 - a61, a72 - a62, a73 - a63, a74 - a64, a75 - a65, a76 - γ, γ]
#     ),
#     c=np.array(
#         [0.52, 1.230333209967908, 0.8957659843500759, 0.43639360985864756, 1.0, 1.0]
#     ),
# )

_ErrorEstimate: TypeAlias = None
_SolverState: TypeAlias = None


class Rodas3(AbstractAdaptiveSolver):
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
        return 3

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
        h = terms.contr(t0, t1)

        # linear inverse
        I  = jnp.eye(n)
        J  = jax.jacfwd(lambda y: terms.vf(t=t0, y=y, args=args))(y0)

        A  = I - gamma * h * J
        LU, piv = jax.scipy.linalg.lu_factor(A, overwrite_a=True, check_finite=False)

        # num_stages = 4

        rhs1 = h*terms.vf(t0, y0, args=args)
        k1 = jax.scipy.linalg.lu_solve( (LU, piv), rhs1 )

        rhs2 = h*terms.vf(t0 + c2 * h, y0 + a21 * k1, args=args) # + h*h* C21 * k1
        k2 = jax.scipy.linalg.lu_solve( (LU, piv), rhs2 )

        rhs3 = h*terms.vf(t0 + c3 * h, y0 + a31 * k1 + a32 * k2, args=args) # + h*h* (C31 * k1 + C32 * k2)
        k3 = jax.scipy.linalg.lu_solve( (LU, piv), rhs3 )

        rhs4 = h*terms.vf(t0 + c4 * h, y0 + a41 * k1 + a42 * k2 + a43 *k3, args=args) # + h*h* (C41 * k1 + C42 * k2 + C43 *k3)
        k4 = jax.scipy.linalg.lu_solve( (LU, piv), rhs4 )

        
        y1 = y0 + b1 * k1 + b2 *k2 + b3 * k3 + b4* k4

        y1_tilde = y0 + btilde1 * k1 + btilde2 *k2 + btilde3 * k3 + btilde4* k4
        
        # # TODO: adapt tu use butcher tableus and equinox while loops
        # final_val = eqxi.while_loop(cond_stage,
        #     implicit_rk_stage,
        #     init_val,
        #     max_steps=num_stages,
        #     buffers=buffers,
        #     kind="checkpointed" if self.scan_kind is None else self.scan_kind,
        #     checkpoints=num_stages,
        #     base=num_stages,
        # )


        dense_info = dict(y0=y0, y1=y1)
        return y1, 0, dense_info, None, RESULTS.successful

    def func(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> VF:
        return terms.vf(t0, y0, args)