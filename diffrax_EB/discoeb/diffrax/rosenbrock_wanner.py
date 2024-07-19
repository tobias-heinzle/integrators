
from dataclasses import dataclass, field
from typing import (
    Any,
    Literal,
    TYPE_CHECKING,
    Union,
    TypeAlias,
)

import numpy as np

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax.internal as lxi


import equinox.internal as eqxi
from equinox.internal import ω

import typing

from typing import TYPE_CHECKING, Union
from jaxtyping import (
    Array,
    ArrayLike,
    Bool,
    Float,
    Int,
    PyTree,
    Shaped,
)

from diffrax import RESULTS
from diffrax import AbstractTerm
from diffrax import AbstractAdaptiveSolver

if TYPE_CHECKING:
    BoolScalarLike = Union[bool, Array, jnp.ndarray]
    FloatScalarLike = Union[float, Array, jnp.ndarray]
    IntScalarLike = Union[int, Array, jnp.ndarray]
elif getattr(typing, "GENERATING_DOCUMENTATION", False):
    # Skip the union with Array in docs.
    BoolScalarLike = bool
    FloatScalarLike = float
    IntScalarLike = int

    #
    # Because they appear in our docstrings, we also monkey-patch some non-Diffrax
    # types that have similar defined-in-one-place, exported-in-another behaviour.
    #

    jtu.Partial.__module__ = "jax.tree_util"

else:
    BoolScalarLike = Bool[ArrayLike, ""]
    FloatScalarLike = Float[ArrayLike, ""]
    IntScalarLike = Int[ArrayLike, ""]


RealScalarLike = Union[FloatScalarLike, IntScalarLike]

Y = PyTree[Shaped[ArrayLike, "?*y"], "Y"]
VF = PyTree[Shaped[ArrayLike, "?*vf"], "VF"]
Control = PyTree[Shaped[ArrayLike, "?*control"], "C"]
Args = PyTree[Any]

# from diffrax.diffrax._custom_types import Args, BoolScalarLike, DenseInfo, RealScalarLike, VF, Y
# from diffrax.diffrax._term import AbstractTerm
# from diffrax.diffrax._solution import RESULTS
# from diffrax.diffrax._solver.base import AbstractAdaptiveSolver




if TYPE_CHECKING:
    from typing import ClassVar as AbstractClassVar
else:
    from equinox import AbstractClassVar


Args: TypeAlias = Any
_ErrorEstimate: TypeAlias = None
_SolverState: TypeAlias = None

@dataclass(frozen=True)
class RosenbrockTableau:
    """Tableau for Rosenbrock--Wanner methods. Needs some different parameters than RK,
    since a transformation is used to save some matrix computations.
    See """


    a_lower: tuple[np.ndarray, ...]
    c_lower: tuple[np.ndarray, ...]
    alpha: np.ndarray
    gamma: np.ndarray
    gamma_diagonal: float
    m_weights: np.array


    num_stages: int = field(init=False)

    def __post_init__(self):
        # TODO: further checks for validity!

        # assert self.gamma_sum.ndim == 1
        # for gamma_i in self.gamma_lower:
        #     assert gamma_i.ndim == 1
        # assert self.gamma_sum.shape[0] == len(self.gamma_lower)
        # assert all(i + 1 == gamma_i.shape[0] for i, gamma_i in enumerate(self.gamma_lower))
        # for gamma_row, gamma_row_sum in zip(self.gamma_lower, self.gamma_sum):
        #     assert np.allclose(sum(gamma_row) + self.gamma_diagonal, gamma_row_sum)
        # assert np.allclose(sum(self.b_sol), 1.0)
        # assert np.allclose(sum(self.b_error), 0.0)

       
        object.__setattr__(self, "num_stages", len(self.c_lower) + 1)

class AbstractRosenbrockWanner(AbstractAdaptiveSolver):
    """Abstract base class for linearly implicit Runge--Kutta solvers, also called
    Rosenbrock--Wanner methods.

    Subclasses should specify two class-level attributes. The first is `tableaus`, an
    instance of [`diffrax.RosenbrockWannerTableaus`][]..
    """

    scan_kind: Union[None, Literal["lax", "checkpointed", "bounded"]] = None
    
    tableau: AbstractClassVar[RosenbrockTableau]


    def init(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> _SolverState:
        return None
    
    def func(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> VF:
        return terms.vf(t0, y0, args)

    def step(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, Y, Any, _SolverState, RESULTS]:
        del solver_state, made_jump
        
        n  = y0.shape[0]
        dt = terms.contr(t0, t1)

        num_stages = self.tableau.num_stages
        init_stage_index = 0
        ks = jnp.zeros((num_stages,) + y0.shape, y0.dtype)

        gamma_diagonal = self.tableau.gamma_diagonal
        dtgamma = dt * gamma_diagonal

        jacobian_f = jax.jacfwd(lambda y: terms.vf(t=t0, y=y, args=args))(y0)
        time_derivative_f = jax.jacfwd(lambda t: terms.vf(t=t, y=y0, args=args))(t0)

        W = jnp.eye(n)/dtgamma - jacobian_f

        LU, piv = jax.scipy.linalg.lu_factor(W)


        y0_leaves = jtu.tree_leaves(y0)
        if len(y0_leaves) == 0:
            tableau_dtype = lxi.default_floating_dtype()
        else:
            tableau_dtype = jnp.result_type(*y0_leaves)

        def embed_lower_triangular(lower: tuple[np.ndarray]):
            embedded_lower = np.zeros(
                (num_stages, num_stages), dtype=np.result_type(*lower)
            )
            for i, a_lower_i in enumerate(lower):
                embedded_lower[i + 1, : i + 1] = a_lower_i
            return jnp.asarray(embedded_lower, dtype=tableau_dtype)

        tableau_a_lower = embed_lower_triangular(self.tableau.a_lower)
        tableau_c_lower = embed_lower_triangular(self.tableau.c_lower)/dt
        alpha = jnp.asarray(self.tableau.alpha, dtype=jnp.result_type(t0, t1))
        gamma = jnp.asarray(self.tableau.gamma, dtype=tableau_dtype)*dt


        for stage_index in range(num_stages):
            a_lower_i = tableau_a_lower[stage_index]
            c_lower_i = tableau_c_lower[stage_index]
            alpha_i = alpha[stage_index]
            gamma_i = gamma[stage_index]


            u = y0 + a_lower_i.T @ ks[:,:]
            du = terms.vf(t=t0 + alpha_i*dt, y=u, args=args)
            rhs = du + gamma_i*time_derivative_f + jnp.einsum('i,ij', c_lower_i, ks[:,:])
            
            ki = jax.scipy.linalg.lu_solve( (LU, piv), rhs)
            ks = ks.at[stage_index].set(ki)


        # def rosenbrock_stage(val):
        #     stage_index, ks, result = val


        #     a_lower_i = tableau_a_lower[stage_index]
        #     c_lower_i = tableau_c_lower[stage_index]
        #     alpha_i = alpha[stage_index]
        #     gamma_i = gamma[stage_index]


        #     u = y0 + a_lower_i.T @ ks[:,:]
        #     du = terms.vf(t=t0 + alpha_i*dt, y=u, args=args)
        #     rhs = du + gamma_i*time_derivative_f + jnp.einsum('i,ij', c_lower_i, ks[:,:])
            
        #     ki = jax.scipy.linalg.lu_solve( (LU, piv), rhs)
        #     ks = ks.at[stage_index].set(ki)


        #     return (
        #          stage_index + 1,
        #          ks,
        #          result
        #     )


        # init_val = (
        #         init_stage_index,
        #         ks,
        #         RESULTS.successful,
        #     )        

        # final_val = eqxi.while_loop(
        #     lambda val: val[0] < num_stages,
        #     rosenbrock_stage,
        #     init_val,
        #     max_steps=num_stages,
        #     kind="bounded",# "checkpointed" if self.scan_kind is None else self.scan_kind,
        #     checkpoints=num_stages,
        #     base=num_stages,
        # )

        # _, ks_final, result = final_val

        y1 = y0 + self.tableau.m_weights @ ks#ks_final

        dense_info = dict(y0=y0, y1=y1)

        return y1, ks[-1], dense_info, None, RESULTS.successful

        


        
