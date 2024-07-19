from typing import Callable, ClassVar
import numpy as np

from diffrax import LocalLinearInterpolation
from diffrax import RESULTS
from diffrax import AbstractTerm
from diffrax import AbstractAdaptiveSolver


from .rosenbrock_wanner import RosenbrockTableau, AbstractRosenbrockWanner

γ_diagonal = .19
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
c21 = -10.31323885133993
c31 = -21.04823117650003
c32 = -7.234992135176716
c41 = 32.22751541853323 
c42 = -4.943732386540191
c43 = 19.44922031041879 
c51 = -20.69865579590063
c52 = -8.816374604402768
c53 = 1.260436877740897 
c54 = -.7495647613787146
c61 = -46.22004352711257
c62 = -17.49534862857472
c63 = -289.6389582892057
c64 = 93.60855400400906 
c65 = 318.3822534212147 
c71 = 34.20013733472935 
c72 = -14.15535402717690
c73 = 57.82335640988400 
c74 = 25.83362985412365 
c75 = 1.408950972071624 
c76 = -6.551835421242162
c81 = 42.57076742291101 
c82 = -13.80770672017997
c83 = 93.98938432427124 
c84 = 18.77919633714503 
c85 = -31.58359187223370
c86 = -6.685968952921985
c87 = -5.810979938412932
α1 = 0.0
α2 = 0.38              
α3 = 0.3878509998321533
α4 = 0.4839718937873840
α5 = 0.4570477008819580
α6 = 1.0
α7 = 1.0
α8 = 1.0
γ1 = γ_diagonal               
γ2 = -0.1823079225333714636
γ3 = -0.319231832186874912 
γ4 =  0.3449828624725343   
γ5 = -0.377417564392089818
γ6 = 0.0
γ7 = 0.0
γ8 = 0.0

m1 = -14.09640773051259
m2 = 6.925207756232704
m3 = -41.47510893210728
m4 = 2.343771018586405
m5 = 24.13215229196062
m6 = 1.0
m7 = 1.0
m8 = 1.0

_rodas5_tableau = RosenbrockTableau(
    a_lower = (
        np.array([a21]),
        np.array([a31, a32]),
        np.array([a41, a42, a43]),
        np.array([a51, a52, a53, a54]),
        np.array([a61, a62, a63, a64, a65]),
        np.array([a71, a72, a73, a74, a75, a76]),
        np.array([a81, a82, a83, a84, a85, a86, a87]),
    ),
    c_lower = (
        np.array([c21]),
        np.array([c31, c32]),
        np.array([c41, c42, c43]),
        np.array([c51, c52, c53, c54]),
        np.array([c61, c62, c63, c64, c65]),
        np.array([c71, c72, c73, c74, c75, c76]),
        np.array([c81, c82, c83, c84, c85, c86, c87]),
    ),
    alpha = np.array([α1, α2, α3, α4 ,α5, α6, α7, α8 ]),
    gamma = np.array([γ1, γ2, γ3, γ4, γ5, γ6, γ7, γ8]),
    gamma_diagonal=γ_diagonal,
    m_weights = np.array([m1, m2, m3, m4, m5, m6, m7, m8]),

)


class Rodas5Generic(AbstractRosenbrockWanner):
    term_structure: ClassVar = AbstractTerm
    interpolation_cls: ClassVar[
         Callable[..., LocalLinearInterpolation]
    ] = LocalLinearInterpolation

    tableau: ClassVar[RosenbrockTableau] = _rodas5_tableau

    def order(self, terms):
        return 5