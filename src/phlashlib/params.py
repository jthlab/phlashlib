"Different parameterizations needed for MCMC and HMM"

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float, PyTree, ScalarLike

from phlashlib.iicr import PiecewiseConstant
from phlashlib.transition import transition_matrix

PSMCParamsType = PyTree[Float[ArrayLike, "M"], "PSMCParams"]


class PSMCParams(NamedTuple):
    b: Float[Array, "M"]
    d: Float[Array, "M"]
    u: Float[Array, "M"]
    v: Float[Array, "M"]
    emis0: Float[Array, "M"]
    emis1: Float[Array, "M"]
    pi: Float[Array, "M"]

    @property
    def M(self) -> int:
        "The number of discretization intervals"
        M = self.d.shape[-1]
        assert all(a.shape[-1] == M for a in self)
        return M

    @classmethod
    def from_piecewise_const(
        cls, eta: PiecewiseConstant, theta: ScalarLike, rho: ScalarLike
    ) -> "PSMCParams":
        "Initialize parameters from a demographic model"
        t = jnp.append(eta.t, jnp.inf)
        u = 2 * theta * eta.ect()
        emis0 = jnp.exp(-u)
        emis1 = -jnp.expm1(-u)
        pi = eta.pi
        A = transition_matrix(eta, rho)
        pi, A = jax.tree.map(lambda a: a.clip(1e-20, 1.0 - 1e-20), (pi, A))
        b, d, u = (jnp.diag(A, i) for i in [-1, 0, 1])
        v = A[0, 1:] / A[0, 1]
        ut = u / v
        return cls(
            b=jnp.append(b, 0.0),
            d=d,
            u=jnp.append(ut, 0.0),
            v=jnp.insert(v, 0, 0.0),
            emis0=emis0,
            emis1=emis1,
            pi=pi,
        )
