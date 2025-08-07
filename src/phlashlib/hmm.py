import operator

import jax
import jax.numpy as jnp
from jax import lax, vmap
from jaxtyping import Array, ArrayLike, Float, Int, Scalar
from loguru import logger

from .params import PSMCParams, PSMCParamsType


def _matvec_smc(v: Float[Array, "M"], pp: PSMCParamsType) -> Float[Array, "M"]:
    # v @ A where A is the SMC' transition matrix.
    vr = lax.associative_scan(operator.add, jnp.append(v, 0.0)[1:], reverse=True)
    lower = vr * pp.b

    def f(s, tup):
        ppi, vi = tup
        t = s * ppi.v
        s += ppi.u * vi
        return s, t

    _, upper = lax.scan(f, 0.0, (pp, v))

    return lower + pp.d * v + upper


def forward(
    pp: PSMCParamsType, data: Int[ArrayLike, "L"]
) -> tuple[Float[Array, "M"], Scalar]:
    emis = jnp.stack([pp.emis0, pp.emis1, jnp.ones_like(pp.emis0)])

    def fwd(tup, ob):
        alpha_hat, ll = tup
        alpha_hat = _matvec_smc(alpha_hat, pp)
        alpha_hat *= emis[ob]
        c = alpha_hat.sum()
        ll += jnp.log(c)
        alpha_hat /= c
        return (alpha_hat, ll), None

    init = (pp.pi, 0.0)
    return lax.scan(fwd, init, data)[0]
