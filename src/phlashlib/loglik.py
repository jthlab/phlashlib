"Log-likelihood of sequence data for given IICR curve"

from warnings import warn

import jax.numpy as jnp
from beartype.typing import Callable
from jax import vmap
from jaxtyping import ArrayLike, Float, Int, Int8, Scalar, ScalarLike
from loguru import logger

import phlashlib.hmm as hmm

from .iicr import PiecewiseConstant
from .params import PSMCParams, PSMCParamsType

try:
    from phlashlib.gpu import _gpu_ll as _logloglik
except Exception as e:
    warn("GPU support not available, falling back to CPU implementation.")
    logger.debug("GPU support not available: {}", e)

    def _logloglik(log_pp: PSMCParamsType, data: Int8[ArrayLike, "L"]):
        "log-likelihood of log params"
        pp = jax.tree.map(jnp.exp, log_pp)
        return hmm.forward(pp, data)[1]


def loglik(
    data: Int[ArrayLike, "L"],
    coal_rate: Callable[[Float], Float],
    times: Float[ArrayLike, "T"],
    theta: ScalarLike,
    rho: ScalarLike,
    warmup: int = 500,
    chunk_size: int = None,
) -> Scalar:
    """
    Compute the log-likelihood of the sequence data given the IICR curve.

    Params:
        data: The sequence data as binary vector: 1 for difference, 0 for identity. -1 encodes
            missing data.
        coal_rate: The coalescent rate function.
        times: Time discretization for the psmc HMM.
        theta: The scaled mutation rate.
        rho: The scaled recombination rate.
        warmup: The sequence length to warm up the Markov chain, for parallel computation.
        chunk_size: The chunk size for parallel computation. If None, choose automatically to
            maximize GPU utilization.

    Returns:
        The log-likelihood of the sequence data.
    """
    # chunk the data
    data = jnp.asarray(data).astype(jnp.int8)
    (L,) = data.shape

    if chunk_size is None:
        chunk_size = int(min(10_000, L / 100))

    # approximate coal rate by piecewise constant function
    c = jnp.array([coal_rate(t) for t in times])
    pwc = PiecewiseConstant(t=times, c=c)
    pp = PSMCParams.from_piecewise_const(pwc, theta=theta, rho=rho)

    if chunk_size < warmup or L < warmup:
        return hmm.forward(pp, data)[1]

    # pad data so that it is evenly divisible by chunk_size
    pad = chunk_size - L % chunk_size
    data = jnp.pad(
        data, (0, chunk_size - L % chunk_size), mode="constant", constant_values=-1
    )
    chunks = data.reshape(-1, chunk_size)
    warmups = data[:, -warmup:][:-1]

    # compute initial dist for each chunk
    pps = jax.vmap(lambda w: hmm.forward(pp, w)[1])(warmups)

    # initial dist for chunk0 = pi
    pps = jax.tree.map(lambda x, y: jnp.concatenate((x[None], y)), pp, pps)

    # log parameters and return
    log_pps = jax.tree.map(jnp.log, pps)
    return vmap(_logloglik)(log_pps, chunks).sum()
