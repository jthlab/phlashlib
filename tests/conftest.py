import jax
import numpy as np
from pytest import fixture

from phlashlib.iicr import PiecewiseConstant
from phlashlib.params import PSMCParams, PSMCParamsType

jax.config.update("jax_enable_x64", True)


@fixture(params=range(3))
def rng(request):
    """Fixture that provides a random number generator with a fixed seed."""
    return np.random.default_rng(request.param)


@fixture(params=[4, 8, 16])
def M(request):
    """Fixture that provides different values for M."""
    return request.param


@fixture
def pwc(rng, M):
    return PiecewiseConstant(
        t=np.arange(M, dtype=np.float32), c=rng.exponential(1.0, (M,))
    )


@fixture
def pp(pwc):
    mu = 1e-4
    rho = 1e-4
    return PSMCParams.from_piecewise_const(pwc, mu, rho)


@fixture
def log_pp(pp):
    return jax.tree.map(np.log, pp)


@fixture
def data(rng):
    return rng.integers(0, 2, 10).astype(np.int8)


@fixture
def data_long(rng):
    return rng.integers(0, 2, 10_000).astype(np.int8)
