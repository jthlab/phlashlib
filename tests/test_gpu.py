import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads
from pytest import fixture

try:
    import phlashlib.gpu as gpu
except:
    pytest.skip("GPU support not available", allow_module_level=True)
import phlashlib.hmm as hmm
from phlashlib.iicr import PiecewiseConstant
from phlashlib.params import PSMCParams

assert_allclose = lambda x, y: np.testing.assert_allclose(x, y, atol=1e-5, rtol=1e-5)


def test_loglik(pp, data):
    ll_gpu, dlog_gpu = gpu._call_kernel(pp, data, grad=True, float32=True)
    # test loglik vs the pure python implementation
    ll_cpu = hmm.forward(pp, data)[1]
    assert_allclose(ll_gpu, ll_cpu)


def test_grad(log_pp, data):
    ll_cpu, dlog_cpu = jax.value_and_grad(gpu._gpu_ll_helper)(log_pp, data)
    ll_gpu1 = gpu._gpu_ll(log_pp, data)
    ll_gpu2, dlog_gpu = jax.value_and_grad(gpu._gpu_ll)(log_pp, data)
    assert_allclose(ll_gpu1, ll_cpu)
    assert_allclose(ll_gpu2, ll_cpu)
    jax.tree.map(assert_allclose, dlog_gpu, dlog_cpu)


def test_jit(log_pp, data):
    ll1 = gpu._gpu_ll(log_pp, data)
    ll2 = jax.jit(gpu._gpu_ll)(log_pp, data)
    assert_allclose(ll1, ll2)


def test_vmap(pp, log_pp, data):
    log_pp, pp, data = jax.vmap(lambda _: (log_pp, pp, data))(jnp.arange(5))
    ll1 = jax.vmap(gpu._gpu_ll)(log_pp, data)
    _, ll2 = jax.vmap(hmm.forward)(pp, data)
    assert_allclose(ll1, ll2)


def test_staggered_vmap(pp, log_pp, data):
    data = jax.vmap(lambda _: data)(jnp.arange(5))
    ll1 = jax.vmap(gpu._gpu_ll, (None, 0))(log_pp, data)
    _, ll2 = jax.vmap(hmm.forward, (None, 0))(pp, data)
    assert_allclose(ll1, ll2)


def test_check_grads(log_pp, data):
    check_grads(lambda x: gpu._gpu_ll(x, data), args=(log_pp,), order=1, modes=["rev"])
