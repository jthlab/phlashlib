import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phlashlib.loglik import loglik


def test_warn_nogpu(monkeypatch):
    pytest.skip("Skipping test as it causes issues in some environments.")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    sys.modules.pop("phlashlib.loglik", None)
    sys.modules.pop("phlashlib", None)
    with pytest.warns(UserWarning, match="GPU support not available"):
        import phlashlib.loglik


def test_long_data(data_long, pwc):
    loglik(data_long, pwc, jnp.linspace(0, 10, 16), 1e-4, 1e-4, warmup=10)


def test_vmap(data_long, pwc):
    data_batched = jnp.array([data_long] * 5)
    res = jax.vmap(loglik, in_axes=(0,) + (None,) * 6)(
        data_batched, pwc, jnp.linspace(0, 10, 16), 1e-4, 1e-4, 500, 10
    )
    assert res.shape == (5,)


def test_vmap_grad(data_long, pwc):
    data_batched = jnp.array([data_long] * 5)
    res = jax.vmap(jax.grad(loglik, argnums=1), in_axes=(0,) + (None,) * 6)(
        data_batched, pwc, jnp.linspace(0, 10, 16), 1e-4, 1e-4, 500, 10
    )


def test_grad_float64(data_long, pwc):
    jax.config.update("jax_enable_x64", True)
    pwc = pwc._replace(c=pwc.c.astype(jnp.float64), t=pwc.t.astype(jnp.float64))
    jax.grad(loglik, argnums=1)(
        data_long, pwc, jnp.linspace(0, 10, 16, dtype=jnp.float64), 1e-4, 1e-4, 500, 10
    )


def test_scale_invariant(data_long, pwc):
    t = jnp.insert(jnp.geomspace(1e-4, 10, 15), 0, 0.0)
    theta = rho = 1e-4
    ll1 = loglik(data_long, pwc, t, theta, rho, warmup=10)
    pwc_scaled = pwc._replace(c=pwc.c / 1e4, t=pwc.t * 1e4)
    ll2 = loglik(data_long, pwc_scaled, t * 1e4, theta / 1e4, rho / 1e4, warmup=10)
    np.testing.assert_allclose(ll1, ll2, rtol=1e-5)
