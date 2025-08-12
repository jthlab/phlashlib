import os
import sys

import jax
import jax.numpy as jnp
import pytest


def test_warn_nogpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    sys.modules.pop("phlashlib.loglik", None)
    sys.modules.pop("phlashlib", None)
    with pytest.warns(UserWarning, match="GPU support not available"):
        import phlashlib.loglik


def test_no_chunk_size(data, pwc):
    from phlashlib.loglik import loglik

    loglik(data, pwc, jnp.linspace(0, 10, 16), 1e-4, 1e-4)
