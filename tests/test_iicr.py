import os.path
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from pytest import fixture
from scipy.integrate import quad

from phlashlib.iicr import PiecewiseConstant


@fixture(params=range(3))
def rng(request):
    return np.random.default_rng(request.param)


@fixture
def rand_eta(rng):
    return PiecewiseConstant(
        c=rng.exponential(1.0, size=(5,)),
        t=np.insert(rng.exponential(1.0, size=(4,)).cumsum(), 0, 0.0),
    )


def test_R_quad(rng, rand_eta):
    eta = rand_eta
    t = rng.exponential(1.0)
    R_quad, _ = quad(eta, 0, t, points=eta.t)
    np.testing.assert_allclose(R_quad, eta.R(t), rtol=1e-4)


def test_ect_quad(rng, rand_eta):
    eta = rand_eta
    s = rng.uniform(0.0, eta.t[-1])
    t = rng.uniform(s, eta.t[-1])
    p_coal = np.exp(-eta.R(s)) - np.exp(-eta.R(t))
    dens_quad, err = quad(
        lambda x: eta(x) * np.exp(-eta.R(x)) / p_coal, s, t, points=eta.t
    )
    np.testing.assert_allclose(dens_quad, 1.0, atol=5 * err)
    ects = eta.ect()
    assert np.all(eta.t < ects)
    assert np.all(ects[:-1] < ects[1:])
    assert np.all(ects[:-1] < eta.t[1:])
    np.testing.assert_allclose(ects[-1], eta.t[-1] + 1 / eta.c[-1])
    for ect, s, t, p_coal in zip(ects[:-1], eta.t[:-1], eta.t[1:], eta.pi[:-1]):
        ect_quad, _ = quad(
            lambda x: x * eta(x) * np.exp(-eta.R(x)) / p_coal, s, t, points=eta.t
        )
        np.testing.assert_allclose(ect, ect_quad, rtol=1e-4)
