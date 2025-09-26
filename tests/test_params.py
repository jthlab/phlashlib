import jax
import numpy as np

from phlashlib.params import PSMCParams


def test_scale_invariant(pwc):
    N = 1e4
    theta = rho = 1e-4
    pp1 = PSMCParams.from_piecewise_const(pwc, theta=theta, rho=rho)
    pwc_scaled = pwc._replace(c=pwc.c / N, t=pwc.t * N)
    pp2 = PSMCParams.from_piecewise_const(pwc_scaled, theta=theta / N, rho=rho / N)
    jax.tree.map(np.testing.assert_allclose, pp1, pp2)
