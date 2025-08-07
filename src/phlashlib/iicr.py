import abc
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Float, Scalar, ScalarLike


def _expm1inv(x):
    # 1 / expm1(x)
    x_large = x > 10.0
    x_safe = jnp.where(x_large, 1.0, x)
    # x = id_print(x, what="x")
    return jnp.where(x_large, -jnp.exp(-x) / jnp.expm1(-x), 1.0 / jnp.expm1(x_safe))


class IICRCurve(abc.ABC):
    @abc.abstractmethod
    def __call__(self, t: ScalarLike) -> Scalar:
        "Evaluate the coalescent rate at time points `t`."
        pass

    @abc.abstractmethod
    def ect(self, s: ScalarLike, t: ScalarLike) -> Scalar:
        "Expected coalescent time conditional on coalescing between `s` and `t`."
        pass

    def R(self, t: ScalarLike) -> Scalar:
        "Evaluate the cumulative hazard rate at time points `t`."
        x = jnp.linspace(0.0, t, 1000)
        return jnp.trapezoid(x=x, y=self(x))


@jax.tree_util.register_dataclass
@dataclass
class PiecewiseConstant(IICRCurve):
    "A piecewise constant IICR curve."

    t: Float[ArrayLike, "T"]
    c: Float[ArrayLike, "T"]

    @property
    def T(self):
        return self.t.shape[0]

    def __call__(self, s: ScalarLike) -> ScalarLike:
        i = jnp.searchsorted(jnp.append(self.t, jnp.inf), s, side="right") - 1
        return jnp.take(self.c, i)

    def R(self, s: ScalarLike) -> Scalar:
        ts = jnp.minimum(jnp.append(self.t, jnp.inf), s)
        c_safe = jnp.where(ts[:-1] >= s, 0.0, self.c)
        return jnp.dot(jnp.diff(ts), c_safe)

    @property
    def pi(self) -> Float[ArrayLike, "T"]:
        R = jax.vmap(self.R)(self.t)
        ret = -jnp.diff(jnp.exp(-R))
        ret = jnp.append(ret, 1.0 - ret.sum())
        return ret

    def ect(self) -> Float[ArrayLike, "T"]:
        "Expected time to coalescence within each interval."
        c = self.c[:-1]
        c0 = jnp.isclose(c, 0)
        cinf = jnp.isinf(c) | (c > 100.0)
        c_safe = jnp.where(c0 | cinf, 1.0, c)
        t0 = self.t[:-1]
        t1 = self.t[1:]
        dt = t1 - t0
        # Always have to be careful with exp... NaNs in gradients
        e_coal_safe = 1 / c_safe + t0 - dt * _expm1inv(c_safe * dt)
        # e_coal_safe, *_ = id_print((e_coal_safe, c_safe, dt), what="ect_safe")
        e_coal = jnp.select(
            [c0, cinf],
            [
                (t0 + t1) / 2,
                t0,
            ],
            e_coal_safe,
        )
        e_coal = jnp.append(e_coal, self.t[-1] + 1.0 / self.c[-1])
        # expected coal time of zero messes things up
        e_coal = jnp.maximum(e_coal, 1e-20)
        return e_coal
