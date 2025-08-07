# `phlashlib`

This library computes the log-likelihood of binary heterozygosity sequences under a sequentially Markov coalescent model with recombination.

## API

```python
phlashlib.loglik(
    data: Int[ArrayLike, "L"],
    iicr: IICRCurve,
    times: Float[ArrayLike, "T"],
    theta: ScalarLike,
    rho: ScalarLike,
    warmup: int = 500,
    chunk_size: int = None,
) -> Scalar
```

## Features

- JAX-compatible: `vmap`, `jit`, `grad`, etc.
- GPU-accelerated via `cuda.core.experimental` (fallback to CPU if unavailable)

## Example

```python
from phlashlib.iicr import PiecewiseConstant
from phlashlib.loglik import loglik
import jax.numpy as jnp

times = jnp.array([0.0, 0.1, 0.5, 2.0])
rates = jnp.array([100.0, 10.0, 1.0])
iicr = PiecewiseConstant(t=times[:-1], c=rates)

data = jnp.array([0, 1, 1, 0, -1, 1, 0, 0], dtype=jnp.int8)
theta = 1.5
rho = 0.5

ll = loglik(data, iicr, times, theta, rho)
```

## JAX Use

```python
from jax import jit, vmap, grad

# JIT
f = jit(loglik)

# Vectorized over data
batched_ll = vmap(loglik, in_axes=(0, None, None, None, None))(batch_data, iicr, times, theta, rho)

# Gradient w.r.t. theta
dtheta = grad(loglik, argnums=3)(data, iicr, times, theta, rho)
```

## Notes

- `data`: 1D `int8`, values in `{0, 1, -1}`
- `iicr`: subclass of `IICRCurve`, e.g. `PiecewiseConstant`
- `theta`, `rho`: scalar mutation and recombination rates
- Internally uses a fused CUDA kernel with `jax.custom_vjp` if available
- Fallback implementation uses pure JAX
