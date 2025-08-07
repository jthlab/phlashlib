import operator
from functools import lru_cache, reduce
from pathlib import Path

import cupy as cp
import jax
import jax.numpy as jnp
import numpy as np
from cuda.core.experimental import (
    Device,
    LaunchConfig,
    ObjectCode,
    Program,
    ProgramOptions,
    launch,
)
from jax import custom_vjp
from jaxtyping import (
    Array,
    ArrayLike,
    Bool,
    Float,
    Int,
    Int8,
    PyTree,
    Scalar,
    ScalarLike,
)
from loguru import logger

import phlashlib.hmm as hmm
from phlashlib.params import PSMCParams, PSMCParamsType

dev = Device()
dev.set_current()


@lru_cache(None)
def _compile(M: int, float32: bool) -> ObjectCode:
    kern_path = Path(__file__).parent / "kernel.cu"
    code = kern_path.read_text()
    if float32:
        code = f"#define FLOAT float\n#define M {M}\n" + code
    else:
        code = f"#define FLOAT double\n#define M {M}\n" + code
    arch = "".join(f"{i}" for i in dev.compute_capability)
    program_options = ProgramOptions(std="c++17", arch=f"sm_{arch}")
    prog = Program(code, code_type="c++", options=program_options)
    return prog.compile("cubin", name_expressions=("loglik_grad",))


_compile(16, True)


def _call_kernel(
    pp: PyTree[Float[ArrayLike, "*batch M"], "PSMCParams"],
    data: Int8[ArrayLike, "*batch L"],
    grad: Bool[ArrayLike, ""],
    float32: Bool[ArrayLike, ""],
) -> (
    Float[ArrayLike, "*batch"]
    | tuple[
        Float[ArrayLike, "*batch"], PyTree[Float[ArrayLike, "*batch M"], "PSMCParams"]
    ]
):
    pa = np.stack(pp, -2)  # [*batch, 7, M]
    M = pa.shape[-1]
    mod = _compile(M, bool(float32))
    L = data.shape[-1]
    batch = data.shape[:-1]
    assert pa.shape[-2:] == (7, M)
    B = reduce(operator.mul, batch, 1)

    # params array
    assert np.isfinite(pa).all(), "not all parameters finite"
    pa = pa.reshape(B, 7, M)

    # allocate host and gpu arrays
    float_type = np.float32 if float32 else np.float64
    cpu_bufs = {
        "ll": np.full([B], np.nan, dtype=float_type),
        "pa": pa.astype(float_type),
        "data": data,
    }
    if grad:
        cpu_bufs["dlog"] = np.full_like(cpu_bufs["pa"], np.nan)

    stream = dev.create_stream()
    cp.cuda.ExternalStream(int(stream.handle)).use()
    gpu_bufs = {}
    for name, cpu_buf in cpu_bufs.items():
        cpu_buf = np.ascontiguousarray(cpu_buf)
        gpu_bufs[name] = cp.asarray(cpu_buf)

    # compute grid sizes
    args = (
        gpu_bufs["data"].data.ptr,
        np.int64(L),
        gpu_bufs["pa"].data.ptr,
        gpu_bufs["ll"].data.ptr,
    )
    grid = (B, 1, 1)
    if grad:
        ker = mod.get_kernel("loglik_grad")
        args += (gpu_bufs["dlog"].data.ptr,)
        block = (7, M, 1)
    else:
        ker = mod.get_kernel("loglik")
        block = (M, 1, 1)

    config = LaunchConfig(grid=grid, block=block)
    launch(stream, config, ker, *args)
    stream.sync()
    ll = cpu_bufs["ll"]
    ll[:] = gpu_bufs["ll"].get()
    # return array needs to be reshaped to match the input batch shape
    # in particlar, if the input is a scalar, we return a scalar.
    ll = ll.reshape(batch)
    ret = ll

    if grad:
        dlog = cpu_bufs["dlog"]
        dlog[:] = gpu_bufs["dlog"].get()
        # we need to reshape dlog to match the input batch shape
        dlog = dlog.reshape(batch + (7, M))
        dll = PSMCParams(
            b=dlog[..., 0, :],
            d=dlog[..., 1, :],
            u=dlog[..., 2, :],
            # we have to roll the last axis around by 1 because v is stored
            # in positions [1, ..., M-1] in the input array.
            v=np.roll(dlog[..., 3, :], 1, axis=-1),
            emis0=dlog[..., 4, :],
            emis1=dlog[..., 5, :],
            pi=dlog[..., 6, :],
        )
        ret = (ll, dll)

    return jax.tree.map(jnp.asarray, ret)


def _gpu_ll_helper(log_pp: PSMCParamsType, data: Int[ArrayLike, "L"]) -> Scalar:
    pp = jax.tree.map(jnp.exp, log_pp)
    return hmm.forward(pp, data)[1]


@custom_vjp
def _gpu_ll(log_pp: PSMCParamsType, data: Int8[ArrayLike, "L"]) -> Scalar:
    return _gpu_ll_helper(log_pp, data)


def _gpu_ll_fwd(
    log_pp: PSMCParamsType, data: Int8[ArrayLike, "L"]
) -> tuple[Scalar, PSMCParamsType]:
    pp = jax.tree.map(jnp.exp, log_pp)
    result_shape_dtype = (
        jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        PSMCParams(
            *[jax.ShapeDtypeStruct(shape=(pp.M,), dtype=jnp.float32) for p in pp],
        ),
    )
    return jax.pure_callback(
        _call_kernel,
        result_shape_dtype,
        pp,
        data,
        True,
        True,
        vmap_method="broadcast_all",
    )


def _gpu_ll_bwd(df: PSMCParamsType, g: ScalarLike) -> tuple[PSMCParamsType, None]:
    return jax.tree.map(lambda a: g * a, df), None


_gpu_ll.defvjp(_gpu_ll_fwd, _gpu_ll_bwd)
