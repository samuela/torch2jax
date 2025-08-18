import jax.numpy as jnp
import torch

from torch2jax import j2t, t2j


def test_t2j_dtype():
  assert t2j(torch.float16) == jnp.float16
  assert t2j(torch.float32) == jnp.float32
  assert t2j(torch.float64) == jnp.float64
  assert t2j(torch.int8) == jnp.int8
  assert t2j(torch.int16) == jnp.int16
  assert t2j(torch.int32) == jnp.int32
  assert t2j(torch.int64) == jnp.int64
  assert t2j(torch.uint8) == jnp.uint8
  assert t2j(torch.bool) == jnp.bool_
  assert t2j(torch.complex64) == jnp.complex64
  assert t2j(torch.complex128) == jnp.complex128
  assert t2j(torch.bfloat16) == jnp.bfloat16


def test_j2t_dtype_constructors():
  assert j2t(jnp.float16) == torch.float16
  assert j2t(jnp.float32) == torch.float32
  assert j2t(jnp.float64) == torch.float64
  assert j2t(jnp.int8) == torch.int8
  assert j2t(jnp.int16) == torch.int16
  assert j2t(jnp.int32) == torch.int32
  assert j2t(jnp.int64) == torch.int64
  assert j2t(jnp.uint8) == torch.uint8
  assert j2t(jnp.bool_) == torch.bool
  assert j2t(jnp.complex64) == torch.complex64
  assert j2t(jnp.complex128) == torch.complex128
  assert j2t(jnp.bfloat16) == torch.bfloat16


def test_j2t_dtypes():
  assert j2t(jnp.dtype("float16")) == torch.float16
  assert j2t(jnp.dtype("float32")) == torch.float32
  assert j2t(jnp.dtype("float64")) == torch.float64
  assert j2t(jnp.dtype("int8")) == torch.int8
  assert j2t(jnp.dtype("int16")) == torch.int16
  assert j2t(jnp.dtype("int32")) == torch.int32
  assert j2t(jnp.dtype("int64")) == torch.int64
  assert j2t(jnp.dtype("uint8")) == torch.uint8
  assert j2t(jnp.dtype("bool_")) == torch.bool
  assert j2t(jnp.dtype("complex64")) == torch.complex64
  assert j2t(jnp.dtype("complex128")) == torch.complex128
  assert j2t(jnp.dtype("bfloat16")) == torch.bfloat16


def test_t2j_array_conversion():
  assert t2j(torch.tensor(0.0, dtype=torch.float16)).dtype == jnp.float16
  assert t2j(torch.tensor(0.0, dtype=torch.float32)).dtype == jnp.float32
  # assert t2j(torch.tensor(0.0, dtype=torch.float64)).dtype == jnp.float64  # requires jax_enable_x64
  assert t2j(torch.tensor(0, dtype=torch.int8)).dtype == jnp.int8
  assert t2j(torch.tensor(0, dtype=torch.int16)).dtype == jnp.int16
  assert t2j(torch.tensor(0, dtype=torch.int32)).dtype == jnp.int32
  # assert t2j(torch.tensor(0, dtype=torch.int64)).dtype == jnp.int64  # requires jax_enable_x64
  assert t2j(torch.tensor(0, dtype=torch.uint8)).dtype == jnp.uint8
  assert t2j(torch.tensor(False, dtype=torch.bool)).dtype == jnp.bool_
  assert t2j(torch.tensor(0.0, dtype=torch.complex64)).dtype == jnp.complex64
  # assert t2j(torch.tensor(0.0, dtype=torch.complex128)).dtype == jnp.complex128  # requires jax_enable_x64
  assert t2j(torch.tensor(0.0, dtype=torch.bfloat16)).dtype == jnp.bfloat16


def test_j2t_array_conversion():
  assert j2t(jnp.array(0.0, dtype=jnp.float16)).dtype == torch.float16
  assert j2t(jnp.array(0.0, dtype=jnp.float32)).dtype == torch.float32
  # assert j2t(jnp.array(0.0, dtype=jnp.float64)).dtype == torch.float64  # requires jax_enable_x64
  assert j2t(jnp.array(0, dtype=jnp.int8)).dtype == torch.int8
  assert j2t(jnp.array(0, dtype=jnp.int16)).dtype == torch.int16
  assert j2t(jnp.array(0, dtype=jnp.int32)).dtype == torch.int32
  # assert j2t(jnp.array(0, dtype=jnp.int64)).dtype == torch.int64  # requires jax_enable_x64
  assert j2t(jnp.array(0, dtype=jnp.uint8)).dtype == torch.uint8
  assert j2t(jnp.array(False, dtype=jnp.bool_)).dtype == torch.bool
  assert j2t(jnp.array(0.0, dtype=jnp.complex64)).dtype == torch.complex64
  # assert j2t(jnp.array(0.0, dtype=jnp.complex128)).dtype == torch.complex128  # requires jax_enable_x64
  assert j2t(jnp.array(0.0, dtype=jnp.bfloat16)).dtype == torch.bfloat16
