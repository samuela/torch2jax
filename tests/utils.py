import inspect

import chex
import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax import grad, jit, random

from torch2jax import HANDLED_FUNCTIONS, j2t, t2j


def aac(tree_a, tree_b, **kwargs):
  chex.assert_trees_all_close(tree_a, tree_b, **kwargs)


def anac(*args, **kwargs):
  """Assert not all close"""
  fail = False
  try:
    np.testing.assert_allclose(*args, **kwargs)
    fail = True
  except AssertionError:
    return
  if fail:
    raise AssertionError("Should not be close")


def test_t2j_array():
  # See https://github.com/samuela/torch2jax/issues/7
  aac(t2j(torch.eye(3).unsqueeze(0)), jnp.eye(3)[jnp.newaxis, ...])


def out_kwarg_test(f, args, kwargs={}, **assert_kwargs):
  torch_kwargs = jax.tree_util.tree_map(lambda x: j2t(x) if isinstance(x, jnp.dtype) else x, kwargs)

  def torch_function(*torch_args):
    out1 = f(*torch_args, **torch_kwargs)
    out2 = torch.zeros_like(out1)
    out3 = f(*torch_args, out=out2, **torch_kwargs)
    return out1, out2, out3

  jax_function = t2j(torch_function)
  jax_out1, jax_out2, jax_out3 = jax_function(*args)
  assert jax_out2 is jax_out3
  aac(jax_out1, jax_out3, **assert_kwargs)


def Torchish_member_test(f, args, kwargs={}, inplace=False, **assert_kwargs):
  name = f.__name__ if not inplace else f"{f.__name__}_"
  torch_kwargs = jax.tree_util.tree_map(lambda x: j2t(x) if isinstance(x, jnp.dtype) else x, kwargs)

  def torch_function(*torch_args):
    out1 = f(*torch_args, **torch_kwargs)
    out2 = torch_args[0].clone()
    out3 = getattr(out2, name)(*torch_args[1:], **torch_kwargs)
    return out1, out2, out3

  jax_function = t2j(torch_function)
  jax_out1, jax_out2, jax_out3 = jax_function(*args)
  if inplace:
    assert jax_out2 is jax_out3
  aac(jax_out1, jax_out3, **assert_kwargs)


def args_generator(shapes, samplers=None, rng=random.PRNGKey(123)):
  n_inputs = len(shapes)
  if samplers is None:
    samplers = [random.normal] * n_inputs
  while True:
    rng, rng1 = random.split(rng)
    args = [sampler(rng, shape=shape) for rng, shape, sampler in zip(random.split(rng1, n_inputs), shapes, samplers)]
    yield args


def forward_test(f, args, kwargs={}, **assert_kwargs):
  torch_args = jax.tree_util.tree_map(lambda x: j2t(x.copy()) if isinstance(x, (jnp.ndarray, jnp.dtype)) else x, args)
  torch_kwargs = jax.tree_util.tree_map(lambda x: j2t(x) if isinstance(x, jnp.dtype) else x, kwargs)
  f_ = lambda *args, **kwargs: f(*args, **torch_kwargs)
  torch_output = f_(*torch_args)
  aac(t2j(f_)(*args), torch_output, **assert_kwargs)
  aac(jit(t2j(f_))(*args), torch_output, **assert_kwargs)
  return torch_output


def backward_test(f, args, kwargs={}, grad_argnums=None, **assert_kwargs):
  n_inputs = len(args)
  torch_args = jax.tree_util.tree_map(lambda x: j2t(x.copy()) if isinstance(x, (jnp.ndarray, jnp.dtype)) else x, args)
  torch_kwargs = jax.tree_util.tree_map(lambda x: j2t(x) if isinstance(x, jnp.dtype) else x, kwargs)
  # TODO: consider doing this for all functions by doing eg f_ = lambda x: torch.sum(f(x) ** 2)
  # Can only calculate gradients on scalar-output functions
  f_ = lambda *args: f(*args, **torch_kwargs).flatten()[0]
  argnums = grad_argnums if grad_argnums is not None else tuple(range(n_inputs))
  if len(argnums) == 0:
    return
  for t2j_grad, torch_grad in zip(
    grad(t2j(f_), argnums=argnums)(*args),
    torch.func.grad(f_, argnums=argnums)(*torch_args),
  ):
    aac(t2j_grad.squeeze(), torch_grad.squeeze(), **assert_kwargs)


def t2j_function_test(
  f, args_shapes, kwargs={}, samplers=None, rng=random.PRNGKey(123), grad_argnums=None, num_tests=5, **assert_kwargs
):
  generator = args_generator(args_shapes, samplers, rng)
  for i in range(num_tests):
    args = next(generator)
    torch_output = forward_test(f, args, kwargs, **assert_kwargs)
    if isinstance(torch_output, torch.Tensor) and torch_output.numel() == 1 and len(args) > 0:
      backward_test(f, args, kwargs, grad_argnums=grad_argnums, **assert_kwargs)

    if f in HANDLED_FUNCTIONS:
      # we check whether "out" is the function's argument
      sig = inspect.signature(HANDLED_FUNCTIONS[f], follow_wrapped=False)
      if "out" in sig.parameters:
        out_kwarg_test(f, args, kwargs, **assert_kwargs)

    if hasattr(f, "__name__"):
      if hasattr(torch.Tensor, f.__name__):
        Torchish_member_test(f, args, kwargs, **assert_kwargs)
      if hasattr(torch.Tensor, f"{f.__name__}_"):
        Torchish_member_test(f, args, kwargs, inplace=True, **assert_kwargs)


def assert_state_dicts_allclose(actual, desired, **kwargs):
  assert sorted(actual.keys()) == sorted(desired.keys())
  for k in actual.keys():
    aac(actual[k], desired[k], **kwargs)
