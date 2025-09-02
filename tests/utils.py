import chex
import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax import grad, jit, random
from torch.utils._pytree import tree_leaves

from torch2jax import j2t, t2j

aac = chex.assert_trees_all_close


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
  torch_kwargs = jax.tree.map(lambda x: j2t(x) if isinstance(x, jnp.dtype) else x, kwargs)

  def torch_function(*torch_args):
    out1 = f(*torch_args, **torch_kwargs)
    out2 = torch.zeros_like(out1)
    out3 = f(*torch_args, out=out2, **torch_kwargs)
    return out1, out2, out3

  jax_function = t2j(torch_function)
  jax_out1, jax_out2, jax_out3 = jax_function(*args)
  assert jax_out2 is jax_out3
  aac(jax_out1, jax_out3, **assert_kwargs)


def Torchish_member_test(f, args, kwargs={}, **assert_kwargs):
  torch_kwargs = jax.tree.map(lambda x: j2t(x) if isinstance(x, jnp.dtype) else x, kwargs)

  def torch_function(*torch_args, name=f.__name__):
    out1 = f(*torch_args, **torch_kwargs)
    out2 = torch_args[0].clone()
    out3 = getattr(out2, name)(*torch_args[1:], **torch_kwargs)
    return out1, out2, out3

  jax_out1, jax_out2, jax_out3 = t2j(lambda *args: torch_function(*args, name=f.__name__))(*args)
  aac(jax_out1, jax_out3, **assert_kwargs)
  if hasattr(torch.Tensor, f"{f.__name__}_"):
    jax_out1, jax_out2, jax_out3 = t2j(lambda *args: torch_function(*args, name=f"{f.__name__}_"))(*args)
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


def _arg2t(x):
  if isinstance(x, jnp.ndarray):
    # copy to avoid in-place modification
    return j2t(x.copy())
  elif isinstance(x, jnp.dtype):
    return j2t(x)
  elif isinstance(x, np.ndarray):
    # we allow passing numpy arrays directly in test cases
    # it is needed because torch requires int64 for some functions, while jax's default is int32, using numpy allows them to each make their own casts.
    return torch.from_numpy(x)
  else:
    return x


def forward_test(f, args, kwargs={}, test_jit=True, **assert_kwargs):
  torch_args = jax.tree.map(_arg2t, args)
  torch_kwargs = jax.tree.map(_arg2t, kwargs)
  f_ = lambda *args, **kwargs: f(*args, **torch_kwargs)
  torch_output = f_(*torch_args)
  aac(t2j(f_)(*args), torch_output, **assert_kwargs)
  if test_jit:
    aac(jit(t2j(f_))(*args), torch_output, **assert_kwargs)


def backward_test(f, args, kwargs={}, argnums=None, **assert_kwargs):
  n_inputs = len(args)
  argnums = argnums if argnums is not None else tuple(range(n_inputs))
  if n_inputs == 0 or len(argnums) == 0:
    return
  torch_args = jax.tree.map(_arg2t, args)
  torch_kwargs = jax.tree.map(_arg2t, kwargs)
  # always reduce output to the mean of all elements
  f_ = lambda *args: torch.cat(list(map(lambda x: x.flatten(), tree_leaves(f(*args, **torch_kwargs))))).mean()
  for t2j_grad, torch_grad in zip(
    grad(t2j(f_), argnums=argnums)(*args),
    torch.func.grad(f_, argnums=argnums)(*torch_args),
  ):
    aac(t2j_grad.squeeze(), torch_grad.squeeze(), **assert_kwargs)


def t2j_function_test(
  f,
  args_shapes,
  kwargs={},
  samplers=None,
  rng=random.PRNGKey(123),
  num_tests=5,
  tests=[forward_test, backward_test],
  **assert_kwargs,
):
  generator = args_generator(args_shapes, samplers, rng)
  for test in tests:
    for i in range(num_tests):
      args = next(generator)
      test(f, args, kwargs, **assert_kwargs)


def assert_state_dicts_allclose(actual, desired, **kwargs):
  assert sorted(actual.keys()) == sorted(desired.keys())
  for k in actual.keys():
    aac(actual[k], desired[k], **kwargs)
