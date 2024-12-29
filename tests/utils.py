import jax.numpy as jnp
import numpy as np
import torch
from jax import grad, jit, random

from torch2jax import j2t, t2j

aac = np.testing.assert_allclose


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


def t2j_function_test(f, input_shapes, rng=random.PRNGKey(123), num_tests=5, **assert_kwargs):
  for test_rng in random.split(rng, num_tests):
    inputs = [random.normal(rng, shape) for rng, shape in zip(random.split(test_rng, len(input_shapes)), input_shapes)]
    torch_output = f(*map(j2t, inputs))
    aac(t2j(f)(*inputs), torch_output, **assert_kwargs)
    aac(jit(t2j(f))(*inputs), torch_output, **assert_kwargs)

    # TODO: consider doing this for all functions by doing eg f_ = lambda x: torch.sum(f(x) ** 2)
    # Can only calculate gradients on scalar-output functions
    if torch_output.numel() == 1 and len(inputs) > 0:
      f_ = lambda x: f(x).flatten()[0]

      # Branching is necessary to avoid `TypeError: iteration over a 0-d array` in the zip.
      if len(input_shapes) > 1:
        map(
          lambda x, y: aac(x.squeeze(), y.squeeze(), **assert_kwargs),
          zip(
            grad(t2j(f_))(*inputs),
            torch.func.grad(f_, argnums=tuple(range(len(input_shapes))))(*map(j2t, inputs)),
          ),
        )
      else:
        [input] = inputs
        aac(
          grad(t2j(f_))(input).squeeze(),
          torch.func.grad(f_)(j2t(input)).squeeze(),
          **assert_kwargs,
        )
