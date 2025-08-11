import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest
import torch
from jax import grad, jit, vmap, random

from torch2jax import t2j

from .utils import aac, forward_test, out_kwarg_test, t2j_function_test


def test_arange():
  tests = [forward_test, out_kwarg_test]
  for test in tests:
    test(lambda out=None: torch.arange(10, out=out), [])
    test(lambda out=None: torch.arange(2, 10, out=out), [])
    test(lambda out=None: torch.arange(2, 10, 3, out=out), [])


def test_empty():
  # torch.empty returns uninitialized values, so we need to multiply by 0 for deterministic, testable behavior.
  # NaNs are possible, so we need to convert them first. See
  # https://discuss.pytorch.org/t/torch-empty-returns-nan/181389 and https://github.com/samuela/torch2jax/actions/runs/13348964668/job/37282967463.
  t2j_function_test(lambda: 0 * torch.nan_to_num(torch.empty(())), [])
  t2j_function_test(lambda: 0 * torch.nan_to_num(torch.empty(2)), [])
  t2j_function_test(lambda: 0 * torch.nan_to_num(torch.empty((2, 3))), [])

def test_full():
  tests = [forward_test, out_kwarg_test]
  for test in tests:
    test(lambda out=None: torch.full((), fill_value=1., out=out), [])
    test(lambda out=None: torch.full((2, 3), fill_value=1., out=out), [])

def test_is_floating_point():
  def f(x):
    return torch.is_floating_point(x)
  assert t2j(f)(jnp.zeros((3, 4), dtype=jnp.float32)) == True
  assert t2j(f)(jnp.zeros((3, 4), dtype=jnp.int32)) == False

def test_nan_to_num():
  for value in ["nan", "inf", "-inf"]:
    samplers = lambda rng, shape: jnp.array([float(value), 1.0, 2.0])
    t2j_function_test(torch.nan_to_num, [(3,)], samplers=samplers, num_tests=1)

  # Test handling of all special values with custom replacements
  samplers = lambda rng, shape: jnp.array([float("nan"), float("inf"), float("-inf")])
  t2j_function_test(
    torch.nan_to_num, [(3,)], kwargs=dict(nan=0.0, posinf=1.0, neginf=-1.0), samplers=samplers, num_tests=1
  )


def test_ones():
  tests = [forward_test, out_kwarg_test]
  for test in tests:
    test(lambda out=None: torch.ones((), out=out), [])
    test(lambda out=None: torch.ones(2, out=out), [])
    test(lambda out=None: torch.ones(2, 3, out=out), [])
    test(lambda out=None: torch.ones((2, 3), out=out), [])


def test_ones_like():
  t2j_function_test(torch.ones_like, [()])
  t2j_function_test(torch.ones_like, [(2,)])
  t2j_function_test(torch.ones_like, [(2, 3)])


def test_tensor():
  t2j_function_test(lambda: torch.tensor([]), [])
  t2j_function_test(lambda: torch.tensor([1, 2, 3]), [])
  t2j_function_test(lambda: torch.tensor([[1, 2, 3], [4, 5, 6]]), [])

  # torch allows calling torch.tensor with a torch.Tensor. This gets a little tricky with Torchish.
  t2j_function_test(lambda: torch.tensor(torch.arange(3)), [])


def test_zeros():
  tests = [forward_test, out_kwarg_test]
  for test in tests:
    test(lambda out=None: torch.zeros((), out=out), [])
    test(lambda out=None: torch.zeros(2, out=out), [])
    test(lambda out=None: torch.zeros(2, 3, out=out), [])
    test(lambda out=None: torch.zeros((2, 3), out=out), [])


def test_zeros_like():
  t2j_function_test(torch.zeros_like, [()])
  t2j_function_test(torch.zeros_like, [(2,)])
  t2j_function_test(torch.zeros_like, [(2, 3)])


def test_unbind():
  t2j_function_test(torch.unbind, [(2, 3)])
  t2j_function_test(torch.unbind, [(2, 3)], kwargs={"dim": 1})


def test_oneliners():
  samplers = random.bernoulli
  t2j_function_test(torch.all, [(3, 2)], samplers=samplers, grad_argnums=[])
  t2j_function_test(torch.all, [(3, 2)], samplers=samplers, kwargs=dict(dim=1), grad_argnums=[])
  t2j_function_test(torch.any, [(3, 2)], samplers=samplers, grad_argnums=[])
  t2j_function_test(torch.any, [(3, 2)], samplers=samplers, kwargs=dict(dim=1), grad_argnums=[])

  # bitwise_not on int and bool tensors
  t2j_function_test(torch.bitwise_not, [(3, 2)], samplers=lambda key, shape: random.randint(key, shape, minval=0, maxval=1024))
  t2j_function_test(torch.bitwise_not, [(3, 2)], samplers=random.bernoulli)
  t2j_function_test(torch.cumsum, [(3, 5)], kwargs=dict(dim=1), atol=1e-6)
  t2j_function_test(torch.cumsum, [(3, 5)], kwargs=dict(dim=1), atol=1e-6)

  # isin
  samplers = [lambda key, shape: random.randint(key, shape, minval=0, maxval=2) for _ in range(2)]
  t2j_function_test(torch.isin, [(3, 2), (10,)], samplers=samplers)
  t2j_function_test(torch.isin, [(3, 2), (10,)], samplers=samplers, kwargs=dict(invert=True))

  # logical operations
  t2j_function_test(torch.logical_and, [(3, 2), (3, 2)], samplers=random.bernoulli)
  t2j_function_test(torch.logical_and, [(3, 2), (2)], samplers=random.bernoulli)
  t2j_function_test(torch.logical_and, [(3, 2), (3, 1)], samplers=random.bernoulli)
  t2j_function_test(torch.logical_or, [(3, 2), (3, 2)], samplers=random.bernoulli)
  t2j_function_test(torch.logical_or, [(3, 2), (2)], samplers=random.bernoulli)
  t2j_function_test(torch.logical_or, [(3, 2), (3, 1)], samplers=random.bernoulli)
  t2j_function_test(torch.logical_xor, [(3, 2), (3, 2)], samplers=random.bernoulli)
  t2j_function_test(torch.logical_xor, [(3, 2), (2)], samplers=random.bernoulli)
  t2j_function_test(torch.logical_xor, [(3, 2), (3, 1)], samplers=random.bernoulli)
  t2j_function_test(torch.logical_not, [(3, 2)], samplers=random.bernoulli)
  t2j_function_test(torch.logical_not, [(2)], samplers=random.bernoulli)
  t2j_function_test(torch.logical_not, [(3, 1)], samplers=random.bernoulli)

  # masked_fill
  samplers=[random.normal, random.bernoulli, random.normal]
  t2j_function_test(torch.masked_fill, [(3, 5), (3, 5), ()], samplers=samplers)
  t2j_function_test(torch.masked_fill, [(3, 5), (3, 1), ()], samplers=samplers)
  t2j_function_test(torch.masked_fill, [(3, 5), (5,), ()], samplers=samplers)

  # max, mean
  t2j_function_test(torch.max, [(3, 5)], atol=1e-6)
  # t2j_function_test(torch.max, [(3, 5)], kwargs=dict(dim=1), atol=1e-6)
  # t2j_function_test(torch.max, [(3, 5)], kwargs=dict(dim=1, keepdim=True), atol=1e-6)
  t2j_function_test(torch.mean, [(3, 5)], atol=1e-6)
  t2j_function_test(torch.mean, [(3, 5)], kwargs=dict(dim=1), atol=1e-6)

  t2j_function_test(torch.sigmoid, [(3,)], atol=1e-6)
  t2j_function_test(torch.sigmoid, [(3, 5)], atol=1e-6)
  t2j_function_test(lambda x: torch.softmax(x, 1), [(3, 5)], atol=1e-6)
  t2j_function_test(lambda x: torch.softmax(x, 0), [(3, 5)], atol=1e-6)
  t2j_function_test(lambda x: x.softmax(1), [(3, 5)], atol=1e-6)
  t2j_function_test(lambda x: x.softmax(0), [(3, 5)], atol=1e-6)
  t2j_function_test(torch.squeeze, [(1, 5, 1)], atol=1e-6)
  t2j_function_test(torch.squeeze, [(1, 5, 1)], kwargs=dict(dim=2), atol=1e-6)
  # sort: TODO, out is a tuple
  # topk: TODO, out is a tuple

  # scatter
  index = jnp.array([[0, 1, 2, 0, 2], [1, 0, 0, 2, 1]])
  samplers = [random.normal, lambda key, shape: index, random.normal]
  t2j_function_test(lambda input, index, src: torch.scatter(input, 0, index, src), [(3, 5), (2, 5), (2, 5)], samplers=samplers, atol=1e-6)
  index = jnp.array([[0, 1, 2, 0, 2]])
  samplers = [random.normal, lambda key, shape: index, random.normal]
  t2j_function_test(lambda input, index, src: torch.scatter(input, 0, index, src), [(3, 5), (1, 5), (2, 5)], samplers=samplers, atol=1e-6)
  index = jnp.array([[0, 1, 2, 0]])
  samplers = [random.normal, lambda key, shape: index, random.normal]
  t2j_function_test(lambda input, index, src: torch.scatter(input, 0, index, src), [(3, 5), (1, 4), (2, 5)], samplers=samplers, atol=1e-6)
  index = jnp.array([[4, 2, 3], [3, 0, 4]])
  samplers = [random.normal, lambda key, shape: index, random.normal]
  t2j_function_test(lambda input, index, src: torch.scatter(input, 1, index, src), [(3, 5), (2, 3), (3, 5)], samplers=samplers, atol=1e-6)
  index = jnp.array([[4, 2, 3], [3, 0, 4], [0, 1, 2]])
  samplers = [random.normal, lambda key, shape: index, random.normal]
  t2j_function_test(lambda input, index, src: torch.scatter(input, 1, index, src), [(3, 5), (3, 3), (3, 5)], samplers=samplers, atol=1e-6)

  t2j_function_test(lambda x: torch.pow(x, 2), [()])
  t2j_function_test(lambda x: torch.pow(x, 2), [(3,)])
  t2j_function_test(torch.pow, [(), ()])
  t2j_function_test(torch.pow, [(), (3,)])
  t2j_function_test(torch.pow, [(3,), ()])
  t2j_function_test(torch.pow, [(3,), (3,)])
  t2j_function_test(lambda x: x.pow(3).sum(), [(3,)], atol=1e-6)
  t2j_function_test(lambda x: 3 * torch.mean(x), [(5,)], atol=1e-6)
  t2j_function_test(lambda x: 3.0 * x.mean(), [(5,)], atol=1e-6)
  t2j_function_test(torch.add, [(3,), (3,)])
  t2j_function_test(torch.add, [(3, 1), (1, 3)])
  t2j_function_test(torch.div, [(3,), (3,)])
  t2j_function_test(torch.div, [(3, 1), (1, 3)])
  t2j_function_test(torch.mean, [(5,)], atol=1e-6)
  t2j_function_test(torch.mean, [(5, 6)], kwargs=dict(dim=1, keepdim=False), atol=1e-6)
  t2j_function_test(torch.mean, [(5, 6)], kwargs=dict(dim=1, keepdim=True), atol=1e-6)
  t2j_function_test(torch.mul, [(3,), (3,)])
  t2j_function_test(torch.mul, [(3, 1), (1, 3)])
  t2j_function_test(torch.sqrt, [(5,)])
  t2j_function_test(torch.sub, [(3,), (3,)])
  t2j_function_test(torch.sub, [(3, 1), (1, 3)])
  t2j_function_test(torch.rsqrt, [(5,)])
  t2j_function_test(torch.sum, [(5,)], atol=1e-6)
  t2j_function_test(torch.sum, [(5, 6)], kwargs=dict(dim=1, keepdim=False), atol=1e-6)
  t2j_function_test(torch.sum, [(5, 6)], kwargs=dict(dim=1, keepdim=True), atol=1e-6)
  t2j_function_test(lambda x: 3 * x.sum(), [(5,)], atol=1e-6)
  t2j_function_test(lambda x: 3 * torch.sum(x), [(5,)], atol=1e-6)
  t2j_function_test(torch.sin, [(3,)], atol=1e-6)
  t2j_function_test(torch.cos, [(3,)], atol=1e-6)
  t2j_function_test(lambda x: -x, [(3,)])

  # Seems like an innocent test, but this can cause segfaults when using dlpack in t2j_array
  t2j_function_test(lambda x: torch.tensor([3.0]) * torch.mean(x), [(5,)], atol=1e-6)

  t2j_function_test(lambda x: torch.mul(torch.tensor([3.0]), torch.mean(x)), [(5,)], atol=1e-6)
  t2j_function_test(lambda x: torch.tensor([3]) * torch.mean(torch.sqrt(x)), [(3,)])

  t2j_function_test(lambda x, y: x @ y, [(2, 3), (3, 5)], atol=1e-6)
  t2j_function_test(lambda x: x.view(2, 2), [(2, 2)])
  t2j_function_test(lambda x: x.T, [(2, 2)])
  t2j_function_test(lambda x: x.view(2, 2).T, [(2, 2)])

  # view with list of ints
  t2j_function_test(lambda x: x.view(2, 2) @ x.view(2, 2), [(2, 2)], rtol=1e-6)
  t2j_function_test(lambda x: x.view(2, 2) @ x.view(2, 2).T, [(2, 2)], rtol=1e-6)
  t2j_function_test(lambda x: x.view(2, 2) @ x.view(2, 2).T, [(4,)], rtol=1e-6)
  t2j_function_test(lambda x: x.view(3, 4), [(12,)])
  t2j_function_test(lambda x: x.view(3, 4), [(4, 3)])

  # view with tuple input
  t2j_function_test(lambda x: x.view((2, 2)) @ x.view((2, 2)), [(2, 2)], rtol=1e-6)
  t2j_function_test(lambda x: x.view((2, 2)) @ x.view((2, 2)).T, [(2, 2)], rtol=1e-6)
  t2j_function_test(lambda x: x.view((2, 2)) @ x.view((2, 2)).T, [(4,)], rtol=1e-6)
  t2j_function_test(lambda x: x.view((3, 4)), [(12,)])
  t2j_function_test(lambda x: x.view((3, 4)), [(4, 3)])

  t2j_function_test(torch.unsqueeze, [(4, 3)], kwargs=dict(dim=0))
  t2j_function_test(torch.unsqueeze, [(4, 3)], kwargs=dict(dim=1))
  t2j_function_test(torch.unsqueeze, [(4, 3)], kwargs=dict(dim=2))
  t2j_function_test(lambda x: x.unsqueeze(0), [(4, 3)])
  t2j_function_test(lambda x: x.unsqueeze(1), [(4, 3)])
  t2j_function_test(lambda x: x.unsqueeze(2), [(4, 3)])

  t2j_function_test(lambda x: x.T.contiguous(), [(4, 3)])

  t2j_function_test(lambda x: x.permute(1, 0), [(4, 3)])
  t2j_function_test(lambda x: x.permute(1, 0, 2), [(4, 3, 2)])
  t2j_function_test(lambda x: x.permute(2, 0, 1), [(4, 3, 2)])

  t2j_function_test(lambda x: x.expand(5, -1, -1), [(1, 3, 2)])

  t2j_function_test(lambda x: torch.transpose(x, 0, 1), [(2, 3)])
  t2j_function_test(lambda x: torch.transpose(x, 0, 2), [(2, 3, 5)])
  t2j_function_test(lambda x: torch.transpose(x, 2, 1), [(2, 3, 5)])

  t2j_function_test(lambda x, y: torch.cat((x, y)), [(2, 3), (5, 3)])
  t2j_function_test(lambda x, y: torch.cat((x, y), dim=-1), [(2, 3), (2, 5)])

  t2j_function_test(torch.flatten, [(2, 3, 5)])
  t2j_function_test(torch.flatten, [(2, 3, 5)], kwargs=dict(start_dim=1))
  t2j_function_test(torch.flatten, [(2, 3, 5, 7)], kwargs=dict(start_dim=2))

  t2j_function_test(lambda x: x - 0.5, [(3,)])
  t2j_function_test(lambda x: 0.5 - x, [(3,)])
  t2j_function_test(lambda x: x - 5, [(3,)])
  t2j_function_test(lambda x: 5 - x, [(3,)])

  t2j_function_test(lambda x: x < 0.5, [(3,)])
  t2j_function_test(lambda x: x <= 0.5, [(3,)])
  t2j_function_test(lambda x: x > 0.5, [(3,)])
  t2j_function_test(lambda x: x >= 0.5, [(3,)])
  t2j_function_test(lambda x: x == x, [(3,)])
  t2j_function_test(lambda x: x != x, [(3,)])

  t2j_function_test(torch.abs, [(3,)])
  t2j_function_test(lambda x: (x > 0.0).float(), [(3,)])


def test_Tensor():
  with pytest.raises(ValueError):
    t2j(lambda: torch.Tensor([1, 2, 3]))()

  # Test that original torch.Tensor.__new__ implementation is restored
  aac(torch.Tensor([1, 2, 3]), jnp.array([1, 2, 3]))


def test_Tensor_clone():
  t2j_function_test(lambda x: x.clone(), [()])
  t2j_function_test(lambda x: x.clone(), [(2,)])
  t2j_function_test(lambda x: x.clone().add_(1), [(2,)])

  def f(x):
    x.clone().add_(1)
    return x

  t2j_function_test(f, [(2,)])


def test_Tensor_detach():
  t2j_function_test(lambda x: x.detach() ** 2, [()])
  t2j_function_test(lambda x: x.detach() ** 2, [(3,)])

  # This results in a shapes mismatch due to differences in the shapes that jax.grad and torch.func.grad output.
  #   t2j_function_test(lambda x: torch.sum(x.detach() ** 2), [(3,)])
  # so instead we do:
  aac(grad(t2j(lambda x: torch.sum(x.detach() ** 2)))(2.1 * jnp.arange(3)), 0)


def test_Tensor_item():
  aac(t2j(lambda x: x.item() * x)(jnp.array(3)), 9)
  with pytest.raises(Exception):
    jit(t2j(lambda x: x.item() * x))(jnp.array(3))

  with pytest.raises(Exception):
    grad(t2j(lambda x: x.item() * x))(jnp.array(3))


def test_inplace_Tensor_methods():
  def f(x):
    x = x + torch.tensor([3])
    x.add_(1)
    x.sub_(2.3)
    x.mul_(3.4)
    x.div_(5.6)
    return x

  t2j_function_test(f, [()], atol=1e-6)
  t2j_function_test(f, [(3,)], atol=1e-6)
  t2j_function_test(f, [(3, 5)], atol=1e-6)
  aac(vmap(t2j(f))(jnp.array([1, 2, 3])), jnp.array([f(1.0), f(2.0), f(3.0)]))
