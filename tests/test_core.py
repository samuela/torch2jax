import jax.numpy as jnp
import pytest
import torch
from jax import grad, jit, vmap

from torch2jax import t2j

from .utils import aac, t2j_function_test


def test_arange():
  t2j_function_test(lambda: torch.arange(10), [])
  t2j_function_test(lambda: torch.arange(2, 10), [])
  t2j_function_test(lambda: torch.arange(2, 10, 3), [])


def test_empty():
  # torch.empty returns uninitialized values, so we need to multiply by 0 for deterministic, testable behavior.
  # NaNs are possible, so we need to convert them first. See
  # https://discuss.pytorch.org/t/torch-empty-returns-nan/181389 and https://github.com/samuela/torch2jax/actions/runs/13348964668/job/37282967463.
  t2j_function_test(lambda: 0 * torch.nan_to_num(torch.empty(())), [])
  t2j_function_test(lambda: 0 * torch.nan_to_num(torch.empty(2)), [])
  t2j_function_test(lambda: 0 * torch.nan_to_num(torch.empty((2, 3))), [])


def test_nan_to_num():
  # Test handling of NaN values
  t2j_function_test(lambda: torch.nan_to_num(torch.tensor([float("nan"), 1.0, 2.0])), [])

  # Test handling of positive infinity
  t2j_function_test(lambda: torch.nan_to_num(torch.tensor([float("inf"), 1.0, 2.0])), [])

  # Test handling of negative infinity
  t2j_function_test(lambda: torch.nan_to_num(torch.tensor([float("-inf"), 1.0, 2.0])), [])

  # Test handling of all special values with custom replacements
  t2j_function_test(
    lambda: torch.nan_to_num(
      torch.tensor([float("nan"), float("inf"), float("-inf")]), nan=0.0, posinf=1.0, neginf=-1.0
    ),
    [],
  )


def test_ones():
  t2j_function_test(lambda: torch.ones(()), [])
  t2j_function_test(lambda: torch.ones(2), [])
  t2j_function_test(lambda: torch.ones(2, 3), [])
  t2j_function_test(lambda: torch.ones((2, 3)), [])


def test_ones_like():
  t2j_function_test(lambda x: torch.ones_like(x), [()])
  t2j_function_test(lambda x: torch.ones_like(x), [(2,)])
  t2j_function_test(lambda x: torch.ones_like(x), [(2, 3)])


def test_tensor():
  t2j_function_test(lambda: torch.tensor([]), [])
  t2j_function_test(lambda: torch.tensor([1, 2, 3]), [])
  t2j_function_test(lambda: torch.tensor([[1, 2, 3], [4, 5, 6]]), [])

  # torch allows calling torch.tensor with a torch.Tensor. This gets a little tricky with Torchish.
  t2j_function_test(lambda: torch.tensor(torch.arange(3)), [])


def test_zeros():
  t2j_function_test(lambda: torch.zeros(()), [])
  t2j_function_test(lambda: torch.zeros(2), [])
  t2j_function_test(lambda: torch.zeros(2, 3), [])
  t2j_function_test(lambda: torch.zeros((2, 3)), [])


def test_zeros_like():
  t2j_function_test(lambda x: torch.zeros_like(x), [()])
  t2j_function_test(lambda x: torch.zeros_like(x), [(2,)])
  t2j_function_test(lambda x: torch.zeros_like(x), [(2, 3)])


def test_unbind():
  t2j_function_test(lambda x: torch.unbind(x)[0], [(2, 3)])
  t2j_function_test(lambda x: torch.unbind(x, dim=1)[1], [(2, 3)])
  t2j_function_test(lambda x: x.unbind()[0], [(2, 3)])
  t2j_function_test(lambda x: x.unbind(1)[1], [(2, 3)])


def test_oneliners():
  t2j_function_test(lambda x: torch.pow(x, 2), [()])
  t2j_function_test(lambda x: torch.pow(x, 2), [(3,)])
  t2j_function_test(lambda x, y: torch.pow(x, y), [(), ()])
  t2j_function_test(lambda x, y: torch.pow(x, y), [(), (3,)])
  t2j_function_test(lambda x, y: torch.pow(x, y), [(3,), ()])
  t2j_function_test(lambda x, y: torch.pow(x, y), [(3,), (3,)])
  t2j_function_test(lambda x: x.pow(3), [()])
  t2j_function_test(lambda x: x.pow(3), [(3,)])
  t2j_function_test(lambda x: x.pow(3).sum(), [(3,)], atol=1e-6)
  t2j_function_test(lambda x: 3 * torch.mean(x), [(5,)], atol=1e-6)
  t2j_function_test(lambda x: 3.0 * x.mean(), [(5,)], atol=1e-6)
  t2j_function_test(torch.add, [(3,), (3,)])
  t2j_function_test(torch.add, [(3, 1), (1, 3)])
  t2j_function_test(torch.mean, [(5,)], atol=1e-6)
  t2j_function_test(lambda x: torch.mean(x, dim=1, keepdim=False), [(5, 6)], atol=1e-6)
  t2j_function_test(lambda x: torch.mean(x, dim=1, keepdim=True), [(5, 6)], atol=1e-6)
  t2j_function_test(torch.sqrt, [(5,)])
  t2j_function_test(torch.rsqrt, [(5,)])
  t2j_function_test(torch.sum, [(5,)], atol=1e-6)
  t2j_function_test(lambda x: torch.sum(x, dim=1, keepdim=False), [(5, 6)], atol=1e-6)
  t2j_function_test(lambda x: torch.sum(x, dim=1, keepdim=True), [(5, 6)], atol=1e-6)
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

  # view with dtype
  t2j_function_test(lambda x: x.view(torch.bool), [(12,)])
  t2j_function_test(lambda x: x.view(torch.int32), [(4, 3)])

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
  t2j_function_test(lambda x: torch.flatten(x, start_dim=1), [(2, 3, 5)])
  t2j_function_test(lambda x: torch.flatten(x, start_dim=2), [(2, 3, 5, 7)])

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
  aac(torch.Tensor([1, 2, 3]), [1, 2, 3])


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
  aac(vmap(t2j(f))(jnp.array([1, 2, 3])), [f(1.0), f(2.0), f(3.0)])
