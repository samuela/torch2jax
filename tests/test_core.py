from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax import grad, jit, random, vmap

from torch2jax import t2j

from .utils import Torchish_member_test, aac, backward_test, forward_test, out_kwarg_test, t2j_function_test


def test_arange():
  tests = [forward_test, out_kwarg_test]
  for test in tests:
    test(lambda out=None: torch.arange(10, out=out), [])
    test(lambda out=None: torch.arange(2, 10, out=out), [])
    test(lambda out=None: torch.arange(2, 10, 3, out=out), [])


def test_cast_to_scalar():
  # in future, we should have the __float__ and __int__ methods tested here.
  def f(x):
    return bool(x)

  assert t2j(f)(True)
  assert not t2j(f)(False)
  assert t2j(f)(jnp.array(True))
  assert t2j(f)(jnp.array([True]))
  assert t2j(f)(jnp.array([[True]]))
  # jitted version should fail, because casting traced dynamic value to bool is not allowed in jax.
  # jax.jit(t2j(f))(jnp.array(True))


def test_empty():
  # torch.empty returns uninitialized values, so we need to multiply by 0 for deterministic, testable behavior.
  # NaNs are possible, so we need to convert them first. See
  # https://discuss.pytorch.org/t/torch-empty-returns-nan/181389 and https://github.com/samuela/torch2jax/actions/runs/13348964668/job/37282967463.
  t2j_function_test(lambda: 0 * torch.nan_to_num(torch.empty(())), [])
  t2j_function_test(lambda: 0 * torch.nan_to_num(torch.empty(2)), [])
  t2j_function_test(lambda: 0 * torch.nan_to_num(torch.empty((2, 3))), [])


def test_nan_to_num():
  for value in ["nan", "inf", "-inf"]:
    samplers = [lambda rng, shape: jnp.array([float(value), 1.0, 2.0])]
    t2j_function_test(torch.nan_to_num, [(3,)], samplers=samplers, num_tests=1)

  # Test handling of all special values with custom replacements
  samplers = [lambda rng, shape: jnp.array([float("nan"), float("inf"), float("-inf")])]
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
  tests = [forward_test]
  t2j_function_test(torch.ones_like, [()], tests=tests)
  t2j_function_test(torch.ones_like, [(2,)], tests=tests)
  t2j_function_test(torch.ones_like, [(2, 3)], tests=tests)


def test_tree_coerce():
  @dataclass
  class A:
    a: torch.Tensor

  @dataclass
  class B:
    a: A
    b: int

  torch.utils._pytree.register_dataclass(A)
  torch.utils._pytree.register_dataclass(B)

  def f():
    t1 = torch.return_types.max([1, torch.ones(3)])
    t2 = torch.return_types.topk([3.0, t1])
    t3 = A(torch.ones(3, 2))
    t4 = B(t3, 7)
    return (t2, t3, t4)

  # test jit version works first
  jax.jit(t2j(f))()
  # test non-jit version
  t2j(f)()

  # test the values in the leaves are correct
  tests = [forward_test]
  t2j_function_test(f, [], tests=tests)


def test_tensor():
  tests = [forward_test]
  t2j_function_test(lambda: torch.tensor([]), [], tests=tests)
  t2j_function_test(lambda: torch.tensor([1, 2, 3]), [], tests=tests)
  t2j_function_test(lambda: torch.tensor([[1, 2, 3], [4, 5, 6]]), [], tests=tests)

  # test that an integer/boolean input will have the correct dtype
  # explicitly cast them to clearly show the intention
  t2j_function_test(lambda: torch.tensor(int(1)), [], tests=tests)
  t2j_function_test(lambda: torch.tensor(bool(True)), [], tests=tests)
  # torch allows calling torch.tensor with a torch.Tensor. This gets a little tricky with Torchish.
  t2j_function_test(lambda: torch.tensor(torch.arange(3)), [], tests=tests)


def test_zeros():
  tests = [forward_test, out_kwarg_test]
  for test in tests:
    test(lambda out=None: torch.zeros((), out=out), [])
    test(lambda out=None: torch.zeros(2, out=out), [])
    test(lambda out=None: torch.zeros(2, 3, out=out), [])
    test(lambda out=None: torch.zeros((2, 3), out=out), [])


def test_zeros_like():
  tests = [forward_test]
  t2j_function_test(torch.zeros_like, [()], tests=tests)
  t2j_function_test(torch.zeros_like, [(2,)], tests=tests)
  t2j_function_test(torch.zeros_like, [(2, 3)], tests=tests)


def test_unbind():
  tests = [forward_test, backward_test, Torchish_member_test]
  t2j_function_test(torch.unbind, [(2, 3)], tests=tests)
  t2j_function_test(torch.unbind, [(2, 3)], kwargs={"dim": 1}, tests=tests)


def test_get_set_item():
  # slice(torch.tensor(1), torch.tensor(3)) is not jittable in jax, because the start/end are dynamic,
  # and the shape can't be inferred. But it works when it is not jitted. This behavior is aligned with pure jax code,
  # where dynamic slice only works when not jitted. Therefore we turn off jit test here.

  # getitem
  tests = [forward_test, backward_test]
  tests_nojit = [partial(forward_test, test_jit=False), backward_test]
  t2j_function_test(lambda x: x[0, 1, 2], [(3, 4, 5)], tests=tests)
  t2j_function_test(lambda x: x[:, 1, 2], [(3, 4, 5)], tests=tests)
  t2j_function_test(lambda x: x[0, :, 2], [(3, 4, 5)], tests=tests)
  t2j_function_test(lambda x: x[0, 1, :], [(3, 4, 5)], tests=tests)
  t2j_function_test(lambda x: x[1:, 1:3, 2], [(3, 4, 5)], tests=tests)
  t2j_function_test(lambda x: x[torch.tensor([1, 2]), :, torch.tensor([2, 3])], [(3, 4, 5)], tests=tests)
  # when the slice object is dynamic, jax will refuse to run with jit.
  t2j_function_test(lambda x: x[1:, torch.tensor(1) : torch.tensor(3), 2], [(3, 4, 5)], tests=tests_nojit)

  # setitem
  def f(x, key, y):
    x[*key] = y
    return x

  # in torch, __setitem__ is for inplace assignment, it is not differentiable in torch.
  tests = [forward_test, partial(backward_test, argnums=(1,))]
  tests_nojit = [partial(forward_test, test_jit=False), partial(backward_test, argnums=(1,))]
  t2j_function_test(lambda x, y: f(x, [0, 1, 2], y), [(3, 4, 5), ()], tests=tests)
  t2j_function_test(lambda x, y: f(x, [slice(None), 1, 2], y), [(3, 4, 5), (3,)], tests=tests)
  t2j_function_test(lambda x, y: f(x, [0, slice(None), 2], y), [(3, 4, 5), (4,)], tests=tests)
  t2j_function_test(lambda x, y: f(x, [0, 1, slice(None)], y), [(3, 4, 5), (5,)], tests=tests)
  t2j_function_test(lambda x, y: f(x, [slice(1, None), slice(1, 3), 2], y), [(3, 4, 5), (2, 2)], tests=tests)
  t2j_function_test(lambda x, y: f(x, [slice(1, None), slice(1, 3), 2], y), [(3, 4, 5), (2, 1)], tests=tests)
  t2j_function_test(
    lambda x, y: f(x, [torch.tensor([1, 2]), slice(None), torch.tensor([2, 3])], y),
    [(3, 4, 5), (2, 4)],
    tests=tests,
  )
  t2j_function_test(
    lambda x, y: f(x, [slice(1, None), slice(torch.tensor(1), torch.tensor(3)), 2], y),
    [(3, 4, 5), (2, 2)],
    tests=tests_nojit,
  )
  t2j_function_test(
    lambda x, y: f(x, [slice(1, None), slice(torch.tensor(1), torch.tensor(3)), 2], y),
    [(3, 4, 5), ()],
    tests=tests_nojit,
  )


def test_oneliners():
  f = [forward_test]
  fb = f + [backward_test]
  fbm = fb + [Torchish_member_test]
  fbo = fb + [out_kwarg_test]
  fbmo = fbm + [out_kwarg_test]
  fmo = f + [Torchish_member_test, out_kwarg_test]

  t2j_function_test(lambda x: torch.pow(x, 2), [()], tests=fb)
  t2j_function_test(lambda x: torch.pow(x, 2), [(3,)], tests=fb)
  t2j_function_test(torch.pow, [(), ()], tests=fbmo)
  t2j_function_test(torch.pow, [(), (3,)], tests=fbmo)
  t2j_function_test(torch.pow, [(3,), ()], tests=fbmo)
  t2j_function_test(torch.pow, [(3,), (3,)], tests=fbmo)
  t2j_function_test(lambda x: x.pow(3).sum(), [(3,)], atol=1e-6, tests=fb)
  t2j_function_test(lambda x: 3 * torch.mean(x), [(5,)], atol=1e-6, tests=fb)
  t2j_function_test(lambda x: 3.0 * x.mean(), [(5,)], atol=1e-6, tests=fb)
  t2j_function_test(torch.add, [(3,), (3,)], tests=fbmo)
  t2j_function_test(torch.add, [(3, 1), (1, 3)], tests=fbmo)
  t2j_function_test(torch.div, [(3,), (3,)], tests=fbmo)
  t2j_function_test(torch.div, [(3, 1), (1, 3)], tests=fbmo)
  t2j_function_test(torch.mean, [(5,)], atol=1e-6, tests=fbmo)
  t2j_function_test(torch.mean, [(5, 6)], kwargs=dict(dim=1, keepdim=False), atol=1e-6, tests=fbmo)
  t2j_function_test(torch.mean, [(5, 6)], kwargs=dict(dim=1, keepdim=True), atol=1e-6, tests=fbmo)
  t2j_function_test(torch.mul, [(3,), (3,)], tests=fbmo)
  t2j_function_test(torch.mul, [(3, 1), (1, 3)], tests=fbmo)
  t2j_function_test(torch.sqrt, [(5,)], tests=fbmo)
  t2j_function_test(torch.sub, [(3,), (3,)], tests=fbmo)
  t2j_function_test(torch.sub, [(3, 1), (1, 3)], tests=fbmo)
  t2j_function_test(torch.rsqrt, [(5,)], tests=fbmo)
  t2j_function_test(torch.sum, [(5,)], atol=1e-6, tests=fbm)
  t2j_function_test(torch.sum, [(5, 6)], kwargs=dict(dim=1, keepdim=False), atol=1e-6, tests=fbmo)
  t2j_function_test(torch.sum, [(5, 6)], kwargs=dict(dim=1, keepdim=True), atol=1e-6, tests=fbmo)
  t2j_function_test(lambda x: 3 * x.sum(), [(5,)], atol=1e-6, tests=fb)
  t2j_function_test(lambda x: 3 * torch.sum(x), [(5,)], atol=1e-6, tests=fb)
  t2j_function_test(torch.sin, [(3,)], atol=1e-6, tests=fbmo)
  t2j_function_test(torch.cos, [(3,)], atol=1e-6, tests=fbmo)
  t2j_function_test(lambda x: -x, [(3,)], tests=fb)

  samplers = [random.bernoulli]
  t2j_function_test(torch.all, [(3, 2)], samplers=samplers, tests=fmo)
  t2j_function_test(torch.all, [(3, 2)], samplers=samplers, kwargs=dict(dim=1), tests=fmo)
  t2j_function_test(torch.any, [(3, 2)], samplers=samplers, tests=fmo)
  t2j_function_test(torch.any, [(3, 2)], samplers=samplers, kwargs=dict(dim=1), tests=fmo)

  # bitwise_not on int and bool tensors
  t2j_function_test(
    torch.bitwise_not,
    [(3, 2)],
    samplers=[lambda key, shape: random.randint(key, shape, minval=0, maxval=1024)],
    tests=fmo,
  )
  t2j_function_test(torch.bitwise_not, [(3, 2)], samplers=[random.bernoulli], tests=fmo)
  # logical operations
  samplers = [random.bernoulli, random.bernoulli]
  t2j_function_test(torch.logical_and, [(3, 2), (3, 2)], samplers=samplers, tests=fmo)
  t2j_function_test(torch.logical_and, [(3, 2), (2)], samplers=samplers, tests=fmo)
  t2j_function_test(torch.logical_and, [(3, 2), (3, 1)], samplers=samplers, tests=fmo)
  t2j_function_test(torch.logical_or, [(3, 2), (3, 2)], samplers=samplers, tests=fmo)
  t2j_function_test(torch.logical_or, [(3, 2), (2)], samplers=samplers, tests=fmo)
  t2j_function_test(torch.logical_or, [(3, 2), (3, 1)], samplers=samplers, tests=fmo)
  t2j_function_test(torch.logical_xor, [(3, 2), (3, 2)], samplers=samplers, tests=fmo)
  t2j_function_test(torch.logical_xor, [(3, 2), (2)], samplers=samplers, tests=fmo)
  t2j_function_test(torch.logical_xor, [(3, 2), (3, 1)], samplers=samplers, tests=fmo)
  t2j_function_test(torch.logical_not, [(3, 2)], samplers=[random.bernoulli], tests=fmo)
  t2j_function_test(torch.logical_not, [(2)], samplers=[random.bernoulli], tests=fmo)
  t2j_function_test(torch.logical_not, [(3, 1)], samplers=[random.bernoulli], tests=fmo)
  # tensor operators
  t2j_function_test(lambda x, y: x & y, [(3, 1), (3, 1)], samplers=samplers, tests=f)
  t2j_function_test(lambda x, y: x | y, [(3, 1), (3, 1)], samplers=samplers, tests=f)
  t2j_function_test(lambda x, y: x ^ y, [(3, 1), (3, 1)], samplers=samplers, tests=f)
  t2j_function_test(lambda x: ~x, [(3, 1)], samplers=[random.bernoulli], tests=f)

  t2j_function_test(torch.max, [(3, 5)], atol=1e-6)
  t2j_function_test(torch.max, [(3, 5)], kwargs=dict(dim=1), atol=1e-6)
  t2j_function_test(torch.max, [(3, 5)], kwargs=dict(dim=1, keepdim=True), atol=1e-6)
  t2j_function_test(torch.max, [(3, 5, 7)], kwargs=dict(dim=0, keepdim=True), atol=1e-6)
  t2j_function_test(torch.max, [(3, 5, 7)], kwargs=dict(dim=1, keepdim=True), atol=1e-6)
  t2j_function_test(torch.max, [(3, 5, 7)], kwargs=dict(dim=2, keepdim=True), atol=1e-6)
  t2j_function_test(torch.max, [(3, 5, 7)], kwargs=dict(dim=0, keepdim=False), atol=1e-6)
  t2j_function_test(torch.max, [(3, 5, 7)], kwargs=dict(dim=1, keepdim=False), atol=1e-6)
  t2j_function_test(torch.max, [(3, 5, 7)], kwargs=dict(dim=2, keepdim=False), atol=1e-6)
  t2j_function_test(torch.max, [(3, 5, 7), (3, 5, 7)], atol=1e-6)
  t2j_function_test(torch.max, [(3, 5, 7), (5, 7)], atol=1e-6)  # broadcasting
  t2j_function_test(torch.max, [(3, 5, 7), (7)], atol=1e-6)  # broadcasting
  t2j_function_test(torch.sort, [(3, 5)], kwargs=dict(dim=0), atol=1e-6)
  t2j_function_test(torch.sort, [(3, 5)], kwargs=dict(dim=1), atol=1e-6)
  t2j_function_test(torch.sort, [(3, 5)], kwargs=dict(dim=0, descending=True), atol=1e-6)
  t2j_function_test(torch.topk, [(7, 9)], kwargs=dict(k=1, dim=0), atol=1e-6)
  t2j_function_test(torch.topk, [(7, 9)], kwargs=dict(k=2, dim=0), atol=1e-6)
  t2j_function_test(torch.topk, [(7, 9)], kwargs=dict(k=3, dim=0), atol=1e-6)
  t2j_function_test(torch.topk, [(7, 9)], kwargs=dict(k=1, dim=1), atol=1e-6)
  t2j_function_test(torch.topk, [(7, 9)], kwargs=dict(k=2, dim=1), atol=1e-6)
  t2j_function_test(torch.topk, [(7, 9)], kwargs=dict(k=3, dim=1), atol=1e-6)

  # Seems like an innocent test, but this can cause segfaults when using dlpack in t2j_array
  t2j_function_test(lambda x: torch.tensor([3.0]) * torch.mean(x), [(5,)], atol=1e-6, tests=fb)

  t2j_function_test(lambda x: torch.mul(torch.tensor([3.0]), torch.mean(x)), [(5,)], atol=1e-6, tests=fb)
  t2j_function_test(lambda x: torch.tensor([3]) * torch.mean(torch.sqrt(x)), [(3,)], tests=fb)

  t2j_function_test(lambda x, y: x @ y, [(2, 3), (3, 5)], atol=1e-6, tests=fb)
  t2j_function_test(lambda x: x.view(2, 2), [(2, 2)], tests=fb)
  t2j_function_test(lambda x: x.T, [(2, 2)], tests=fb)
  t2j_function_test(lambda x: x.view(2, 2).T, [(2, 2)], tests=fb)

  # view with list of ints
  t2j_function_test(lambda x: x.view(2, 2) @ x.view(2, 2), [(2, 2)], rtol=1e-6, tests=fb)
  t2j_function_test(lambda x: x.view(2, 2) @ x.view(2, 2).T, [(2, 2)], rtol=1e-6, tests=fb)
  t2j_function_test(lambda x: x.view(2, 2) @ x.view(2, 2).T, [(4,)], rtol=1e-6, tests=fb)
  t2j_function_test(lambda x: x.view(3, 4), [(12,)], tests=fb)
  t2j_function_test(lambda x: x.view(3, 4), [(4, 3)], tests=fb)

  # view with tuple input
  t2j_function_test(lambda x: x.view((2, 2)) @ x.view((2, 2)), [(2, 2)], rtol=1e-6, tests=fb)
  t2j_function_test(lambda x: x.view((2, 2)) @ x.view((2, 2)).T, [(2, 2)], rtol=1e-6, tests=fb)
  t2j_function_test(lambda x: x.view((2, 2)) @ x.view((2, 2)).T, [(4,)], rtol=1e-6, tests=fb)
  t2j_function_test(lambda x: x.view((3, 4)), [(12,)], tests=fb)
  t2j_function_test(lambda x: x.view((3, 4)), [(4, 3)], tests=fb)

  # view with dtype
  t2j_function_test(lambda x: x.view(torch.bool), [(12,)], tests=f)
  t2j_function_test(lambda x: x.view(torch.int32), [(4, 3)], tests=f)

  t2j_function_test(lambda x: x.to(torch.float16), [(4, 3)], tests=f)
  t2j_function_test(lambda x: x.to(torch.int32), [(4, 3)], tests=f)
  t2j_function_test(lambda x: x.to(dtype=torch.float16), [(4, 3)], tests=f)
  t2j_function_test(lambda x: x.to(dtype=torch.int32), [(4, 3)], tests=f)

  t2j_function_test(torch.unsqueeze, [(4, 3)], kwargs=dict(dim=0), tests=fbm)
  t2j_function_test(torch.unsqueeze, [(4, 3)], kwargs=dict(dim=1), tests=fbm)
  t2j_function_test(torch.unsqueeze, [(4, 3)], kwargs=dict(dim=2), tests=fbm)

  t2j_function_test(lambda x: x.unsqueeze(0), [(4, 3)], tests=fb)
  t2j_function_test(lambda x: x.unsqueeze(1), [(4, 3)], tests=fb)
  t2j_function_test(lambda x: x.unsqueeze(2), [(4, 3)], tests=fb)

  t2j_function_test(lambda x: x.T.contiguous(), [(4, 3)], tests=fb)

  t2j_function_test(lambda x: x.permute(1, 0), [(4, 3)], tests=fb)
  t2j_function_test(lambda x: x.permute(1, 0, 2), [(4, 3, 2)], tests=fb)
  t2j_function_test(lambda x: x.permute(2, 0, 1), [(4, 3, 2)], tests=fb)

  t2j_function_test(lambda x: x.expand(5, -1, -1), [(1, 3, 2)], tests=fb)

  t2j_function_test(lambda x: torch.transpose(x, 0, 1), [(2, 3)], tests=fb)
  t2j_function_test(lambda x: torch.transpose(x, 0, 2), [(2, 3, 5)], tests=fb)
  t2j_function_test(lambda x: torch.transpose(x, 2, 1), [(2, 3, 5)], tests=fb)

  t2j_function_test(lambda x, y, out=None: torch.cat((x, y), out=out), [(2, 3), (5, 3)], tests=fbo)
  t2j_function_test(lambda x, y, out=None: torch.cat((x, y), dim=-1, out=out), [(2, 3), (2, 5)], tests=fbo)

  t2j_function_test(torch.flatten, [(2, 3, 5)], tests=fbm)
  t2j_function_test(torch.flatten, [(2, 3, 5)], kwargs=dict(start_dim=1), tests=fbm)
  t2j_function_test(torch.flatten, [(2, 3, 5, 7)], kwargs=dict(start_dim=2), tests=fbm)

  t2j_function_test(lambda x: x - 0.5, [(3,)], tests=fb)
  t2j_function_test(lambda x: 0.5 - x, [(3,)], tests=fb)
  t2j_function_test(lambda x: x - 5, [(3,)], tests=fb)
  t2j_function_test(lambda x: 5 - x, [(3,)], tests=fb)

  t2j_function_test(lambda x: x < 0.5, [(3,)], tests=f)
  t2j_function_test(lambda x: x <= 0.5, [(3,)], tests=f)
  t2j_function_test(lambda x: x > 0.5, [(3,)], tests=f)
  t2j_function_test(lambda x: x >= 0.5, [(3,)], tests=f)
  t2j_function_test(lambda x: x == x, [(3,)], tests=f)
  t2j_function_test(lambda x: x != x, [(3,)], tests=f)

  t2j_function_test(torch.abs, [(3,)], tests=fbmo)
  t2j_function_test(lambda x: (x > 0.0).float(), [(3,)], tests=fb)


def test_scatter():
  # scatter
  index = np.array([[0, 1, 2, 0, 2], [1, 0, 0, 2, 1]], dtype=np.int64)
  samplers = [random.normal, lambda key, shape: index, random.normal]
  tests = [forward_test, partial(backward_test, argnums=(0, 2))]
  t2j_function_test(
    lambda input, index, src: torch.scatter(input, 0, index, src),
    [(3, 5), (2, 5), (2, 5)],
    samplers=samplers,
    atol=1e-6,
    tests=tests,
  )
  # Disable gradient testing when index.shape != src.shape
  # This is an existing problem of pytorch https://github.com/pytorch/pytorch/issues/27614
  index = np.array([[0, 1, 2, 0, 2]], dtype=np.int64)
  samplers = [random.normal, lambda key, shape: index, random.normal]
  tests = [forward_test]
  t2j_function_test(
    lambda input, index, src: torch.scatter(input, 0, index, src),
    [(3, 5), (1, 5), (2, 5)],
    samplers=samplers,
    atol=1e-6,
    tests=tests,
  )
  index = np.array([[0, 1, 2, 0]], dtype=np.int64)
  samplers = [random.normal, lambda key, shape: index, random.normal]
  t2j_function_test(
    lambda input, index, src: torch.scatter(input, 0, index, src),
    [(3, 5), (1, 4), (2, 5)],
    samplers=samplers,
    atol=1e-6,
    tests=tests,
  )
  index = np.array([[4, 2, 3], [3, 0, 4]], dtype=np.int64)
  samplers = [random.normal, lambda key, shape: index, random.normal]
  t2j_function_test(
    lambda input, index, src: torch.scatter(input, 1, index, src),
    [(3, 5), (2, 3), (3, 5)],
    samplers=samplers,
    atol=1e-6,
    tests=tests,
  )
  index = np.array([[4, 2, 3], [3, 0, 4], [0, 1, 2]], dtype=np.int64)
  samplers = [random.normal, lambda key, shape: index, random.normal]
  t2j_function_test(
    lambda input, index, src: torch.scatter(input, 1, index, src),
    [(3, 5), (3, 3), (3, 5)],
    samplers=samplers,
    atol=1e-6,
    tests=tests,
  )


def test_Tensor():
  with pytest.raises(ValueError):
    t2j(lambda: torch.Tensor([1, 2, 3]))()

  # Test that original torch.Tensor.__new__ implementation is restored
  aac(torch.Tensor([1, 2, 3]), jnp.array([1, 2, 3]))


def test_Tensor_clone():
  fb = [forward_test, backward_test]
  fbm = [forward_test, backward_test, Torchish_member_test]
  t2j_function_test(torch.clone, [()], tests=fbm)
  t2j_function_test(torch.clone, [(2,)], tests=fbm)
  t2j_function_test(lambda x: x.clone().add_(1), [(2,)], tests=fb)

  def f(x):
    x.clone().add_(1)
    return x

  t2j_function_test(f, [(2,)], tests=fb)


def test_Tensor_detach():
  tests = [forward_test, backward_test]
  t2j_function_test(lambda x: x.detach() ** 2, [()], tests=tests)
  t2j_function_test(lambda x: x.detach() ** 2, [(3,)], tests=tests)

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
  tests = [forward_test, backward_test]

  def f(x):
    x = x + torch.tensor([3])
    x.add_(1)
    x.sub_(2.3)
    x.mul_(3.4)
    x.div_(5.6)
    return x

  t2j_function_test(f, [()], atol=1e-6, tests=tests)
  t2j_function_test(f, [(3,)], atol=1e-6, tests=tests)
  t2j_function_test(f, [(3, 5)], atol=1e-6, tests=tests)
  aac(vmap(t2j(f))(jnp.array([1, 2, 3])), jnp.array([f(1.0), f(2.0), f(3.0)]))


def test_grad_on_off():
  with torch.no_grad():
    assert torch.is_grad_enabled() is False

  with torch.enable_grad():
    assert torch.is_grad_enabled() is True

  with torch.set_grad_enabled(True):
    assert torch.is_grad_enabled() is True

  with torch.set_grad_enabled(False):
    assert torch.is_grad_enabled() is False

  # test no_grad context
  def f1(x):
    with torch.no_grad():
      a = x * 2
    b = torch.sin(a)
    c = torch.cos(x)
    with torch.no_grad():
      d = torch.pow(c, 2)
    return b * c + d

  tests = [forward_test, backward_test]
  t2j_function_test(f1, [()], atol=1e-6, tests=tests)

  # test enable_grad
  def f2(x):
    @torch.enable_grad()
    def doubler(x):
      return x * 2

    with torch.no_grad():
      z = doubler(x)
    return z

  t2j_function_test(f2, [()], atol=1e-6, tests=tests)

  # test set_grad_enabled
  def f3(x):
    with torch.set_grad_enabled(False):
      with torch.set_grad_enabled(True):
        y = torch.sin(x)
      y = y * 2
    with torch.set_grad_enabled(True):
      with torch.set_grad_enabled(False):
        z = torch.cos(x)
      z = z * 3
    return y + z

  t2j_function_test(f3, [()], atol=1e-6, tests=tests)

  # test inplace functions
  def f4(x):
    # this is effectively an identity function
    # but with no_grad, the gradient should be zero
    y = -x
    with torch.no_grad():
      return torch.nn.functional.relu(x, inplace=True) - torch.nn.functional.relu(y, inplace=True)

  t2j_function_test(f4, [()], atol=1e-6, tests=tests)


# See https://github.com/samuela/torch2jax/issues/30.
def test_tuple_return_function():
  def f(x):
    y1 = x + 1
    y2 = x * 2
    return y1, y2

  t2j_function_test(f, [(3,)], tests=[forward_test])
