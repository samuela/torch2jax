import numpy as np
import torch
from jax import jit, random

from torch2jax import t2j

aac = np.testing.assert_allclose


def anac(*args, **kwargs):
  fail = False
  try:
    np.testing.assert_allclose(*args, **kwargs)
    fail = True
  except AssertionError:
    return
  if fail:
    raise AssertionError("Should not be close")


torch.manual_seed(123)


# blocked on getting rng's in methods on Torchish
# def test_bernoulli():
#   f1 = t2j(lambda: torch.bernoulli(torch.empty(3, 3).uniform_(0, 1)))
#   aac(f1(rng=random.PRNGKey(0)), f1(rng=random.PRNGKey(0)))
#   anac(f1(rng=random.PRNGKey(0)), f1(rng=random.PRNGKey(1)))


def test_multinomial():
  pass  # TODO


def test_normal():
  def test(f):
    f1 = t2j(f)
    aac(f1(rng=random.PRNGKey(0)), f1(rng=random.PRNGKey(0)))
    anac(f1(rng=random.PRNGKey(0)), f1(rng=random.PRNGKey(1)))

    f2 = jit(t2j(f))
    aac(f2(rng=random.PRNGKey(0)), f2(rng=random.PRNGKey(0)))
    anac(f2(rng=random.PRNGKey(0)), f2(rng=random.PRNGKey(1)))

  # tensor (mean)
  test(lambda: torch.normal(1.0 * torch.arange(3)))

  # tensor, tensor
  test(lambda: torch.normal(1.0 * torch.arange(3), 2.5 * torch.arange(3)))

  # tensor, float
  test(lambda: torch.normal(1.0 * torch.arange(3), 2.5))

  # float, tensor
  test(lambda: torch.normal(1.0, 2.5 * torch.arange(3)))

  # float, float, tuple[int]
  test(lambda: torch.normal(1.0, 2.5, (2, 3)))

  # mean=tensor
  test(lambda: torch.normal(mean=1.0 * torch.arange(3)))

  # mean=tensor, std=tensor
  test(lambda: torch.normal(mean=1.0 * torch.arange(3), std=2.5 * torch.arange(3)))


def test_poisson():
  pass  # TODO


def test_rand_and_randn():
  for torchfn in {torch.rand, torch.randn}:
    f = jit(t2j(lambda: torchfn(())))
    aac(f(rng=random.PRNGKey(0)), f(rng=random.PRNGKey(0)))
    anac(f(rng=random.PRNGKey(0)), f(rng=random.PRNGKey(1)))

    f = jit(t2j(lambda x: x * torchfn(())))
    aac(f(10, rng=random.PRNGKey(0)), f(10, rng=random.PRNGKey(0)))
    anac(f(10, rng=random.PRNGKey(0)), f(10, rng=random.PRNGKey(1)))

    f = jit(t2j(lambda x: x * torchfn((2, 3))))
    aac(f(10, rng=random.PRNGKey(0)), f(10, rng=random.PRNGKey(0)))
    anac(f(10, rng=random.PRNGKey(0)), f(10, rng=random.PRNGKey(1)))

    f = jit(t2j(lambda x: x * torchfn([2, 3])))
    aac(f(10, rng=random.PRNGKey(0)), f(10, rng=random.PRNGKey(0)))
    anac(f(10, rng=random.PRNGKey(0)), f(10, rng=random.PRNGKey(1)))

    f = jit(t2j(lambda x: x * torchfn(2, 3)))
    aac(f(10, rng=random.PRNGKey(0)), f(10, rng=random.PRNGKey(0)))
    anac(f(10, rng=random.PRNGKey(0)), f(10, rng=random.PRNGKey(1)))
