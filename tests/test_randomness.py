import pytest
import torch
from jax import jit, random

import torch2jax
from torch2jax import t2j

from .utils import aac, anac


def test_mk_rng():
  s = torch2jax._RNG_POOPER_STACK
  rngeq = lambda x, y: (x == y).all()

  with pytest.raises(Exception):
    torch2jax.mk_rng()

  assert len(s) == 0
  with torch2jax.RngPooperContext(torch2jax.RngPooper(random.PRNGKey(0))):
    assert len(s) == 1 and rngeq(s[-1].rng, random.PRNGKey(0))
    assert rngeq(torch2jax.mk_rng(), random.split(random.PRNGKey(0))[1])
    assert len(s) == 1 and rngeq(s[-1].rng, random.split(random.PRNGKey(0))[0])

    with torch2jax.RngPooperContext(torch2jax.RngPooper(random.PRNGKey(123))):
      assert len(s) == 2 and rngeq(s[-1].rng, random.PRNGKey(123))
      assert rngeq(torch2jax.mk_rng(), random.split(random.PRNGKey(123))[1])
      assert len(s) == 2 and rngeq(s[-1].rng, random.split(random.PRNGKey(123))[0])

      with torch2jax.RngPooperContext(None):
        assert len(s) == 3 and s[-1] is None
        with pytest.raises(Exception):
          torch2jax.mk_rng()

      assert len(s) == 2 and rngeq(s[-1].rng, random.split(random.PRNGKey(123))[0])

    assert len(s) == 1 and rngeq(s[-1].rng, random.split(random.PRNGKey(0))[0])

    # Consecutive calls to mk_rng should return different keys.
    assert not rngeq(torch2jax.mk_rng(), torch2jax.mk_rng())

  assert len(s) == 0


def test_bernoulli():
  f1 = t2j(lambda: torch.bernoulli(torch.empty(3, 3).uniform_(0, 1)))
  aac(f1(rng=random.PRNGKey(0)), f1(rng=random.PRNGKey(0)))
  anac(f1(rng=random.PRNGKey(0)), f1(rng=random.PRNGKey(1)))


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
  def test(f):
    f1 = t2j(f)
    aac(f1(rng=random.PRNGKey(0)), f1(rng=random.PRNGKey(0)))
    anac(f1(rng=random.PRNGKey(0)), f1(rng=random.PRNGKey(1)))

    f2 = jit(t2j(f))
    aac(f2(rng=random.PRNGKey(0)), f2(rng=random.PRNGKey(0)))
    anac(f2(rng=random.PRNGKey(0)), f2(rng=random.PRNGKey(1)))

  test(lambda: torch.poisson(1.0 * torch.arange(3)))


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


def test_Tensor_uniform_():
  f = t2j(lambda: torch.empty(3, 3).uniform_(11.2, 42.4))
  aac(f(rng=random.PRNGKey(0)), f(rng=random.PRNGKey(0)))
  anac(f(rng=random.PRNGKey(0)), f(rng=random.PRNGKey(1)))
  anac(f(rng=random.PRNGKey(0)), f(rng=random.PRNGKey(1)))
