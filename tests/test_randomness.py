import jax.numpy as jnp
import jax.random as jr
import pytest
import torch
from jax import jit

import torch2jax
from torch2jax import t2j

from .utils import aac, anac

jrk = jr.PRNGKey


def test_mk_rng():
  s = torch2jax._RNG_POOPER_STACK
  rngeq = lambda x, y: (x == y).all()
  jrs = jr.split

  with pytest.raises(Exception):
    torch2jax.mk_rng()

  assert len(s) == 0
  with torch2jax.RngPooperContext(torch2jax.RngPooper(jrk(0))):
    assert len(s) == 1 and rngeq(s[-1].rng, jrk(0))
    assert rngeq(torch2jax.mk_rng(), jrs(jrk(0))[1])
    assert len(s) == 1 and rngeq(s[-1].rng, jrs(jrk(0))[0])

    with torch2jax.RngPooperContext(torch2jax.RngPooper(jrk(123))):
      assert len(s) == 2 and rngeq(s[-1].rng, jrk(123))
      assert rngeq(torch2jax.mk_rng(), jrs(jrk(123))[1])
      assert len(s) == 2 and rngeq(s[-1].rng, jrs(jrk(123))[0])

      with torch2jax.RngPooperContext(None):
        assert len(s) == 3 and s[-1] is None
        with pytest.raises(Exception):
          torch2jax.mk_rng()

      assert len(s) == 2 and rngeq(s[-1].rng, jrs(jrk(123))[0])

    assert len(s) == 1 and rngeq(s[-1].rng, jrs(jrk(0))[0])

    # Consecutive calls to mk_rng should return different keys.
    assert not rngeq(torch2jax.mk_rng(), torch2jax.mk_rng())

  assert len(s) == 0


def test_bernoulli():
  f1 = t2j(lambda: torch.bernoulli(torch.empty(3, 3).uniform_(0, 1)))
  aac(f1(rng=jrk(0)), f1(rng=jrk(0)))
  anac(f1(rng=jrk(0)), f1(rng=jrk(1)))


def test_multinomial():
  ### 1D input
  aac(
    t2j(lambda w: torch.sort(torch.multinomial(w, w.shape[-1], replacement=False)))(
      jr.uniform(jrk(0), (10,)), rng=jrk(0)
    ),
    jnp.arange(10),
  )

  # 1D input, replacement=True
  t2j(lambda w: torch.multinomial(w, 11, replacement=True))(jr.uniform(jrk(0), (10,)), rng=jrk(0))

  # Test that replacement=False catches 11 draws from a 10-element tensor.
  with pytest.raises(ValueError):
    t2j(lambda w: torch.multinomial(w, 11, replacement=False))(jr.uniform(jrk(0), (10,)), rng=jrk(0))

  ### 2D input
  aac(
    t2j(lambda w: torch.sort(torch.multinomial(w, w.shape[-1], replacement=False)))(
      jr.uniform(jrk(0), (3, 10)), rng=jrk(0)
    ),
    jnp.stack([jnp.arange(10), jnp.arange(10), jnp.arange(10)]),
  )

  # 2D input, replacement=True
  t2j(lambda w: torch.multinomial(w, 11, replacement=True))(jr.uniform(jrk(0), (3, 10)), rng=jrk(0))

  # Test that replacement=False catches 11 draws from a 10-element tensor.
  with pytest.raises(ValueError):
    t2j(lambda w: torch.multinomial(w, 11, replacement=False))(jr.uniform(jrk(0), (3, 10)), rng=jrk(0))


def test_normal():
  def test(f):
    f1 = t2j(f)
    aac(f1(rng=jrk(0)), f1(rng=jrk(0)))
    anac(f1(rng=jrk(0)), f1(rng=jrk(1)))

    f2 = jit(t2j(f))
    aac(f2(rng=jrk(0)), f2(rng=jrk(0)))
    anac(f2(rng=jrk(0)), f2(rng=jrk(1)))

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
    aac(f1(rng=jrk(0)), f1(rng=jrk(0)))
    anac(f1(rng=jrk(0)), f1(rng=jrk(1)))

    f2 = jit(t2j(f))
    aac(f2(rng=jrk(0)), f2(rng=jrk(0)))
    anac(f2(rng=jrk(0)), f2(rng=jrk(1)))

  test(lambda: torch.poisson(1.0 * torch.arange(3)))


def test_rand_and_randn():
  for torchfn in {torch.rand, torch.randn}:
    f = jit(t2j(lambda: torchfn(())))
    aac(f(rng=jrk(0)), f(rng=jrk(0)))
    anac(f(rng=jrk(0)), f(rng=jrk(1)))

    f = jit(t2j(lambda x: x * torchfn(())))
    aac(f(10, rng=jrk(0)), f(10, rng=jrk(0)))
    anac(f(10, rng=jrk(0)), f(10, rng=jrk(1)))

    f = jit(t2j(lambda x: x * torchfn((2, 3))))
    aac(f(10, rng=jrk(0)), f(10, rng=jrk(0)))
    anac(f(10, rng=jrk(0)), f(10, rng=jrk(1)))

    f = jit(t2j(lambda x: x * torchfn([2, 3])))
    aac(f(10, rng=jrk(0)), f(10, rng=jrk(0)))
    anac(f(10, rng=jrk(0)), f(10, rng=jrk(1)))

    f = jit(t2j(lambda x: x * torchfn(2, 3)))
    aac(f(10, rng=jrk(0)), f(10, rng=jrk(0)))
    anac(f(10, rng=jrk(0)), f(10, rng=jrk(1)))


def test_rand_like():
  t2j(lambda x: torch.abs(torch.rand_like(x).mean() - 0.5) < 1e-3)(jnp.zeros(1_000_000), rng=jrk(0))
  t2j(lambda x: torch.sum(torch.abs(torch.rand_like(x) >= 0)) == x.numel())(jnp.zeros(1000), rng=jrk(0))
  t2j(lambda x: torch.sum(torch.abs(torch.rand_like(x) < 1)) == x.numel())(jnp.zeros(1000), rng=jrk(0))


def test_randint():
  f = t2j(lambda: torch.randint(0, 10, (2, 3)))
  aac(f(rng=jrk(0)), f(rng=jrk(0)))
  anac(f(rng=jrk(0)), f(rng=jrk(1)))

  f = t2j(lambda: torch.randint(10, (2, 3)))
  aac(f(rng=jrk(0)), f(rng=jrk(0)))
  anac(f(rng=jrk(0)), f(rng=jrk(1)))

  t2j(lambda: torch.abs(torch.mean(torch.randint(-10, 10, (1000,)) < 0) - 0.5) < 1e-2)(rng=jrk(0))
  t2j(lambda: torch.abs(torch.mean(torch.randint(-10, 10, (1000,)) > 0) - 0.5) < 1e-2)(rng=jrk(0))
  t2j(lambda: torch.abs(torch.mean(torch.randint(0, 10, (1000,)) == 0) - 0.1) < 1e-2)(rng=jrk(0))
  t2j(lambda: torch.abs(torch.mean(torch.randint(10, (1000,)) == 0) - 0.1) < 1e-2)(rng=jrk(0))
  t2j(lambda: torch.sum(torch.randint(10, (1000,)) == 10) == 0)(rng=jrk(0))


################################################################################
# torch.Tensor methods


def test_Tensor_bernoulli_():
  aac(t2j(lambda x: x.bernoulli_(0))(jnp.zeros(5), rng=jrk(0)), jnp.zeros(5))
  aac(t2j(lambda x: x.bernoulli_(1))(jnp.zeros(5), rng=jrk(0)), jnp.ones(5))
  aac(t2j(lambda x: x.bernoulli_(0.0))(jnp.zeros(5), rng=jrk(0)), jnp.zeros(5))
  aac(t2j(lambda x: x.bernoulli_(1.0))(jnp.zeros(5), rng=jrk(0)), jnp.ones(5))

  # test torch.Tensor values for p
  aac(t2j(lambda x: x.bernoulli_(torch.tensor(0.0)))(jnp.zeros(5), rng=jrk(0)), jnp.zeros(5))
  aac(t2j(lambda x: x.bernoulli_(torch.tensor(1.0)))(jnp.zeros(5), rng=jrk(0)), jnp.ones(5))


def test_Tensor_uniform_():
  f = t2j(lambda: torch.empty(3, 3).uniform_(11.2, 42.4))
  aac(f(rng=jrk(0)), f(rng=jrk(0)))
  anac(f(rng=jrk(0)), f(rng=jrk(1)))
  anac(f(rng=jrk(0)), f(rng=jrk(1)))
