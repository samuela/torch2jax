import socket

import pytest
import torch
from jax import grad, jit, random

from torch2jax import RngPooper, j2t, t2j

from .utils import aac, assert_state_dicts_allclose


def test_mlp():
  for activation in [torch.nn.ReLU, torch.nn.Tanh, torch.nn.Sigmoid]:
    model = torch.nn.Sequential(torch.nn.Linear(2, 3, bias=False), activation(), torch.nn.Linear(3, 5))
    input_batch = random.normal(random.PRNGKey(123), (3, 2))
    params = {k: random.normal(random.PRNGKey(123), v.shape) for k, v in model.named_parameters()}

    model.load_state_dict({k: j2t(v) for k, v in params.items()})
    res_torch = model(j2t(input_batch))

    jaxified_module = t2j(model)
    res_jax = jaxified_module(input_batch, state_dict=params)
    res_jax_jit = jit(jaxified_module)(input_batch, state_dict=params)

    # Test forward pass with and without jax.jit
    aac(res_jax, res_torch.numpy(force=True), atol=1e-6)
    aac(res_jax_jit, res_torch.numpy(force=True), atol=1e-6)

    # Test gradients
    jax_grad = grad(lambda p: (jaxified_module(input_batch, state_dict=p) ** 2).sum())(params)

    res_torch.pow(2).sum().backward()
    torch_grad = {k: v.grad for k, v in model.named_parameters()}
    for k, v in model.named_parameters():
      aac(jax_grad[k], torch_grad[k], atol=1e-5)


def is_network_reachable():
  """Determine whether DNS resolution works on download.pytorch.org.

  The nix build environment disallows network access, making some tests
  impossible. We use this function to selectively disable those tests."""
  try:
    socket.gethostbyname("download.pytorch.org")
    return True
  except socket.gaierror:
    return False


@pytest.mark.skipif(not is_network_reachable(), reason="Network is not reachable")
def test_torchvision_models_resnet18():
  import torchvision

  rp = RngPooper(random.PRNGKey(123))
  input_batch = random.normal(rp.poop(), (5, 3, 224, 224))

  for eval in [False, True]:
    model = torchvision.models.resnet18(weights="DEFAULT")
    if eval:
      model.eval()

    # Copying is necessary since buffers are mutable in training mode
    parameters = {k: t2j(v) for k, v in model.named_parameters()}
    buffers = {k: t2j(v).copy() for k, v in model.named_buffers()}

    res_torch = model(j2t(input_batch))
    sd_torch = model.state_dict()

    jaxified_module = t2j(model)
    res_jax, sd_jax = jaxified_module(input_batch, state_dict={**parameters, **buffers}, return_state_dict=True)
    res_jax_jit, sd_jax_jit = jit(jaxified_module, static_argnames=["return_state_dict"])(
      input_batch, state_dict={**parameters, **buffers}, return_state_dict=True
    )

    # Test forward pass with and without jax.jit
    aac(res_jax, res_torch.numpy(force=True), atol=1e-4)
    aac(res_jax_jit, res_torch.numpy(force=True), atol=1e-4)

    assert_state_dicts_allclose(sd_jax, sd_torch, atol=1e-6)
    assert_state_dicts_allclose(sd_jax_jit, sd_torch, atol=1e-6)

    # Models use different convolution backends and are too deep to compare gradients programmatically. But they line up
    # to reasonable expectations.


@pytest.mark.skipif(not is_network_reachable(), reason="Network is not reachable")
def test_torchvision_models_vit_b_16():
  import torchvision

  model = torchvision.models.vit_b_16(weights="DEFAULT")
  model.eval()

  parameters = {k: t2j(v) for k, v in model.named_parameters()}
  assert len(dict(model.named_buffers()).keys()) == 0

  input_batch = random.normal(random.PRNGKey(123), (1, 3, 224, 224))
  res_torch = model(j2t(input_batch))

  jaxified_module = t2j(model)
  res_jax = jaxified_module(input_batch, state_dict=parameters)
  res_jax_jit = jit(jaxified_module)(input_batch, state_dict=parameters)

  # Test forward pass with and without jax.jit
  aac(res_jax, res_torch.numpy(force=True), atol=1e-1)
  aac(res_jax_jit, res_torch.numpy(force=True), atol=1e-1)

  # Models use different convolution backends and are too deep to compare gradients programmatically. But they line up
  # to reasonable expectations.
