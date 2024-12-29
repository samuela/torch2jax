# TODO: test relu, including inplace=True
# TODO: test batchnorm2d
# TODO: test torch.nn.functional.layer_norm

import socket

import jax.numpy as jnp
import pytest
import torch
from jax import grad, jit, random

from torch2jax import RngPooper, j2t, t2j

from .utils import aac, anac, t2j_function_test

# def test_torch_nn_functional_batch_norm():
#   raise


def test_torch_nn_functional_scaled_dot_product_attention():
  t2j_function_test(lambda x, y: x @ y, [(2, 3, 5), (5, 7)], atol=1e-6)

  sdpa = torch.nn.functional.scaled_dot_product_attention
  t2j_function_test(sdpa, [(5, 3, 7), (5, 2, 7), (5, 2, 7)], atol=1e-6)
  t2j_function_test(sdpa, [(5, 7, 11), (5, 7, 11), (5, 7, 11)], atol=1e-6)

  E = 6
  num_heads = 2
  t2j_function_test(
    lambda q, k, v, ipw, ipb, opw, opb: torch.nn.functional.multi_head_attention_forward(
      q, k, v, E, num_heads, ipw, ipb, None, None, False, 0.0, opw, opb, training=False, need_weights=False
    )[0],
    [(3, 1, 6)] * 3 + [(3 * E, E), (3 * E,), (E, E), (E,)],
    atol=1e-5,
  )
  # TODO test MHA without batch dimension


def test_torch_nn_AdaptiveAvgPool2d():
  # for output_size in [1, 2, (3, 4), (None, 4), (5, None)]:
  for output_size in [1]:
    model = torch.nn.AdaptiveAvgPool2d(output_size)
    input_batch = random.normal(random.PRNGKey(123), (7, 2, 16, 16))
    res_torch = model(j2t(input_batch))

    jaxified_module = t2j(model)
    res_jax = jaxified_module(input_batch)
    res_jax_jit = jit(jaxified_module)(input_batch)

    # Test forward pass with and without jax.jit
    aac(res_jax, res_torch.numpy(force=True), atol=1e-5)
    aac(res_jax_jit, res_torch.numpy(force=True), atol=1e-5)

    # Test gradients
    jax_grad = grad(lambda x: (jaxified_module(x) ** 2).sum())(input_batch)

    x = j2t(input_batch)
    x.requires_grad = True
    model(x).pow(2).sum().backward()
    # Note these gradients are just the same value repeated. TODO: verify that makes sense mathematically.
    aac(jax_grad, x.grad, atol=1e-8)


def test_torch_nn_Linear():
  model = torch.nn.Linear(2, 5)
  input_batch = random.normal(random.PRNGKey(123), (3, 2))
  params = {
    "weight": random.normal(random.PRNGKey(123), (5, 2)),
    "bias": random.normal(random.PRNGKey(123), (5,)),
  }

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
  aac(jax_grad["weight"], model.weight.grad, atol=1e-6)
  aac(jax_grad["bias"], model.bias.grad, atol=1e-6)


def test_torch_nn_Linear_no_bias():
  model = torch.nn.Linear(2, 5, bias=False)
  input_batch = random.normal(random.PRNGKey(123), (3, 2))
  params = {"weight": random.normal(random.PRNGKey(123), (5, 2))}

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
  aac(jax_grad["weight"], model.weight.grad, atol=1e-6)


def test_torch_nn_Conv2d():
  for bias in [False, True]:
    for stride in [1, (1, 1), 2, (2, 2), (1, 3), (3, 1)]:
      for padding in [0, 1, 2, "valid", "same", (1, 2)]:
        for dilation in [1, 2, (1, 1), (2, 3)]:
          if padding == "same":
            # ValueError: padding='same' is not supported for strided convolutions
            stride = 1
          model = torch.nn.Conv2d(2, 3, (5, 5), bias=bias, stride=stride, padding=padding, dilation=dilation)

          input_batch = random.normal(random.PRNGKey(123), (7, 2, 16, 16))
          params = {k: random.normal(random.PRNGKey(123), v.shape) for k, v in model.named_parameters()}

          model.load_state_dict({k: j2t(v) for k, v in params.items()})
          res_torch = model(j2t(input_batch))

          jaxified_module = t2j(model)
          res_jax = jaxified_module(input_batch, state_dict=params)
          res_jax_jit = jit(jaxified_module)(input_batch, state_dict=params)

          # Test forward pass with and without jax.jit
          aac(res_jax, res_torch.numpy(force=True), atol=1e-5)
          aac(res_jax_jit, res_torch.numpy(force=True), atol=1e-5)

          # Test gradients
          jax_grad = grad(lambda p: (jaxified_module(input_batch, state_dict=p) ** 2).sum())(params)

          res_torch.pow(2).sum().backward()
          aac(jax_grad["weight"], model.weight.grad, rtol=1e-3)
          if bias:
            aac(jax_grad["bias"], model.bias.grad, rtol=1e-3)


def test_torch_nn_BatchNorm1d():
  # TODO: test (N, C, L) shape input
  rp = RngPooper(random.PRNGKey(123))

  for eps in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
    for affine in [False, True]:
      model = torch.nn.BatchNorm1d(5, eps=eps, affine=affine)
      model.eval()

      input_batch = random.normal(rp.poop(), (3, 5))  # (N, C, L) is also supported
      params = {
        "running_mean": random.normal(rp.poop(), (5,)),
        "running_var": random.uniform(rp.poop(), (5,)),
        "num_batches_tracked": random.randint(rp.poop(), (1,), 0, 100),
      }
      if affine:
        params["weight"] = random.normal(rp.poop(), (5,))
        params["bias"] = random.normal(rp.poop(), (5,))

      model.load_state_dict({k: j2t(v) for k, v in params.items()})
      res_torch = model(j2t(input_batch))

      jaxified_module = t2j(model)
      res_jax = jaxified_module(input_batch, state_dict=params)
      res_jax_jit = jit(jaxified_module)(input_batch, state_dict=params)

      # Test forward pass with and without jax.jit
      aac(res_jax, res_torch.numpy(force=True), atol=1e-6)
      aac(res_jax_jit, res_torch.numpy(force=True), atol=1e-6)

      # Test gradients
      if affine:
        jax_grad = grad(
          lambda p: (jaxified_module(input_batch, state_dict={**p, "num_batches_tracked": 0}) ** 2).sum()
        )({k: v for k, v in params.items() if k != "num_batches_tracked"})

        res_torch.pow(2).sum().backward()
        aac(jax_grad["weight"], model.weight.grad, atol=1e-5)
        aac(jax_grad["bias"], model.bias.grad, atol=1e-6)


def test_torch_nn_MaxPool1d():
  for kernel_size in [1, 2, 3, 4, 5, (1,), (2,), (3,), (4,), (5,)]:
    for stride in [None, 1, (1,), 2, (2,), 3, (3,)]:
      for padding in [0, 1, (2,), (5,)]:
        model = torch.nn.MaxPool1d(kernel_size, stride, padding)
        input_batch = random.normal(random.PRNGKey(123), (7, 2, 16))

        try:
          res_torch = model(j2t(input_batch))
        except Exception:
          # RuntimeError: max_pool1d() padding should be at most half of kernel size, but got padding=2 and kernel_size=2
          continue

        jaxified_module = t2j(model)
        res_jax = jaxified_module(input_batch)
        res_jax_jit = jit(jaxified_module)(input_batch)

        # Test forward pass with and without jax.jit
        aac(res_jax, res_torch.numpy(force=True), atol=1e-5)
        aac(res_jax_jit, res_torch.numpy(force=True), atol=1e-5)

        # Test gradients
        jax_grad = grad(lambda x: (jaxified_module(x) ** 2).sum())(input_batch)

        x = j2t(input_batch)
        x.requires_grad = True
        model(x).pow(2).sum().backward()
        aac(jax_grad, x.grad)


def test_torch_nn_MaxPool2d():
  for kernel_size in [1, 2, 3, 4, 5, (1, 2), (2, 4), (3, 2), (4, 1), (5, 5)]:
    for stride in [None, 1, (1, 1), 2, (2, 4), 3, (3, 2)]:
      for padding in [0, 1, (2, 7), (5, 1)]:
        model = torch.nn.MaxPool2d(kernel_size, stride, padding)
        input_batch = random.normal(random.PRNGKey(123), (7, 2, 16, 16))
        try:
          res_torch = model(j2t(input_batch))
        except Exception:
          # RuntimeError: pad should be at most half of kernel size, but got pad=7 and kernel_size=5
          continue

        jaxified_module = t2j(model)
        res_jax = jaxified_module(input_batch)
        res_jax_jit = jit(jaxified_module)(input_batch)

        # Test forward pass with and without jax.jit
        aac(res_jax, res_torch.numpy(force=True), atol=1e-5)
        aac(res_jax_jit, res_torch.numpy(force=True), atol=1e-5)

        # Test gradients
        jax_grad = grad(lambda x: (jaxified_module(x) ** 2).sum())(input_batch)

        x = j2t(input_batch)
        x.requires_grad = True
        model(x).pow(2).sum().backward()
        aac(jax_grad, x.grad)


def test_torch_nn_ConvTranspose2d():
  for in_channels in [1, 2]:
    for out_channels in [1, 2]:
      for kernel_size in [1, 2, (1, 2)]:
        for stride in [1, 2, (1, 2)]:
          for padding in [(0, 0), 1, 2, (1, 2)]:
            for output_padding in [0, 1, 2, (1, 2)]:
              for bias in [False, True]:
                for dilation in [1, 2, (1, 2)]:
                  model = torch.nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                    bias=bias,
                    dilation=dilation,
                  )
                  params = {k: random.normal(random.PRNGKey(123), v.shape) for k, v in model.named_parameters()}
                  model.load_state_dict({k: j2t(v) for k, v in params.items()})

                  input_batch = random.normal(random.PRNGKey(123), (3, in_channels, 16, 16))
                  try:
                    res_torch = model(j2t(input_batch))
                  except RuntimeError:
                    # RuntimeError: output padding must be smaller than either stride or dilation
                    continue

                  res_jax = t2j(model)(input_batch, state_dict=params)
                  aac(res_jax, res_torch.numpy(force=True), atol=1e-4)


def test_torch_nn_Dropout():
  model = torch.nn.Dropout()
  input_batch = random.normal(random.PRNGKey(123), (1024,))
  res_torch = model(j2t(input_batch))
  res_jax = t2j(model)(input_batch, rng=random.PRNGKey(123))
  anac(res_jax, res_torch.numpy(force=True))

  model = torch.nn.Dropout(p=0.0)
  input_batch = random.normal(random.PRNGKey(123), (1024,))
  res_torch = model(j2t(input_batch))
  res_jax = t2j(model)(input_batch, rng=random.PRNGKey(123))
  aac(res_jax, res_torch.numpy(force=True))
  aac(res_jax, input_batch)

  model = torch.nn.Dropout(p=1.0)
  input_batch = random.normal(random.PRNGKey(123), (1024,))
  res_torch = model(j2t(input_batch))
  res_jax = t2j(model)(input_batch, rng=random.PRNGKey(123))
  aac(res_jax, res_torch.numpy(force=True))
  aac(res_jax, jnp.zeros_like(res_jax))

  model = torch.nn.Dropout(p=0.1)
  input_batch = random.normal(random.PRNGKey(123), (1024,))
  res_torch = model(j2t(input_batch))
  res_jax = t2j(model)(input_batch, rng=random.PRNGKey(123))
  anac(res_jax, res_torch.numpy(force=True))
  assert jnp.mean(res_jax == res_torch.numpy(force=True)) > 0.5


################################################################################
# Models


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

  model = torchvision.models.resnet18(weights="DEFAULT").eval()

  parameters = {k: t2j(v) for k, v in model.named_parameters()}
  buffers = {k: t2j(v) for k, v in model.named_buffers()}

  rp = RngPooper(random.PRNGKey(123))

  input_batch = random.normal(rp.poop(), (5, 3, 224, 224))
  res_torch = model(j2t(input_batch))

  jaxified_module = t2j(model)
  res_jax = jaxified_module(input_batch, state_dict={**parameters, **buffers})
  res_jax_jit = jit(jaxified_module)(input_batch, state_dict={**parameters, **buffers})

  # Test forward pass with and without jax.jit
  aac(res_jax, res_torch.numpy(force=True), atol=1e-5)
  aac(res_jax_jit, res_torch.numpy(force=True), atol=1e-5)

  # Models use different convolution backends and are too deep to compare gradients programmatically. But they line up
  # to reasonable expectations.


@pytest.mark.skip(reason="https://github.com/jax-ml/jax/issues/25066")
@pytest.mark.skipif(not is_network_reachable(), reason="Network is not reachable")
def test_torchvision_models_vit_b_16():
  import torchvision

  model = torchvision.models.vit_b_16(weights="DEFAULT")
  model.eval()

  parameters = {k: t2j(v) for k, v in model.named_parameters()}
  # buffers = {k: t2j(v) for k, v in model.named_buffers()}
  # assert len(buffers.keys()) == 0

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
  # to reasonable expectations.
