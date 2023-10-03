# torch2jax

Run PyTorch in JAX. 🤝

Mix-and-match PyTorch and JAX code with seamless, end-to-end autodiff, use JAX classics like `jit`, `grad`, and `vmap` on PyTorch code, and run PyTorch models on TPUs.

torch2jax uses abstract interpretation (aka tracing) to move JAX values through PyTorch code. As a result, you get a JAX-native computation graph that follows _exactly_ your PyTorch code, down to the last epsilon.

```python
from torch2jax import j2t, t2j

vit = torchvision.models.vit_b_16().eval()
batch = torch.randn(1, 3, 224, 224)
vit(x)
# => [-5.3352e-01, ..., 2.0390e-01]

jax_vit = t2j(vit)
jax_batch = t2j(batch)
params = {k: t2j(v) for k, v in vit.named_parameters()}
jit(jax_vit)(jax_batch, state_dict=params)
# => [-5.3125e-01, ..., 2.0735e-01]
```

torch2jax even works with in-place PyTorch operations:

```python
def f(x):
    x.add_(1)
    x.mul_(2)
    return x

f(torch.Tensor([3]))                # => torch.Tensor([8])

jax_f = t2j(f)
jax_f(jnp.array([3]))               # => jnp.array([8])
vmap(jax_f)(jnp.array([1, 2, 3]))   # => jnp.array([[4], [6], [8]])
grad(jax_f)(jnp.array([2.0]))       # => jnp.array([2.0])
```

torch2jax offers a simple API with two functions:

1. `j2t`: Convert a JAX `jax.numpy.ndarray` to a `torch.Tensor`.
2. `t2j`: Convert a PyTorch function, `torch.nn.Module`, or `torch.Tensor` to their JAX equivalent.

Internally, the core of torch2jax is `Torchish`, a class that mimics `torch.Tensor` via [`__torch_function__`](https://pytorch.org/docs/stable/notes/extending.html#operations-on-multiple-types-that-define-torch-function). A `Torchish` object is backed by a JAX `jax.numpy.ndarray`, and proxies PyTorch operations onto the underlying `jax.numpy.ndarray`. As a result, you get a JAX-native computation graph that exactly follows your PyTorch code.

## Installation

### PyPI

```
pip install torch2jax
```

### Nix flake

torch2jax is available as a [Nix](https://nixos.org/) [flake](https://www.tweag.io/blog/2020-05-25-flakes/).

```
$ nix shell github:samuela/torch2jax
(shell) $ python -c "from torch2jax import j2t, t2j"
```

## FAQ

### Help! I've encountered a PyTorch operation that isn't implemented yet.

torch2jax is an implementation of the PyTorch standard library written in JAX. If you come across an operation that isn't implemented yet, please file an issue and/or PR!

Adding new PyTorch operations is straightforward. Check the source for functions decorated with `@implements` to get started.

### My PyTorch model includes dropout (or some other random operation), and does not work in training mode. Why?

JAX mandates [deterministic randomness](https://jax.readthedocs.io/en/latest/jax-101/05-random-numbers.html), while PyTorch does not. This leads to some API friction. torch2jax does not currently offer a means to bridge this gap. I have an idea for how to accomplish it. If this is important to you, please open an issue.

In the meantime, make sure to call `.eval()` on your `torch.nn.Module` before conversion.

### My PyTorch model includes batch norm (or some other `torch.nn.Module` utilizing buffers), and does not work in training mode. What can I do?

Similar to the randomness story, PyTorch and JAX have different approaches to maintaining state. Operations like batch norm require maintaining running statistics. In PyTorch, this is accomplished via [buffers](https://stackoverflow.com/questions/57540745/what-is-the-difference-between-register-parameter-and-register-buffer-in-pytorch/57546078#57546078).

torch2jax supports running batch norm models in `eval()`-mode. Just don't forget that you should avoid taking gradients w.r.t. buffers. For example,

```python
rn18 = torchvision.models.resnet18(weights="DEFAULT").eval()
loss = lambda x: torch.sum(x ** 2)

batch = torch.randn(1, 3, 224, 224)
loss(rn18(batch)).backward()

parameters = {k: t2j(v) for k, v in rn18.named_parameters()}
buffers = {k: t2j(v) for k, v in rn18.named_buffers()}

jax_rn18 = t2j(rn18)
grad(lambda params, x: loss(jax_rn18(x, state_dict={**params, **buffers})))(parameters, t2j(batch))
```

I have an idea for how to implement buffers, including in training mode. If this is important to you, please open an issue.

### I'm seeing slightly different numerical results between PyTorch and JAX. Is it a bug?

Floating point arithmetic is hard. There are a number of sources of divergence preventing bit-for-bit equivalence:

1. torch2jax guarantees equivalence with PyTorch standard library functions in the mathematical sense, but not necessarily in their operational execution. This can lead to slight differences in results.
2. The JAX/XLA and PyTorch compilers apply different optimizations and should be expected to rewrite computation graphs in exciting and unpredictable ways, potentially invoking different CUDA kernels.
3. CUDA kernels can be non-deterministic, for example as a result of floating point addition being non-associative.

Also bear in mind that floating point errors compound, so larger models will experience increased divergence.

### What about going the other way around? Running JAX code in PyTorch?

Check out [jax2torch](https://github.com/lucidrains/jax2torch).

## Contributing

PyTorch has a non-trivial API surface to cover. Contributions are welcome!

Run the test suite with `pytest` running in `nix shell`. Format imports with `isort --dont-follow-links .`.
