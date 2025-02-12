# torch2jax

Run PyTorch in JAX. 🤝

Mix-and-match PyTorch and JAX code with seamless, end-to-end autodiff, use JAX classics like `jit`, `grad`, and `vmap` on PyTorch code, and run PyTorch models on TPUs.

torch2jax uses abstract interpretation (aka tracing) to move JAX values through PyTorch code. As a result, you get a JAX-native computation graph that follows _exactly_ your PyTorch code, down to the last epsilon.

```python
from torch2jax import j2t, t2j

vit = torchvision.models.vit_b_16().eval()
batch = torch.randn(1, 3, 224, 224)
vit(batch)
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

f(torch.tensor([3]))                # => torch.Tensor([8])

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

### My PyTorch model includes dropout or some other random operation. How does this work with torch2jax?

Pass a `jax.random.PRNGKey` to the converted function:

```python
t2j(lambda: torch.randn(3))(rng=jax.random.PRNGKey(123))
# => [-0.56996626, -0.6440589 ,  0.28660855]

t2j(lambda: torch.randn(3))(rng=jax.random.PRNGKey(456))
# => [-1.3227656, -1.4896724, -2.5057693]
```

After conversion, random state will be handled entirely in JAX. `torch.manual_seed` and its ilk will have no effect on the converted function.

If you only care about running a model and not training it, you can call `.eval()` on it to avoid the randomness issue altogether, at least for most common random operations like dropout:

```python
rn18 = torchvision.models.resnet18().eval()
t2j(rn18)(t2j(torch.randn(1, 3, 224, 224)))     # Look ma, no `rng` kwarg!
```

> [!NOTE]
> Non-deterministic behavior is, well, non-deterministic. You will not see the same results with the same random seed when switching between PyTorch and JAX. However, the sampling process _will_ be equivalent.

### My PyTorch model includes batch norm or some other `torch.nn.Module` that mutates buffers. How does this work with torch2jax?

Some PyTorch modules like `torch.nn.BatchNorm1d` mutate internal state in the form of [buffers](https://stackoverflow.com/questions/57540745/what-is-the-difference-between-register-parameter-and-register-buffer-in-pytorch/57546078#57546078).

torch2jax supports this with the optional `return_state_dict` argument:

```python
rn18 = torchvision.models.resnet18()
batch = torch.randn(1, 3, 224, 224)

before_state_dict = {k: t2j(v) for k, v in rn18.state_dict().items()}
out, after_state_dict = t2j(rn18)(t2j(batch), state_dict=before_state_dict, return_state_dict=True)
```

As with randomness, if you only care about running a model and not training it, you can call `.eval()` on it to avoid buffer issues altogether in most cases.

Also, don't forget to avoid taking gradients w.r.t. buffers. For example,

```python
rn18 = torchvision.models.resnet18().eval()
loss = lambda x: torch.sum(x ** 2)

batch = torch.randn(1, 3, 224, 224)
loss(rn18(batch)).backward()

parameters = {k: t2j(v) for k, v in rn18.named_parameters()}
buffers = {k: t2j(v) for k, v in rn18.named_buffers()}

jax_rn18 = t2j(rn18)
grad(lambda params, x: loss(jax_rn18(x, state_dict={**params, **buffers})))(parameters, t2j(batch))
```

### I'm seeing slightly different numerical results between PyTorch and JAX. Is it a bug?

Floating point arithmetic is hard. There are a number of sources of divergence preventing bit-for-bit equivalence:

1. torch2jax guarantees equivalence with PyTorch standard library functions in the mathematical sense, but not necessarily in their operational execution. This can lead to slight differences in results. For example, the multi-head attention implementations calculate the same mathematical function, but may vary in execution details such as the order of operations, the use of fused kernels, and so forth.
2. The JAX/XLA and PyTorch compilers apply different optimizations and should be expected to rewrite computation graphs in exciting and unpredictable ways, potentially invoking different CUDA kernels.
3. CUDA kernels can be non-deterministic, for example as a result of floating point addition being non-associative.

Also bear in mind that floating point errors compound, so larger models will experience increased divergence.

### What about going the other way around? Running JAX code in PyTorch?

Check out [jax2torch](https://github.com/lucidrains/jax2torch).

## Contributing

PyTorch has a non-trivial API surface to cover. Contributions are welcome!

Run the test suite with `pytest` running in `nix develop`. Format the codebase with `ruff check --fix . && ruff format .`. Build the package with `nix build`.

CI is handled by GitHub Actions. When modifying the CI configuration, it can be handy to test locally before pushing. This can be achieved with [act](https://github.com/nektos/act/issues/269). Run `act` within `nix develop` to run the CI locally.

## License
torch2jax is licensed depending on your usecase. In general, torch2jax is licensed under the [GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html) license. That being said, torch2jax is also available under an [MIT](https://opensource.org/license/mit) license in the following contexts:

1. You are using torch2jax for personal, non-commercial use.
2. You are using torch2jax in a not-for-profit organization and for non-commercial use, eg. academia.
3. You are using torch2jax in a commercial context within a company of 25 people or fewer.

Please reach out to discuss licensing options if you are interested in using torch2jax under an MIT license in any other context. A portion of all proceeds go to axial spondyloarthritis (aka ankylosing spondylitis) research.
