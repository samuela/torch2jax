import copy
import functools
from collections import deque
from contextlib import contextmanager
from functools import partial
from typing import Literal, Optional, Sequence, Tuple, Union

import jax
import jax.dlpack
import jax.numpy as jnp
import numpy as np
import torch
from torch.overrides import TorchFunctionMode, resolve_name
from torch.utils._pytree import register_pytree_node as torch_register_pytree_node
from torch.utils._pytree import tree_map as torch_tree_map
from torch.utils._pytree import tree_structure as torch_tree_structure

# so that __getitem__ & __setitem__ with mixed keys of int / tensor could work
torch_register_pytree_node(slice, lambda s: ((s.start, s.stop, s.step), None), lambda values, ctx: slice(*values))


class RngPooper:
  """A stateful wrapper around stateless jax.random.PRNGKey's."""

  def __init__(self, init_rng: jax.random.PRNGKey):
    self.rng = init_rng

  def poop(self) -> jax.random.PRNGKey:
    self.rng, rng_key = jax.random.split(self.rng)
    return rng_key


_RNG_POOPER_STACK = []


def mk_rng() -> jax.random.PRNGKey:
  assert len(_RNG_POOPER_STACK) > 0, "Attempted `mk_rng()` outside of a `RngPooperContext`"
  assert _RNG_POOPER_STACK[-1] is not None, (
    "Attempted `mk_rng()` with a `None` `RngPooperContext`. You're probably seeing this error message because you forgot to include a `rng` kwarg in your function call: `t2j(f)(..., rng=jax.random.PRNGKey(0))`. "
  )
  return _RNG_POOPER_STACK[-1].poop()


@contextmanager
def RngPooperContext(value: RngPooper | None):
  _RNG_POOPER_STACK.append(value)
  try:
    yield
  finally:
    _RNG_POOPER_STACK.pop()


def t2j_array(torch_array):
  # Using dlpack here causes segfaults on eg `t2j(lambda x: torch.Tensor([3.0]) * x)(jnp.array([0.0]))` when we use
  # `torch.func.functionalize` in `t2j_function`. For now, we're avoiding `torch.func.functionalize`, but something to
  # be wary of in the future.

  # RuntimeError: Can't export tensors that require gradient, use tensor.detach()
  torch_array = torch_array.detach()

  # See https://github.com/google/jax/issues/8082.
  # torch_array = torch_array.contiguous()

  return jax.dlpack.from_dlpack(torch_array)

  # Alternative, but copying implementation:
  # Note FunctionalTensor.numpy() returns incorrect results, preventing us from using torch.func.functionalize.
  # return jnp.array(torch_array.numpy(force=True))


def j2t_array(jax_array):
  return torch.utils.dlpack.from_dlpack(jax_array)

  # Alternative, but copying implementation:
  # return torch.from_numpy(jax_array.asnumpy())


HANDLED_FUNCTIONS = {}


class Torchish:
  def __init__(self, value):
    self.value = value

  @property
  def value(self):
    return self._value

  @value.setter
  def value(self, val):
    # See https://github.com/google/jax/issues/2115 re `isinstance(value, jnp.ndarray)`.
    assert isinstance(val, jnp.ndarray) or isinstance(val, (np.ndarray, int, float)), (
      f"Tried to create Torchish with unsupported type: {type(val)}"
    )
    self._value = val if torch.is_grad_enabled() else jax.lax.stop_gradient(val)

  # In order for PyTorch to accept an object as one of its own and allow dynamic dispatch it must either subclass
  # `torch.Tensor` or have a `__torch_function__` method. We opt to take the method route. Dispatch logic is handled in
  # `TorchishMode`.
  @classmethod
  def __torch_function__(cls, func, types, args=(), kwargs=None):
    raise NotImplementedError(f"Torchish.__torch_function__: {func}")

  # fmt: off
  @property
  def device(self):
    return torch.device("cpu")

  @property
  def dtype(self) -> torch.dtype: return j2t_dtype(self.value.dtype)
  @property
  def ndim(self) -> int: return len(self.value.shape)
  @property
  def shape(self): return self.value.shape
  # fmt: on

  @property
  def T(self):
    # Infinite hang, disturbing:
    # return self.permute(*torch.tensor([1, 0]))

    return self.permute(*list(range(self.ndim))[::-1])

  @property
  def is_nested(self):
    # NOTE: we disallow instantiating with NestedTensors.
    return False

  def expand(self, *sizes):
    assert len(sizes) == self.ndim, "TODO: implement len(sizes) > self.ndim"
    newshape = [new if new != -1 else old for old, new in zip(self.shape, sizes)]
    for i, (old, new) in enumerate(zip(self.shape, sizes)):
      if old != 1:
        assert newshape[i] == old, (
          f"Attempted to expand dimension {i} from {old} to {new}. Cannot expand on non-singleton dimensions."
        )

    return Torchish(jnp.broadcast_to(self.value, newshape))

  # fmt: off
  def __add__(self, other): return Torchish(self.value + _coerce(other))
  def __bool__(self): return bool(self.value)
  def __getitem__(self, key): return Torchish(self.value.__getitem__(torch_tree_map(_coerce, key)))
  def __int__(self): return int(self.value)
  def __invert__(self): return torch.bitwise_not(self)
  def __lt__(self, other): return Torchish(self.value < _coerce(other))
  def __le__(self, other): return Torchish(self.value <= _coerce(other))
  def __eq__(self, other): return Torchish(self.value == _coerce(other))
  def __ne__(self, other): return Torchish(self.value != _coerce(other))
  def __gt__(self, other): return Torchish(self.value > _coerce(other))
  def __ge__(self, other): return Torchish(self.value >= _coerce(other))
  def __matmul__(self, other): return Torchish(self.value @ _coerce(other))
  def __mul__(self, other): return Torchish(self.value * _coerce(other))
  def __neg__(self): return Torchish(-self.value)
  def __pow__(self, other): return Torchish(self.value ** _coerce(other))
  def __radd__(self, other): return Torchish(_coerce(other) + self.value)
  def __rmatmul__(self, other): return Torchish(_coerce(other) @ self.value)
  def __rmul__(self, other): return Torchish(_coerce(other) * self.value)
  def __rsub__(self, other): return Torchish(_coerce(other) - self.value)
  def __setitem__(self, key, value):
    self.value = self.value.at[torch_tree_map(_coerce, key)].set(_coerce(value))
  def __sub__(self, other): return Torchish(self.value - _coerce(other))

  def __or__(self, other): return Torchish(self.value | _coerce(other))
  def __and__(self, other): return Torchish(self.value & _coerce(other))
  def __xor__(self, other): return Torchish(self.value ^ _coerce(other))
  # For some reason `foo = torch.foo` doesn't work on these
  def contiguous(self): return self
  def detach(self): return Torchish(jax.lax.stop_gradient(self.value))
  def dim(self): return self.ndim
  def float(self): return Torchish(jnp.astype(self.value, jnp.float32))
  def item(self): return self.value.item()
  def permute(self, *shape): return torch.permute(self, shape)
  def size(self): return self.shape
  def type_as(self, other): return Torchish(jnp.astype(self.value, other.value.dtype))
  # fmt: on

  def view(self, *shape_or_dtype):
    if len(shape_or_dtype) == 1:
      if isinstance(shape_or_dtype[0], Sequence):
        return Torchish(self.value.reshape(shape_or_dtype[0]))
      elif isinstance(dtype := shape_or_dtype[0], torch.dtype):
        return Torchish(self.value.view(t2j_dtype(dtype)))
      else:
        raise ValueError(f"Tensor.view takes shape or dtype, got {shape_or_dtype[0]}")

    return Torchish(jnp.reshape(self.value, shape_or_dtype))

  def to(self, *args, **kwargs):
    # ignore device movement, jax manages its own placement
    if len(args) > 0 and isinstance(args[0], torch.dtype):
      return Torchish(jnp.astype(self.value, t2j_dtype(args[0])))
    if dtype := kwargs.get("dtype"):
      return Torchish(jnp.astype(self.value, t2j_dtype(dtype)))
    return self

  reshape = view

  def bernoulli_(self, p=0.5):
    # Torch accepts ints, floats, and even torch.Tensor's for p, but jax.numpy only accepts floats, so we convert.
    p = p.item() if isinstance(p, Torchish) else float(p)
    self.value = jax.random.bernoulli(mk_rng(), shape=self.shape, p=p).astype(self.value.dtype)
    return self

  def uniform_(self, a, b):
    self.value = jax.random.uniform(mk_rng(), shape=self.shape, dtype=self.value.dtype, minval=a, maxval=b)
    return self


def _coerce(x):
  """Coerce an input into something JAX-compatible.

  There are functions like `torch.pow` which accept Python ints, Python floats,
  or `torch.Tensor`s, so some fuss is necessary to get everything
  JAX-compatible."""
  if isinstance(x, Torchish):
    return x.value
  elif isinstance(x, (int, float, np.ndarray, jnp.ndarray)):  # jax compatible types
    return x
  elif any(x is e for e in (None, Ellipsis)):  # jax compatible special values
    return x
  elif isinstance(x, torch.dtype):
    return t2j_dtype(x)
  else:
    raise NotImplementedError(
      f"Attempted to _coerce with {x}, type {type(x)}. Don't know what to do with that. _coerce supports int, float, numpy arrays, None, Ellipsis, and Torchish values."
    )


def _tree_coerce(x):
  spec = torch_tree_structure(x)
  specs = deque([spec])

  while len(specs) > 0:
    spec = specs.popleft()
    specs.extend(spec.children_specs)
    if spec.is_leaf():
      continue
    if spec.type in jax._src.tree_util._registry:
      continue  # already registered
    node = torch.utils._pytree.SUPPORTED_NODES.get(spec.type, None)
    if node is None:
      continue

    def flip_args(fn):
      return lambda a, b: fn(b, a)

    jax.tree_util.register_pytree_node(
      spec.type,
      node.flatten_fn,
      flip_args(node.unflatten_fn),  # torch & jax have different order of args
    )

  return jax.tree.map(_coerce, x)


def _v(x):
  assert isinstance(x, Torchish)
  return x.value


def _args_to_shape(args):
  assert len(args) >= 1
  return args if isinstance(args[0], int) else args[0]


def implements(torch_function, Torchishify_output=True, out_kwarg=False, Torchish_member=False):
  """Register a torch function override"""

  def decorator(func):
    if out_kwarg:

      def func1(*args, out=None, **kwargs):
        if out is not None:
          out.value = func(*args, **kwargs)
          return out
        else:
          return Torchish(func(*args, **kwargs))
    elif Torchishify_output:
      func1 = lambda *args, **kwargs: Torchish(func(*args, **kwargs))
    else:
      func1 = func
    functools.update_wrapper(func1, torch_function)
    HANDLED_FUNCTIONS[torch_function] = func1
    if Torchish_member:
      name = torch_function.__name__
      if hasattr(torch.Tensor, name):
        setattr(Torchish, name, func1)
      if hasattr(torch.Tensor, name + "_"):

        def func1_(self, *args, **kwargs):
          self.value = func1(self, *args, **kwargs).value
          return self

        setattr(Torchish, name + "_", func1_)

    return func1

  return decorator


def auto_implements(torch_function, jax_function, dont_coerce_argnums=(), out_kwarg=False, Torchish_member=False):
  @implements(torch_function, out_kwarg=out_kwarg, Torchish_member=Torchish_member)
  def fn(*args, **kwargs):
    # NOTE: we don't _coerce values in kwargs! So far this has not been problematic.
    return jax_function(
      *(arg if i in dont_coerce_argnums else _coerce(arg) for i, arg in enumerate(args)),
      **kwargs,
    )


auto_implements(torch.abs, jnp.abs, out_kwarg=True, Torchish_member=True)
auto_implements(torch.add, jnp.add, out_kwarg=True, Torchish_member=True)
auto_implements(torch.cos, jnp.cos, out_kwarg=True, Torchish_member=True)
auto_implements(torch.clone, lambda x: x, Torchish_member=True)  # jax arrays are immutable, no copy needed
auto_implements(torch.div, jnp.divide, out_kwarg=True, Torchish_member=True)
auto_implements(torch.exp, jnp.exp, out_kwarg=True, Torchish_member=True)
auto_implements(torch.nn.functional.gelu, jax.nn.gelu)
auto_implements(torch.mul, jnp.multiply, out_kwarg=True, Torchish_member=True)
auto_implements(torch.nan_to_num, jnp.nan_to_num, out_kwarg=True, Torchish_member=True)
# Tensor.permute has a different signature than torch.permute
auto_implements(torch.permute, jnp.transpose, dont_coerce_argnums=(1, 2))  # TODO: do we need argnum 2?
auto_implements(torch.pow, jnp.power, out_kwarg=True, Torchish_member=True)
auto_implements(torch.rsqrt, jax.lax.rsqrt, out_kwarg=True, Torchish_member=True)
auto_implements(torch.sigmoid, jax.nn.sigmoid, out_kwarg=True, Torchish_member=True)
auto_implements(torch.sin, jnp.sin, out_kwarg=True, Torchish_member=True)
auto_implements(torch.sqrt, jnp.sqrt, out_kwarg=True, Torchish_member=True)
auto_implements(torch.sub, jnp.subtract, out_kwarg=True, Torchish_member=True)
auto_implements(torch.tanh, jnp.tanh, out_kwarg=True, Torchish_member=True)
auto_implements(torch.transpose, jnp.swapaxes, Torchish_member=True)


@implements(torch.all, Torchish_member=True, out_kwarg=True)
def _all(input, dim=None, keepdim=False):
  return jnp.all(_v(input), axis=dim, keepdims=keepdim)


@implements(torch.any, out_kwarg=True, Torchish_member=True)
def _any(input, dim=None, keepdim=False):
  return jnp.any(_v(input), axis=dim, keepdims=keepdim)


@implements(torch._assert, Torchishify_output=False)
def _assert(condition, message):
  if not condition:
    raise AssertionError(message)


@implements(torch.arange, out_kwarg=True)
def arange(*args, **kwargs):
  dtype = t2j_dtype(
    kwargs.get(
      "dtype",
      torch.get_default_dtype()
      if (
        isinstance(args[0], float)
        or (len(args) > 1 and isinstance(args[1], float))
        or (len(args) > 2 and isinstance(args[2], float))
      )
      else torch.int64,
    )
  )
  # TODO: test this with kwargs
  if len(args) == 1:
    return jnp.arange(args[0], dtype=dtype)
  elif len(args) == 2:
    return jnp.arange(args[0], args[1], dtype=dtype)
  elif len(args) == 3:
    return jnp.arange(args[0], args[1], args[2], dtype=dtype)
  else:
    raise ValueError("torch.arange takes 1-3 arguments")


@implements(torch.bernoulli, out_kwarg=True)  # don't set member because Tensor.bernoulli has different signature
def bernoulli(input, generator=None):
  assert generator is None, "TODO: implement `generator`"
  return jax.random.bernoulli(mk_rng(), p=_v(input))


@implements(torch.bitwise_not, out_kwarg=True, Torchish_member=True)
def bitwise_not(input):
  return jnp.invert(_v(input))


@implements(torch.cat, out_kwarg=True)
def cat(tensors, dim=0):
  return jnp.concatenate([_v(x) for x in tensors], axis=dim)


@implements(torch.device, Torchishify_output=False)
def device(device):
  # device doesn't matter to jax at all, because jax has its own implicit device
  # management, the user has no mechanism to do something like `to(device)`.
  # Therefore, we always return a CPU device, which makes the torch side to be
  # always consistent.
  return torch.device("cpu")


@implements(torch.empty, out_kwarg=True)
def empty(
  *args,
  dtype=None,
  layout=torch.strided,
  device=None,
  requires_grad=False,
  pin_memory=False,
  memory_format=torch.contiguous_format,
):
  return jnp.empty(_args_to_shape(args), dtype=t2j_dtype(dtype or torch.get_default_dtype()))


@implements(torch.flatten, Torchish_member=True)
def flatten(input, start_dim=0, end_dim=-1):
  assert end_dim == -1, "TODO: implement end_dim"
  return jnp.reshape(_v(input), input.shape[:start_dim] + (-1,))


@implements(torch.logical_and, out_kwarg=True, Torchish_member=True)
def logical_and(input, other):
  return jnp.logical_and(_v(input), _v(other))


@implements(torch.logical_or, out_kwarg=True, Torchish_member=True)
def logical_or(input, other):
  return jnp.logical_or(_v(input), _v(other))


@implements(torch.logical_not, out_kwarg=True, Torchish_member=True)
def logical_not(input):
  return jnp.logical_not(_v(input))


@implements(torch.logical_xor, out_kwarg=True, Torchish_member=True)
def logical_xor(input, other):
  return jnp.logical_xor(_v(input), _v(other))


@implements(torch.multinomial, out_kwarg=True, Torchish_member=True)
def multinomial(input, num_samples, replacement=False, generator=None):
  assert generator is None, "TODO: implement `generator`"

  if input.ndim == 1:
    N = input.shape[0]
    return jax.random.choice(mk_rng(), N, shape=(num_samples,), replace=replacement, p=_v(input) / _v(input).sum())
  elif input.ndim == 2:
    m, N = input.shape
    rngs = jax.random.split(mk_rng(), m)
    return jax.vmap(
      lambda rng, w: jax.random.choice(rng, N, shape=(num_samples,), replace=replacement, p=w / w.sum()),
      in_axes=(0, 0),
    )(rngs, _v(input))
  else:
    raise ValueError(f"unsupported shape: {input.shape}")


@implements(torch.mean, Torchish_member=True, out_kwarg=True)
def mean(input, dim=None, keepdim=False, dtype=None):
  dtype = t2j_dtype(dtype) if dtype is not None else None
  return jnp.mean(_v(input), axis=dim, keepdims=keepdim, dtype=dtype)


@implements(torch.normal, out_kwarg=True)
def normal(*args, **kwargs):
  assert kwargs.get("generator", None) is None, "TODO: implement `generator`"
  assert len(args) <= 3, f"too many arguments to normal: {args}"

  mean = _coerce(kwargs.get("mean", args[0] if len(args) > 0 else 0.0))
  std = _coerce(kwargs.get("std", args[1] if len(args) > 1 else 1.0))
  shape = kwargs.get(
    "size",
    args[2]
    if len(args) == 3
    else (mean.shape if isinstance(mean, jnp.ndarray) else (std.shape if isinstance(std, jnp.ndarray) else None)),
  )

  return (
    jax.random.normal(
      mk_rng(),
      shape=shape,
      dtype=t2j_dtype(kwargs.get("dtype", torch.get_default_dtype())),
    )
    * std
    + mean
  )


@implements(torch.numel, Torchishify_output=False, Torchish_member=True)
def numel(input):
  return _v(input).size


@implements(torch.ones, out_kwarg=True)
def ones(*args, dtype=None, layout=torch.strided, device=None, requires_grad=False):
  return jnp.ones(_args_to_shape(args), dtype=t2j_dtype(dtype or torch.get_default_dtype()))


@implements(torch.ones_like)
def ones_like(
  input,
  dtype=None,
  layout=None,
  device=None,
  requires_grad=False,
  memory_format=torch.preserve_format,
):
  assert not requires_grad
  return jnp.ones_like(_v(input), dtype=t2j_dtype(dtype or input.dtype))


@implements(torch.poisson)
def poisson(input, generator=None):
  assert generator is None, "TODO: implement `generator`"
  return jax.random.poisson(mk_rng(), lam=_v(input))


@implements(torch.rand, out_kwarg=True)
def rand(
  *args,
  generator=None,
  dtype=None,
  layout=torch.strided,
  device=None,
  requires_grad=False,
  pin_memory=False,
):
  assert generator is None, "TODO: implement `generator`"
  return jax.random.uniform(
    mk_rng(),
    shape=_args_to_shape(args),
    dtype=t2j_dtype(dtype or torch.get_default_dtype()),
  )


@implements(torch.rand_like)
def rand_like(input, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format):
  return jax.random.uniform(mk_rng(), shape=input.shape, dtype=t2j_dtype(dtype or input.dtype))


@implements(torch.randint, out_kwarg=True)
def randint(*args, **kwargs):
  assert kwargs.get("generator", None) is None, "TODO: implement `generator`"
  low = kwargs.get("low", args[0] if len(args) == 3 else 0)
  high = kwargs.get("high", args[1] if len(args) == 3 else args[0])
  shape = kwargs.get("size", args[-1])
  return jax.random.randint(
    mk_rng(), shape=shape, minval=low, maxval=high, dtype=t2j_dtype(kwargs.get("dtype", torch.int64))
  )


@implements(torch.randint_like)
def randint_like(*args, **kwargs):
  input = kwargs.get("input", args[0])
  low = kwargs.get("low", args[1] if len(args) == 3 else 0)
  high = kwargs.get("high", args[2] if len(args) == 3 else args[1])
  return jax.random.randint(
    mk_rng(), shape=input.shape, minval=low, maxval=high, dtype=t2j_dtype(kwargs.get("dtype", input.dtype))
  )


@implements(torch.randn, out_kwarg=True)
def randn(
  *args,
  generator=None,
  dtype=None,
  layout=torch.strided,
  device=None,
  requires_grad=False,
  pin_memory=False,
):
  assert generator is None, "TODO: implement `generator`"
  return jax.random.normal(
    mk_rng(),
    shape=_args_to_shape(args),
    dtype=t2j_dtype(dtype or torch.get_default_dtype()),
  )


@implements(torch.randn_like)
def randn_like(input, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format):
  return jax.random.normal(mk_rng(), shape=input.shape, dtype=t2j_dtype(dtype or input.dtype))


@implements(torch.randperm, out_kwarg=True)
def randperm(
  n: int,
  generator=None,
  dtype=torch.int64,
  layout=torch.strided,
  device=None,
  requires_grad=False,
  pin_memory=False,
):
  assert generator is None, "TODO: implement `generator`"
  return jax.random.permutation(mk_rng(), n).astype(dtype or torch.int64)


@implements(torch._C._set_grad_enabled, Torchishify_output=False)
def _set_grad_enabled(mode):
  torch._C._set_grad_enabled(mode)


def scatter_impl(input, dim, index, src, *, reduce=None):
  # Code adopted from https://github.com/jax-ml/jax/issues/8487#issuecomment-1555311635
  # in jax the dims other than dim for index & src should be equal,
  # but torch allows index to be smaller, we pad it with a out-of-bound index to match the src.
  out_of_range_idx = input.shape[dim]
  padding = tuple((0, d2 - d1) for d1, d2 in zip(index.shape, src.shape))
  index = jnp.pad(index, padding, constant_values=out_of_range_idx)
  dnums = jax.lax.ScatterDimensionNumbers(
    update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)
  )
  if reduce is None:
    _scatter = jax.lax.scatter
  elif reduce == "add":
    _scatter = jax.lax.scatter_add
  elif reduce == "multiply":
    _scatter = jax.lax.scatter_mul

  _scatter = partial(_scatter, dimension_numbers=dnums, mode="drop")
  vmap_inner = partial(jax.vmap, in_axes=(0, 0, 0), out_axes=0)

  for _ in range(len(input.shape) - 1):
    _scatter = vmap_inner(_scatter)
  swap = lambda x: jnp.swapaxes(x, dim, -1)
  input, index, src = list(map(swap, (input, index, src)))
  return swap(_scatter(input, jnp.expand_dims(index, axis=-1), src))


@implements(torch.scatter, Torchish_member=True)
def scatter(input, dim, index, src):
  return scatter_impl(_v(input), dim, _v(index), _v(src), reduce=None)


@implements(torch.scatter_add, Torchish_member=True)
def scatter_add(input, dim, index, src):
  return scatter_impl(_v(input), dim, _v(index), _v(src), reduce="add")


@implements(torch.sort, out_kwarg=True, Torchish_member=True)
def sort(input, dim=-1, descending=False, stable=False):
  return jnp.sort(_v(input), axis=dim, stable=stable, descending=descending)


@implements(torch.sum, Torchish_member=True, out_kwarg=True)
def sum(input, dim=None, keepdim=False, dtype=None):
  dtype = t2j_dtype(dtype) if dtype is not None else None
  return jnp.sum(_v(input), axis=dim, keepdims=keepdim, dtype=dtype)


@implements(torch.tensor)
def tensor(data, dtype=None, device=None, requires_grad=False, pin_memory=False):
  assert not requires_grad
  return jnp.array(
    data.value if isinstance(data, Torchish) else data, dtype=(t2j_dtype(dtype) if dtype is not None else None)
  )


@implements(torch.unbind, Torchishify_output=False, Torchish_member=True)
def unbind(input, dim=0) -> Sequence[Torchish]:
  return tuple(Torchish(input.value[(slice(None),) * dim + (i,)]) for i in range(input.value.shape[dim]))


@implements(torch.unsqueeze, Torchish_member=True)
def unsqueeze(input, dim):
  return jnp.expand_dims(_v(input), axis=dim)


@implements(torch.zeros, out_kwarg=True)
def zeros(*args, dtype=None, layout=torch.strided, device=None, requires_grad=False):
  assert not requires_grad
  return jnp.zeros(_args_to_shape(args), dtype=t2j_dtype(dtype or torch.get_default_dtype()))


@implements(torch.zeros_like)
def zeros_like(
  input,
  dtype=None,
  layout=None,
  device=None,
  requires_grad=False,
  memory_format=torch.preserve_format,
):
  assert not requires_grad
  return jnp.zeros_like(input, dtype=t2j_dtype(dtype or input.dtype))


################################################################################
# torch.nn.functional


@implements(torch.nn.functional.adaptive_avg_pool2d)
def adaptive_avg_pool2d(input, output_size):
  assert output_size == 1 or output_size == (1, 1), "TODO: implement output_size != 1"
  assert input.ndim == 4, "TODO: implement non-batched input"
  return jnp.mean(_v(input), axis=(2, 3), keepdims=True)


@implements(torch.nn.functional.batch_norm)
def batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
  assert isinstance(input, Torchish)
  assert isinstance(running_mean, Torchish)
  assert isinstance(running_var, Torchish)
  assert weight is None or isinstance(weight, Torchish)
  assert bias is None or isinstance(bias, Torchish)
  assert isinstance(momentum, float)
  assert isinstance(eps, float)

  x = input.value

  working_mean = running_mean.value
  working_var = running_var.value
  if training:
    x_ = x.reshape((x.shape[0], x.shape[1], -1))

    mean = jnp.mean(x_, axis=(0, 2))
    var = jnp.var(x_, axis=(0, 2), ddof=1)
    running_mean.value = jax.lax.stop_gradient(momentum * mean + (1 - momentum) * running_mean.value)
    running_var.value = jax.lax.stop_gradient(momentum * var + (1 - momentum) * running_var.value)

    # Why different ddof values are used for running_var and working_var I will never understand...
    working_mean = mean
    working_var = jnp.var(x_, axis=(0, 2), ddof=0)

  newshape = (1, -1) + (1,) * (len(x.shape) - 2)
  res = (x - working_mean.reshape(newshape)) * jax.lax.rsqrt(working_var.reshape(newshape) + eps)
  if weight is not None:
    res *= weight.value.reshape(newshape)
  if bias is not None:
    res += bias.value.reshape(newshape)
  return res


@implements(torch.nn.functional.conv2d)
def conv2d(
  input,
  weight,
  bias=None,
  stride=1,
  padding: Union[int, Tuple[int, int], Literal["same", "valid"]] = 0,
  dilation=1,
  groups=1,
):
  # jax.lax.conv_general_dilated supports different lo/hi padding, whereas PyTorch applies the same padding on both
  # sides. Note that we can't use the same trick as in conv_transpose2d since we also have to support "valid" and "same"
  # values for `padding`.
  if isinstance(padding, tuple):
    p1, p2 = padding
    padding = [(p1, p1), (p2, p2)]

  res = jax.lax.conv_general_dilated(
    lhs=_v(input),
    rhs=_v(weight),
    window_strides=stride,
    padding=padding,
    rhs_dilation=dilation,
    feature_group_count=groups,
  )
  if bias is not None:
    res += _v(bias)[jnp.newaxis, :, jnp.newaxis, jnp.newaxis]
  return res


@implements(torch.nn.functional.conv_transpose2d)
def conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
  # This implementation is taken from this PR https://github.com/google/jax/pull/5772
  assert input.ndim == 4, "TODO: implement non-batched input"
  assert groups == 1, "TODO: implement groups != 1"

  ph, pw = (padding, padding) if isinstance(padding, int) else padding
  res = gradient_based_conv_transpose(
    lhs=_v(input),
    rhs=_v(weight),
    strides=stride,
    padding=[(ph, ph), (pw, pw)],
    output_padding=output_padding,
    dilation=dilation,
    dimension_numbers=("NCHW", "OIHW", "NCHW"),
  )
  if bias is not None:
    res += _v(bias)[jnp.newaxis, :, jnp.newaxis, jnp.newaxis]
  return res


def _deconv_output_length(input_length, filter_size, padding, output_padding=None, stride=0, dilation=1):
  """Taken from https://github.com/google/jax/pull/5772
  Determines the output length of a transposed convolution given the input length.
  Function modified from Keras.
  Arguments:
      input_length: Integer.
      filter_size: Integer.
      padding: one of `"SAME"`, `"VALID"`, or a 2-integer tuple.
      output_padding: Integer, amount of padding along the output dimension. Can
        be set to `None` in which case the output length is inferred.
      stride: Integer.
      dilation: Integer.
  Returns:
      The output length (integer).
  """
  if input_length is None:
    return None

  # Get the dilated kernel size
  filter_size = filter_size + (filter_size - 1) * (dilation - 1)

  # Infer length if output padding is None, else compute the exact length
  if output_padding is None:
    if padding == "VALID":
      length = input_length * stride + jax.lax.max(filter_size - stride, 0)
    elif padding == "SAME":
      length = input_length * stride
    else:
      length = (input_length - 1) * stride + filter_size - padding[0] - padding[1]

  else:
    if padding == "SAME":
      pad = filter_size // 2
      total_pad = pad * 2
    elif padding == "VALID":
      total_pad = 0
    else:
      total_pad = padding[0] + padding[1]

    length = (input_length - 1) * stride + filter_size - total_pad + output_padding

  return length


def _compute_adjusted_padding(
  input_size: int,
  output_size: int,
  kernel_size: int,
  stride: int,
  padding: Union[str, Tuple[int, int]],
  dilation: int = 1,
) -> Tuple[int, int]:
  """
  Taken from https://github.com/google/jax/pull/5772
  Computes adjusted padding for desired ConvTranspose `output_size`.
  Ported from DeepMind Haiku.
  """
  kernel_size = (kernel_size - 1) * dilation + 1

  if padding == "VALID":
    expected_input_size = (output_size - kernel_size + stride) // stride
    if input_size != expected_input_size:
      raise ValueError(
        f"The expected input size with the current set of input "
        f"parameters is {expected_input_size} which doesn't "
        f"match the actual input size {input_size}."
      )
    padding_before = 0
  elif padding == "SAME":
    expected_input_size = (output_size + stride - 1) // stride
    if input_size != expected_input_size:
      raise ValueError(
        f"The expected input size with the current set of input "
        f"parameters is {expected_input_size} which doesn't "
        f"match the actual input size {input_size}."
      )
    padding_needed = jax.lax.max(0, (input_size - 1) * stride + kernel_size - output_size)
    padding_before = padding_needed // 2
  else:
    padding_before = padding[0]  # type: ignore[assignment]

  expanded_input_size = (input_size - 1) * stride + 1
  padded_out_size = output_size + kernel_size - 1
  pad_before = kernel_size - 1 - padding_before
  pad_after = padded_out_size - expanded_input_size - pad_before
  return (pad_before, pad_after)


def gradient_based_conv_transpose(
  lhs,
  rhs,
  strides: Sequence[int],
  padding: Union[str, Sequence[Tuple[int, int]]],
  output_padding: Optional[Sequence[int]] = None,
  output_shape: Optional[Sequence[int]] = None,
  dilation: Optional[Sequence[int]] = None,
  dimension_numbers: jax.lax.ConvGeneralDilatedDimensionNumbers = None,
  transpose_kernel: bool = True,
  precision=None,
):
  """
  Taken from https://github.com/google/jax/pull/5772
  Convenience wrapper for calculating the N-d transposed convolution.
  Much like `conv_transpose`, this function calculates transposed convolutions
  via fractionally strided convolution rather than calculating the gradient
  (transpose) of a forward convolution. However, the latter is more common
  among deep learning frameworks, such as TensorFlow, PyTorch, and Keras.
  This function provides the same set of APIs to help reproduce results in these frameworks.
  Args:
    lhs: a rank `n+2` dimensional input array.
    rhs: a rank `n+2` dimensional array of kernel weights.
    strides: sequence of `n` integers, amounts to strides of the corresponding forward convolution.
    padding: `"SAME"`, `"VALID"`, or a sequence of `n` integer 2-tuples that controls
      the before-and-after padding for each `n` spatial dimension of
      the corresponding forward convolution.
    output_padding: A sequence of integers specifying the amount of padding along
      each spacial dimension of the output tensor, used to disambiguate the output shape of
      transposed convolutions when the stride is larger than 1.
      (see a detailed description at
      1https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html)
      The amount of output padding along a given dimension must
      be lower than the stride along that same dimension.
      If set to `None` (default), the output shape is inferred.
      If both `output_padding` and `output_shape` are specified, they have to be mutually compatible.
    output_shape: Output shape of the spatial dimensions of a transpose
      convolution. Can be `None` or an iterable of `n` integers. If a `None` value is given (default),
      the shape is automatically calculated.
      Similar to `output_padding`, `output_shape` is also for disambiguating the output shape
      when stride > 1 (see also
      https://www.tensorflow.org/api_docs/python/tf/nn/conv2d_transpose)
      If both `output_padding` and `output_shape` are specified, they have to be mutually compatible.
    dilation: `None`, or a sequence of `n` integers, giving the
      dilation factor to apply in each spatial dimension of `rhs`. Dilated convolution
      is also known as atrous convolution.
    dimension_numbers: tuple of dimension descriptors as in
      lax.conv_general_dilated. Defaults to tensorflow convention.
    transpose_kernel: if `True` flips spatial axes and swaps the input/output
      channel axes of the kernel. This makes the output of this function identical
      to the gradient-derived functions like keras.layers.Conv2DTranspose and
      torch.nn.ConvTranspose2d applied to the same kernel.
      Although for typical use in neural nets this is unnecessary
      and makes input/output channel specification confusing, you need to set this to `True`
      in order to match the behavior in many deep learning frameworks, such as TensorFlow, Keras, and PyTorch.
    precision: Optional. Either ``None``, which means the default precision for
      the backend, a ``lax.Precision`` enum value (``Precision.DEFAULT``,
      ``Precision.HIGH`` or ``Precision.HIGHEST``) or a tuple of two
      ``lax.Precision`` enums indicating precision of ``lhs``` and ``rhs``.
  Returns:
    Transposed N-d convolution.
  """
  assert len(lhs.shape) == len(rhs.shape) and len(lhs.shape) >= 2
  ndims = len(lhs.shape)
  one = (1,) * (ndims - 2)
  # Set dimensional layout defaults if not specified.
  if dimension_numbers is None:
    if ndims == 2:
      dimension_numbers = ("NC", "IO", "NC")
    elif ndims == 3:
      dimension_numbers = ("NHC", "HIO", "NHC")
    elif ndims == 4:
      dimension_numbers = ("NHWC", "HWIO", "NHWC")
    elif ndims == 5:
      dimension_numbers = ("NHWDC", "HWDIO", "NHWDC")
    else:
      raise ValueError("No 4+ dimensional dimension_number defaults.")
  dn = jax.lax.conv_dimension_numbers(lhs.shape, rhs.shape, dimension_numbers)
  k_shape = jnp.take(jnp.array(rhs.shape), jnp.array(dn.rhs_spec))
  k_sdims = k_shape[2:]  # type: ignore[index]
  i_shape = jnp.take(jnp.array(lhs.shape), jnp.array(dn.lhs_spec))
  i_sdims = i_shape[2:]  # type: ignore[index]

  # Calculate correct output shape given padding and strides.
  if dilation is None:
    dilation = (1,) * (rhs.ndim - 2)

  if output_padding is None:
    output_padding = [None] * (rhs.ndim - 2)  # type: ignore[list-item]

  if isinstance(padding, str):
    if padding in {"SAME", "VALID"}:
      padding = [padding] * (rhs.ndim - 2)  # type: ignore[list-item]
    else:
      raise ValueError(f"`padding` must be 'VALID' or 'SAME'. Passed: {padding}.")

  inferred_output_shape = tuple(
    map(_deconv_output_length, i_sdims, k_sdims, padding, output_padding, strides, dilation)
  )
  if output_shape is None:
    output_shape = inferred_output_shape  # type: ignore[assignment]
  else:
    if not output_shape == inferred_output_shape:
      raise ValueError(
        f"`output_padding` and `output_shape` are not compatible."
        f"Inferred output shape from `output_padding`: {inferred_output_shape}, "
        f"but got `output_shape` {output_shape}"
      )

  pads = tuple(map(_compute_adjusted_padding, i_sdims, output_shape, k_sdims, strides, padding, dilation))

  if transpose_kernel:
    # flip spatial dims and swap input / output channel axes
    rhs = _flip_axes(rhs, dn.rhs_spec[2:])
    rhs = jnp.swapaxes(rhs, dn.rhs_spec[0], dn.rhs_spec[1])
  return jax.lax.conv_general_dilated(lhs, rhs, one, pads, strides, dilation, dn, precision=precision)


def _flip_axes(x, axes):
  """
  Taken from https://github.com/google/jax/pull/5772
  Flip ndarray 'x' along each axis specified in axes tuple."""
  for axis in axes:
    x = jnp.flip(x, axis)
  return x


@implements(torch.nn.functional.dropout)
def dropout(input, p=0.5, training=True, inplace=False):
  # p is the probability of an element to be zeroed
  assert 0 <= p <= 1, "dropout probability has to be between 0 and 1, but got {}".format(p)
  assert not inplace, "TODO: implement inplace=True"
  if training:
    # See https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html
    mask = jax.random.bernoulli(mk_rng(), p=1 - p, shape=_v(input).shape)
    res = jnp.where(mask, _v(input), 0)
    # Note that we have to avoid a divide by zero here when p is 1.
    return res / (1 - p) if p < 1 else res
  else:
    return _v(input)


@implements(torch.nn.functional.embedding)
def embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
  assert max_norm is None, "TODO: implement max_norm"
  assert not sparse, "TODO: implement sparse"
  input = _v(input)
  weight = _v(weight)

  if padding_idx is not None:
    weight = weight.at[padding_idx].set(jax.lax.stop_gradient(weight[padding_idx]))

  if scale_grad_by_freq:

    @jax.custom_jvp
    def f(weight):
      return weight[input]

    @f.defjvp
    def f_jvp(primals, tangents):
      w = primals[0]
      w_dot = tangents[0]
      primal_out = f(w)
      size = min(w.shape[0], input.size)
      indices, counts = jnp.unique_counts(input, size=size, fill_value=w.shape[0])
      inv_freq = 1.0 / counts
      slicing = (...,) + (jnp.newaxis,) * (weight.ndim - 1)
      inv_freq = inv_freq[slicing]
      w_dot = w_dot.at[indices].multiply(inv_freq, mode="drop")
      return primal_out, w_dot[input]

    return f(weight)

  return weight[input]


@implements(torch.nn.functional.layer_norm)
def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05):
  input = _v(input)

  d = len(normalized_shape)
  mean = jnp.mean(input, axis=tuple(range(input.ndim)[-d:]), keepdims=True)
  # NOTE: According to https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html, PyTorch calculates variance
  # without Bessel's correction. This is the default behavior in numpy, but we set `ddof=0` to be explicit.
  var = jnp.var(input, axis=tuple(range(input.ndim)[-d:]), keepdims=True, ddof=0)

  res = (input - mean) / jnp.sqrt(var + eps)
  if weight is not None:
    res *= _v(weight)
  if bias is not None:
    res += _v(bias)
  return res


@implements(torch.nn.functional.linear)
def linear(input, weight, bias=None):
  if bias is None:
    return _v(input) @ _v(weight).T
  else:
    return _v(input) @ _v(weight).T + _v(bias)


@implements(torch.nn.functional.max_pool1d)
def max_pool1d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
  assert dilation == 1, "TODO: implement dilation != 1"
  assert not ceil_mode, "TODO: implement ceil_mode"
  assert not return_indices, "TODO: implement return_indices"

  return jax.lax.reduce_window(
    _v(input),
    -jnp.inf,
    jax.lax.max,
    window_dimensions=(1, 1, kernel_size) if isinstance(kernel_size, int) else (1, 1) + kernel_size,
    window_strides=(1, 1, stride) if isinstance(stride, int) else (1, 1) + stride,
    padding=[(0, 0), (0, 0), (padding, padding)] if isinstance(padding, int) else [(0, 0), (0, 0), padding * 2],
  )


@implements(torch.nn.functional.max_pool2d)
def max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
  assert input.ndim == 4, "TODO: implement non-batched input"
  assert dilation == 1, "TODO: implement dilation != 1"
  assert not ceil_mode, "TODO: implement ceil_mode"
  assert not return_indices, "TODO: implement return_indices"

  # Coerce `padding: Int` -> `padding: Tuple[Int, Int]` if necessary.
  (pad_h, pad_w) = (padding, padding) if isinstance(padding, int) else padding
  return jax.lax.reduce_window(
    _v(input),
    -jnp.inf,
    jax.lax.max,
    # Note that these settings all rely on input.ndim == 4:
    window_dimensions=(1, 1, kernel_size, kernel_size) if isinstance(kernel_size, int) else (1, 1) + kernel_size,
    window_strides=(1, 1, stride, stride) if isinstance(stride, int) else (1, 1) + stride,
    padding=[(0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)],
  )


@implements(torch.nn.functional.relu, Torchishify_output=False, Torchish_member=True)
def relu(x, inplace=False):
  # Can't use `auto_implements` since jax.nn.relu does not have an `inplace` option.
  if inplace:
    assert isinstance(x, Torchish)
    x.value = jax.nn.relu(x.value)
    return x
  else:
    return Torchish(jax.nn.relu(_v(x)))


@implements(torch.nn.functional.silu, Torchishify_output=False)
def silu(x, inplace=False):
  if inplace:
    assert isinstance(x, Torchish)
    x.value = jax.nn.silu(x.value)
    return x
  else:
    return Torchish(jax.nn.silu(_v(x)))


@implements(torch.nn.functional.prelu, Torchish_member=True)
def prelu(input: Torchish, weight: Torchish):
  if weight.numel() != 1:
    assert input.ndim > 0, "Not allow zero-dim input tensor."
    channel_size = input.shape[1] if input.ndim >= 2 else 1
    assert weight.numel() == channel_size, (
      f"Mismatch of parameter numbers and input channel size. Found parameter numbers = {weight.numel()} and channel size = {channel_size}."
    )
  assert weight.ndim == 0 or weight.ndim == 1, (
    f"prelu: Expected `weight` to be a scalar or 1D tensor, but got: ndim = {weight.ndim}"
  )
  if input.ndim == 0:
    weight = weight[0] if weight.ndim == 1 else weight
  else:
    weight = Torchish(
      jax.lax.broadcast_in_dim(_v(weight), input.shape, () if weight.ndim == 0 else (0 if input.ndim == 1 else 1,))
    )
  return jnp.where(_v(input) > 0, _v(input), _v(input) * _v(weight))


@implements(torch.nn.functional.scaled_dot_product_attention)
def scaled_dot_product_attention(
  query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False
):
  # ignore enable_gqa argument, as this feature is automatically enabled on the
  # jax side when there are a different number of query heads as key/value heads
  assert dropout_p == 0.0, "TODO: implement dropout"
  Q, K, V = _v(query), _v(key), _v(value)
  # torch has (batch, num_heads, seq_len, head_dim) for Q, K, V
  # jax has (batch, seq_len, num_heads, head_dim)
  Q, K, V = jnp.swapaxes(Q, -2, -3), jnp.swapaxes(K, -2, -3), jnp.swapaxes(V, -2, -3)
  mask, bias = None, None
  if attn_mask is not None:
    attn_mask = _v(attn_mask)
    if jnp.issubdtype(attn_mask.dtype, jnp.bool_):
      mask = attn_mask
    elif jnp.issubdtype(attn_mask.dtype, jnp.floating):
      bias = attn_mask
    else:
      raise ValueError(f"Unsupported attn_mask dtype: {attn_mask.dtype}. Expected bool or float.")
  output = jax.nn.dot_product_attention(Q, K, V, scale=scale, mask=mask, bias=bias, is_causal=is_causal)
  output = jnp.swapaxes(output, -2, -3)
  if mask is not None:
    # when attn_mask are all false in a row, torch returns 0.
    # while jax simply adds large negative numbers to the logits
    # leading to an uniform attention. Here's some post processing
    # to align with torch behavior.
    # shape of output mask: (batch(optional), num_heads(optional), seq_len, 1)
    output_mask = jnp.any(mask, axis=-1, keepdims=True)
    output *= output_mask
  return output


# NOTE: the "torch.Tensor" type annotations here are a lie, or at least an approximation: In reality, they can be
# anything _coerce-able.
@implements(torch.nn.functional.multi_head_attention_forward, Torchishify_output=False)
def multi_head_attention_forward(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  embed_dim_to_check: int,
  num_heads: int,
  in_proj_weight: Optional[torch.Tensor],
  in_proj_bias: Optional[torch.Tensor],
  bias_k: Optional[torch.Tensor],
  bias_v: Optional[torch.Tensor],
  add_zero_attn: bool,
  dropout_p: float,
  out_proj_weight: torch.Tensor,
  out_proj_bias: Optional[torch.Tensor],
  training: bool = True,
  key_padding_mask: Optional[torch.Tensor] = None,
  need_weights: bool = True,
  attn_mask: Optional[torch.Tensor] = None,
  use_separate_proj_weight: bool = False,
  q_proj_weight: Optional[torch.Tensor] = None,
  k_proj_weight: Optional[torch.Tensor] = None,
  v_proj_weight: Optional[torch.Tensor] = None,
  static_k: Optional[torch.Tensor] = None,
  static_v: Optional[torch.Tensor] = None,
  average_attn_weights: bool = True,
  is_causal: bool = False,
):
  assert in_proj_weight is not None, "TODO: implement in_proj_weight=None"
  assert in_proj_bias is not None, "TODO: implement in_proj_bias=None"
  assert bias_k is None, "TODO: implement bias_k"
  assert bias_v is None, "TODO: implement bias_v"
  assert not add_zero_attn, "TODO: implement add_zero_attn=True"
  # dropout_p
  # out_proj_weight
  assert out_proj_bias is not None, "TODO implement out_proj_bias=None"
  assert not training, "TODO: implement training=True"
  assert key_padding_mask is None, "TODO: implement key_padding_mask"
  assert not need_weights, "TODO: implement need_weights=True"
  assert attn_mask is None, "TODO: implement attn_mask"
  assert not use_separate_proj_weight, "TODO: use_separate_proj_weight"
  assert q_proj_weight is None, "TODO: implement q_proj_weight"
  assert k_proj_weight is None, "TODO: implement k_proj_weight"
  assert v_proj_weight is None, "TODO: implement v_proj_weight"
  assert static_k is None, "TODO: implement static_k"
  assert static_v is None, "TODO: implement static_v"
  assert average_attn_weights, "TODO: implement average_attn_weights=False"
  assert not is_causal, "TODO: implement is_causal=True"

  Q, K, V = _v(query), _v(key), _v(value)
  assert Q.ndim == 3 and K.ndim == 3 and V.ndim == 3, "TODO: implement non-batched version"
  # For some asinine reason, the PyTorch calling signature is query (L, N, E), key (S, N, E), and value (S, N, E).
  Q, K, V = jnp.swapaxes(Q, 0, 1), jnp.swapaxes(K, 0, 1), jnp.swapaxes(V, 0, 1)

  in_proj_weight, in_proj_bias = _v(in_proj_weight), _v(in_proj_bias)
  out_proj_weight, out_proj_bias = _v(out_proj_weight), _v(out_proj_bias)

  w_q, w_k, w_v = jnp.split(in_proj_weight, 3)
  b_q, b_k, b_v = jnp.split(in_proj_bias, 3)
  # print(w_q.shape, w_k.shape, w_v.shape)  # (E, E) (E, E) (E, E)
  Q1, K1, V1 = Q @ w_q.T + b_q, K @ w_k.T + b_k, V @ w_v.T + b_v

  # print(Q1.shape, K1.shape, V1.shape)  # (N, L, E) (N, S, E) (N, S, E)
  sdpa = jnp.concatenate(
    tuple(
      _v(scaled_dot_product_attention(Torchish(q), Torchish(k), Torchish(v)))
      for q, k, v in zip(
        jnp.split(Q1, num_heads, axis=-1),
        jnp.split(K1, num_heads, axis=-1),
        jnp.split(V1, num_heads, axis=-1),
      )
    ),
    axis=-1,
  )
  # print(sdpa.shape)  # (N, L, E)
  out = sdpa @ out_proj_weight.T + out_proj_bias
  return Torchish(jnp.swapaxes(out, 0, 1)), None


class TorchishMode(TorchFunctionMode):
  def __torch_function__(self, func, types, args, kwargs=None):
    # print(f"Function Log: {resolve_name(func)}(*{args}, **{kwargs}) with types {types}")

    kwargs = kwargs or {}

    if func in HANDLED_FUNCTIONS:
      return HANDLED_FUNCTIONS[func](*args, **kwargs)
    else:
      raise NotImplementedError(
        f"Unhandled function call: {resolve_name(func)}(*{args}, **{kwargs}) with types {types}. Please submit a bug report at https://github.com/samuela/torch2jax/issues."
      )


@contextmanager
def override_Tensor_constructor():
  """Context manager to temporarily override torch.Tensor.__new__ construction
  while preserving isinstance checks and inheritance."""
  original_new = torch.Tensor.__new__

  @functools.wraps(original_new)
  def custom_new(cls, *args, **kwargs):
    raise ValueError(
      "Attempted to call `torch.Tensor.__new__`. You're probably seeing this error message because you called `torch.Tensor(...)` instead of `torch.tensor(...)` (note capitalization!). The `torch.Tensor(...)` constructor is deprecated and does not work with __torch_function__ and torch2jax. Please migrate to `torch.tensor(...)`."
    )

  torch.Tensor.__new__ = custom_new

  try:
    yield
  finally:
    torch.Tensor.__new__ = original_new


def t2j_function(f):
  def f_jax(*args, rng=None):
    torch_args = jax.tree.map(Torchish, args)
    with override_Tensor_constructor():
      with RngPooperContext(None if rng is None else RngPooper(rng)):
        with TorchishMode():
          out = f(*torch_args)
    # use the torch's tree_map, because out is generated from torch code
    return _tree_coerce(out)

  return f_jax


TJ_DTYPE_ASSOCIATION = [
  (torch.float16, jnp.float16),
  (torch.float32, jnp.float32),
  (torch.float64, jnp.float64),
  (torch.int8, jnp.int8),
  (torch.int16, jnp.int16),
  (torch.int32, jnp.int32),
  (torch.int64, jnp.int64),
  (torch.uint8, jnp.uint8),
  (torch.bool, jnp.bool_),
  (torch.complex64, jnp.complex64),
  (torch.complex128, jnp.complex128),
  (torch.bfloat16, jnp.bfloat16),
]


def t2j_dtype(dtype):
  return next(j_dtype for t_dtype, j_dtype in TJ_DTYPE_ASSOCIATION if t_dtype == dtype)


def j2t_dtype(dtype):
  return next(t_dtype for t_dtype, j_dtype in TJ_DTYPE_ASSOCIATION if j_dtype == dtype)


def t2j_module(module):
  def f(*args, state_dict={}, rng=None, return_state_dict=False):
    """Call the `torch.nn.Module` `module` with `*args`.

    Arguments:
    - `*args`: arguments to pass to the module.
    - `state_dict`: a dictionary of parameters to use for the module.
    - `rng`: a `jax.random.PRNGKey` to use for random number generation. Only
      necessary if the module forward pass uses PyTorch random functions like
      `torch.randn`.
    - `return_state_dict`: if `True`, return the updated state dict alongside
      the output, as in `y, after_sd = f(x, state_dict=before_sd)`. This is
      useful for modules that involve buffer mutation such as batch norm in
      training mode. Otherwise, return just the output, as in `y = f(x)`. Note
      that if you are `jax.jit`ting this function, you'll need to add
      `static_argnames=["return_state_dict"]` to the jit call.
    """
    # We want to have a non-mutating API, so we need to copy the module before performing parameter surgery. Note that
    # doing this copy in `t2j_module` and outside of `f` is not sufficient: multiple calls to `f` should not step on
    # each others toes.
    m = copy.deepcopy(module)

    # Can't use torch.func.functional_call due to https://github.com/pytorch/pytorch/issues/110249
    assert state_dict.keys() == dict(m.state_dict()).keys()

    reverse_dict = {}

    def visit(m, prefix):
      for name, param in m.named_parameters(recurse=False):
        if param in reverse_dict:
          # sometimes parameters are shared,
          # e.g. llm head shares embedding weights
          # we won't be able to get "llm.head" in the state_dict in this case.
          # because the underlying parameter is only registered once as "embedding.weight".
          m._parameters[name] = Torchish(state_dict[reverse_dict[param]])
        else:
          m._parameters[name] = Torchish(state_dict[".".join(prefix + [name])])
          reverse_dict[param] = ".".join(prefix + [name])

      for name, buffer in m.named_buffers(recurse=False):
        # buffers with register_buffer(persistent=False) won't appear in state_dict
        if ".".join(prefix + [name]) not in module.state_dict().keys():
          m._buffers[name] = Torchish(t2j(buffer))
        else:
          m._buffers[name] = Torchish(state_dict[".".join(prefix + [name])])

      # NOTE: named_children() is the non-recursive version of named_modules()
      for name, child in m.named_children():
        visit(child, prefix=prefix + [name])

    # Replace parameters with Torchish objects
    visit(m, prefix=[])

    out = t2j_function(m)(*args, rng=rng)
    if return_state_dict:
      return out, {k: v.value for k, v in m.state_dict().items()}
    else:
      return out

  return f


def t2j(thing):
  if isinstance(thing, torch.Tensor):
    return t2j_array(thing)
  elif isinstance(thing, torch.nn.Module):
    return t2j_module(thing)
  elif isinstance(thing, torch.dtype):
    return t2j_dtype(thing)
  elif callable(thing):
    # This branch must live below the torch.nn.Module branch!
    return t2j_function(thing)
  else:
    raise NotImplementedError


def j2t(thing):
  if isinstance(thing, jnp.ndarray):
    return j2t_array(thing)
  # We allow dtypes and "dtype constructors". It's subtle, but there's a difference. See https://github.com/jax-ml/jax/discussions/25497.
  if isinstance(thing, jnp.dtype) or thing in [dt for _, dt in TJ_DTYPE_ASSOCIATION]:
    return j2t_dtype(thing)
  else:
    raise NotImplementedError
