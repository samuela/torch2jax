import copy
import functools
import math
from typing import Optional

import jax
import jax.dlpack
import jax.numpy as jnp
import torch


def t2j_array(torch_array):
  # Using dlpack here causes segfaults on eg `t2j(lambda x: torch.Tensor([3.0]) * x)(jnp.array([0.0]))` when we use
  # `torch.func.functionalize` in `t2j_function`. For now, we're avoiding `torch.func.functionalize`, but something to
  # be wary of in the future.

  # See https://github.com/google/jax/issues/8082.
  torch_array = torch_array.contiguous()
  return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(torch_array))

  # Alternative, but copying implementation:
  # Note FunctionalTensor.numpy() returns incorrect results, preventing us from using torch.func.functionalize.
  # return jnp.array(torch_array.numpy(force=True))

def j2t_array(jax_array):
  return torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(jax_array))

  # Alternative, but copying implementation:
  # return torch.from_numpy(jax_array.asnumpy())

HANDLED_FUNCTIONS = {}
class Torchish:
  def __init__(self, value):
    if isinstance(value, Torchish):
      self.value = value.value
    elif isinstance(value, jnp.ndarray) or isinstance(value, int) or isinstance(value, float):
      # See https://github.com/google/jax/issues/2115 re `isinstance(value, jnp.ndarray)`.
      self.value = value
    elif isinstance(value, torch.Tensor):
      assert not value.requires_grad, "cannot Torchish-ify requires_grad Tensors"
      assert not value.is_nested, "Torchish does not support NestedTensors"
      self.value = t2j_array(value)
    else:
      raise NotImplementedError(f"Attempted to instantiate Torchish with {value}. Don't know what to do with that. Torchish supports int, float, jax.numpy.ndarray, and torch.Tensor values.")

  @classmethod
  def __torch_function__(cls, func, types, args=(), kwargs=None):
    if kwargs is None:
      kwargs = {}
    if func not in HANDLED_FUNCTIONS or not all(issubclass(t, (torch.Tensor, Torchish)) for t in types):
      print("ERROR: you tried to do something not supported yet! Open a PR or issue on GitHub if you believe this should be included in torch2jax.")
      return NotImplemented
    # NOTE: some functions, like multi_head_attention_forward return a tuple of torch.Tensor's, and will even return
    # `None` in some configurations. so we cannot necessarily `Torchish` the output of `HANDLED_FUNCTIONS[func]`. Instead
    # the handler functions are responsible for outputting Torchish objects where appropriate.
    return HANDLED_FUNCTIONS[func](*args, **kwargs)

  @property
  def dtype(self): return self.value.dtype
  @property
  def ndim(self): return len(self.value.shape)
  @property
  def shape(self): return self.value.shape
  @property
  def T(self): return self.permute(*torch.arange(self.ndim - 1, -1, -1))

  @property
  def is_nested(self):
    # NOTE: we disallow instantiating with NestedTensors.
    return False

  def detach(self): return Torchish(jax.lax.stop_gradient(self.value))
  def dim(self): return self.ndim
  def item(self): return self.value.item()
  def size(self): return self.shape
  def view(self, *shape): return Torchish(jnp.reshape(self.value, shape))
  reshape = view
  def expand(self, *sizes):
    assert len(sizes) == self.ndim, "TODO: implement len(sizes) > self.ndim"
    newshape = [new if new != -1 else old for old, new in zip(self.shape, sizes)]
    for i, (old, new) in enumerate(zip(self.shape, sizes)):
      if old != 1:
        assert newshape[i] == old, f"Attempted to expand dimension {i} from {old} to {new}. Cannot expand on non-singleton dimensions."

    return Torchish(jnp.broadcast_to(self.value, newshape))

  def __add__(self, other): return Torchish(self.value + coerce(other))
  def __getitem__(self, key): return Torchish(self.value.__getitem__(key))
  def __matmul__(self, other): return Torchish(self.value @ coerce(other))
  def __mul__(self, other): return Torchish(self.value * coerce(other))
  def __pow__(self, other): return Torchish(self.value ** coerce(other))
  def __radd__(self, other): return Torchish(coerce(other) + self.value)
  def __rmatmul__(self, other): return Torchish(coerce(other) @ self.value)
  def __rmul__(self, other): return Torchish(coerce(other) * self.value)

  # For some reason `foo = torch.foo` doesn't work on these
  def flatten(*args, **kwargs): return torch.flatten(*args, **kwargs)
  def mean(*args, **kwargs): return torch.mean(*args, **kwargs)
  def permute(self, *shape): return torch.permute(self, shape)
  def pow(*args, **kwargs): return torch.pow(*args, **kwargs)
  def sum(*args, **kwargs): return torch.sum(*args, **kwargs)
  def transpose(*args, **kwargs): return torch.transpose(*args, **kwargs)

  def add_(self, other):
    self.value += other
    return self
  def sub_(self, other):
    self.value -= other
    return self
  def mul_(self, other):
    self.value *= other
    return self
  def div_(self, other):
    self.value /= other
    return self

coerce = lambda x: Torchish(x).value

def implements(torch_function, JAXishify_output=True):
  """Register a torch function override for Torchish"""
  def decorator(func):
    func1 = (lambda *args, **kwargs: Torchish(func(*args, **kwargs))) if JAXishify_output else func
    functools.update_wrapper(func1, torch_function)
    HANDLED_FUNCTIONS[torch_function] = func1
    return func1
  return decorator

def connect(torch_function, jax_function, dont_coerce_argnums=()):
  @implements(torch_function)
  def fn(*args, **kwargs):
    # NOTE: we don't coerce kwargs! So far this has not been problematic.
    return jax_function(*(arg if i in dont_coerce_argnums else coerce(arg) for i, arg in enumerate(args)), **kwargs)

connect(torch.add, jnp.add)
connect(torch.exp, jnp.exp)
connect(torch.nn.functional.gelu, jax.nn.gelu)
connect(torch.mean, jnp.mean)
connect(torch.mul, jnp.multiply)
connect(torch.permute, jnp.transpose, dont_coerce_argnums=(1, 2))
connect(torch.pow, jnp.power)
connect(torch.sigmoid, jax.nn.sigmoid)
connect(torch.sqrt, jnp.sqrt)
connect(torch.sum, jnp.sum)
connect(torch.tanh, jnp.tanh)
# TODO: test this... it should need dont_coerce_argnums
connect(torch.transpose, jnp.swapaxes)

connect(torch.Tensor.mul, jnp.multiply)

# TODO: test
@implements(torch.cat)
def cat(tensors, dim=0): return jnp.concatenate([coerce(x) for x in tensors], axis=dim)

# TODO: test flatten
@implements(torch.flatten)
def flatten(input, start_dim=0, end_dim=-1):
  assert end_dim == -1, "TODO: implement end_dim"
  return jnp.reshape(coerce(input), input.shape[:start_dim] + (-1,))

@implements(torch.nn.functional.adaptive_avg_pool2d)
def adaptive_avg_pool2d(input, output_size):
  assert output_size == 1 or output_size == (1, 1), "TODO: implement output_size != 1"
  assert input.ndim == 4, "TODO: implement non-batched input"
  return jnp.mean(coerce(input), axis=(2, 3), keepdims=True)

@implements(torch.nn.functional.batch_norm)
def batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
  assert not training, "torch.nn.functional.batch_norm is only supported in eval()-mode"
  newshape = (1, -1) + (1,) * (len(input.shape) - 2)
  res = (coerce(input) - coerce(running_mean).reshape(newshape)) * jax.lax.rsqrt(coerce(running_var).reshape(newshape) + coerce(eps))
  if weight is not None:
    res *= coerce(weight).reshape(newshape)
  if bias is not None:
    res += coerce(bias).reshape(newshape)
  return res

@implements(torch.nn.functional.conv2d)
def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
  assert groups == 1, "conv2d with groups != 1 is not yet supported"

  # jax.lax.conv_general_dilated supports different lo/hi padding, whereas PyTorch applies the same padding on both sides.
  if isinstance(padding, tuple):
    p1, p2 = padding
    padding = [(p1, p1), (p2, p2)]

  res = jax.lax.conv_general_dilated(
      lhs=coerce(input),
      rhs=coerce(weight),
      window_strides=stride,
      padding=padding,
      rhs_dilation=dilation
  )
  if bias is not None:
    res += coerce(bias)[jnp.newaxis, :, jnp.newaxis, jnp.newaxis]
  return res

@implements(torch.nn.functional.dropout)
def dropout(input, p=0.5, training=True, inplace=False):
  assert not training, "TODO: implement dropout=True"
  assert not inplace, "TODO: implement inplace=True"
  return input

@implements(torch.nn.functional.layer_norm)
def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05):
  input = coerce(input)

  d = len(normalized_shape)
  mean = jnp.mean(input, axis=tuple(range(input.ndim)[-d:]), keepdims=True)
  # NOTE: According to https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html, PyTorch calculates variance
  # without Bessel's correction. This is the default behavior in numpy, but we set `ddof=0` to be explicit.
  var = jnp.var(input, axis=tuple(range(input.ndim)[-d:]), keepdims=True, ddof=0)

  res = (input - mean) / jnp.sqrt(var + eps)
  if weight is not None:
    res *= coerce(weight)
  if bias is not None:
    res += coerce(bias)
  return res

@implements(torch.nn.functional.linear)
def linear(input, weight, bias=None):
  if bias is None:
    return coerce(input) @ coerce(weight).T
  else:
    return coerce(input) @ coerce(weight).T + coerce(bias)

@implements(torch.nn.functional.max_pool1d)
def max_pool1d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
  assert dilation == 1, "TODO: implement dilation != 1"
  assert not ceil_mode, "TODO: implement ceil_mode"
  assert not return_indices, "TODO: implement return_indices"

  return jax.lax.reduce_window(
      coerce(input),
      -jnp.inf,
      jax.lax.max,
      window_dimensions=(1, 1, kernel_size) if isinstance(kernel_size, int) else (1, 1) + kernel_size,
      window_strides=(1, 1, stride) if isinstance(stride, int) else (1, 1) + stride,
      padding=[(0, 0), (0, 0), (padding, padding)] if isinstance(padding, int) else [(0, 0), (0, 0), padding * 2])

@implements(torch.nn.functional.max_pool2d)
def max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
  assert dilation == 1, "TODO: implement dilation != 1"
  assert not ceil_mode, "TODO: implement ceil_mode"
  assert not return_indices, "TODO: implement return_indices"

  return jax.lax.reduce_window(
      coerce(input),
      -jnp.inf,
      jax.lax.max,
      window_dimensions=(1, 1, kernel_size, kernel_size) if isinstance(kernel_size, int) else (1, 1) + kernel_size,
      window_strides=(1, 1, stride, stride) if isinstance(stride, int) else (1, 1) + stride,
      padding=[(0, 0), (0, 0), (padding, padding), (padding, padding)] if isinstance(padding, int) else [(0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])])

@implements(torch.nn.functional.relu)
def relu(x, inplace=False):
  # Can't use `connect` since jax.nn.relu does not have an `inplace` option.
  if inplace:
    assert isinstance(x, Torchish)
    x.value = jax.nn.relu(x.value)
    return x
  else:
    return jax.nn.relu(coerce(x))

@implements(torch.nn.functional.scaled_dot_product_attention)
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False):
  assert attn_mask is None, "TODO: implement attn_mask"
  assert dropout_p == 0.0, "TODO: implement dropout"
  assert not is_causal, "TODO: implement is_causal"

  # query is (N, ..., L, E)
  # key, value are (N, ..., S, E)

  Q, K, V = coerce(query), coerce(key), coerce(value)
  # L = Q.shape[-2]
  # S = K.shape[-2]

  # From https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
  # attn_mask = jnp.tril(jnp.ones(L, S, dtype=jnp.bool)) if is_causal else attn_mask
  # See https://github.com/pytorch/pytorch/issues/110341.
  # attn_mask = attn_mask.masked_fill(not attn_mask, -float('inf')) if attn_mask.dtype == jnp.bool else attn_mask

  attn_weight = jax.nn.softmax((Q @ jnp.swapaxes(K, -2, -1) / math.sqrt(Q.shape[-1])), axis=-1)
  # attn_weight = torch.dropout(attn_weight, dropout_p)
  return attn_weight @ V

# NOTE: the "torch.Tensor" type annotations here are a lie, or at least an approximation: In reality, they can be
# anything coerce-able.
@implements(torch.nn.functional.multi_head_attention_forward, JAXishify_output=False)
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
    is_causal: bool = False
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

  Q, K, V = coerce(query), coerce(key), coerce(value)
  assert Q.ndim == 3 and K.ndim == 3 and V.ndim == 3, "TODO: implement non-batched version"
  # For some asinine reason, the PyTorch calling signature is query (L, N, E), key (S, N, E), and value (S, N, E).
  Q, K, V = jnp.swapaxes(Q, 0, 1), jnp.swapaxes(K, 0, 1), jnp.swapaxes(V, 0, 1)

  in_proj_weight, in_proj_bias = coerce(in_proj_weight), coerce(in_proj_bias)
  out_proj_weight, out_proj_bias = coerce(out_proj_weight), coerce(out_proj_bias)

  w_q, w_k, w_v = jnp.split(in_proj_weight, 3)
  b_q, b_k, b_v = jnp.split(in_proj_bias, 3)
  # print(w_q.shape, w_k.shape, w_v.shape)  # (E, E) (E, E) (E, E)
  Q1, K1, V1 = Q @ w_q.T + b_q, K @ w_k.T + b_k, V @ w_v.T + b_v

  # print(Q1.shape, K1.shape, V1.shape)  # (N, L, E) (N, S, E) (N, S, E)
  sdpa = jnp.concatenate(
    tuple(
      scaled_dot_product_attention(q, k, v).value
      for q, k, v in zip(jnp.split(Q1, num_heads, axis=-1),
                         jnp.split(K1, num_heads, axis=-1),
                         jnp.split(V1, num_heads, axis=-1))),
    axis=-1)
  # print(sdpa.shape)  # (N, L, E)
  out = sdpa @ out_proj_weight.T + out_proj_bias
  return Torchish(jnp.swapaxes(out, 0, 1)), None

# It might be nice to use torch.func.functionalize, but it so far seems buggy (FunctionalTensor.numpy() gives incorrect
# results) and unnecessary.
t2j_function = lambda f: lambda *args: f(*jax.tree_util.tree_map(Torchish, args)).value

def t2j_module(module):
  def f(x, state_dict={}):
    # We want to have a non-mutating API, so we need to copy the module before performing parameter surgery. Note that
    # doing this copy in `t2j_module` and outside of `f` is not sufficient: multiple calls to `f` should not step on
    # each others toes.
    m = copy.deepcopy(module)

    # Can't use torch.func.functional_call due to https://github.com/pytorch/pytorch/issues/110249
    assert state_dict.keys() == dict(m.state_dict()).keys()

    def visit(m, prefix):
      for name, _ in m.named_parameters(recurse=False):
        m._parameters[name] = Torchish(state_dict[".".join(prefix + [name])])

      for name, _ in m.named_buffers(recurse=False):
        m._buffers[name] = Torchish(state_dict[".".join(prefix + [name])])

      # NOTE: named_children() is the non-recursive version of named_modules()
      for name, child in m.named_children():
        visit(child, prefix=prefix + [name])

    # Replace parameters with Torchish objects
    visit(m, prefix=[])

    return t2j_function(m)(x)

  return f

def t2j(thing):
  if isinstance(thing, torch.Tensor):
    return t2j_array(thing)
  elif isinstance(thing, torch.nn.Module):
    return t2j_module(thing)
  elif callable(thing):
    # This branch must live below the torch.nn.Module branch!
    return t2j_function(thing)
  else:
    raise NotImplementedError

def j2t(thing):
  if isinstance(thing, jnp.ndarray):
    return j2t_array(thing)
  else:
    raise NotImplementedError
