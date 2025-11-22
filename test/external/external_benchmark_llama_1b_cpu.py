from pathlib import Path
from typing import List
import json, argparse, random, time, os
from extra.models.llama import Transformer, convert_from_huggingface, convert_from_gguf, fix_bf16
import torch
from tinygrad.nn.state import safe_load, torch_load, load_state_dict, get_parameters, gguf_load
from tinygrad import Tensor, dtypes, nn, Context, Device, GlobalCounters, TinyJit
from tinygrad.helpers import Profiling, Timing, DEBUG, colored, fetch, tqdm
from extra.bench_log import BenchEvent, WallTimeEvent
from test.helpers import derandomize_model

# **** helper functions ****
def concat_weights(models, device=None):
  def convert(name) -> Tensor:
    disk_tensors: List[Tensor] = [model[name] for model in models]
    if len(disk_tensors) == 1 or len(disk_tensors[0].shape) == 1:
      return disk_tensors[0].to(device=device)
    axis = 1 if name.endswith((".attention.wo.weight", ".feed_forward.w2.weight")) else 0
    lazy_tensors = [data.to(device=device) for data in disk_tensors]
    return lazy_tensors[0].cat(*lazy_tensors[1:], dim=axis)
  return {name: convert(name) for name in {name: None for model in models for name in model}}

# **** quantized linears ****
class Int8Linear:
  def __init__(self, in_features, out_features, bias=False):
    assert bias == False
    self.weight = Tensor.ones(out_features, in_features, dtype=dtypes.int8)
    self.scale = Tensor.ones(out_features, dtype=dtypes.half)

  def __call__(self, x):
    return x.dot(self.weight.cast(self.scale.dtype).T*self.scale)

  @staticmethod
  def quantize(tensors, device, scale_dtype=dtypes.float16, quantize_embeds=False):
    new_tensors = {}
    for name,v in tensors.items():
      if "feed_forward" in name or "attention.w" in name or (quantize_embeds and "tok_embeddings.weight" in name):
        assert "weight" in name, name
        v = v.cast(scale_dtype)
        scale = v.abs().max(axis=1) / 127.0
        int8_weight = (v.T/scale).T.round().cast(dtype=dtypes.int8) # without round(), cast truncates -34.9 to -34
        new_tensors[name] = int8_weight
        new_tensors[name.replace('weight', 'scale')] = scale
        if isinstance(device, tuple):
          new_tensors[name].shard_(device, axis=-1)
          new_tensors[name.replace('weight', 'scale')].shard_(device, axis=None)
      else:
        new_tensors[name] = v
    if quantize_embeds: new_tensors.update({"output.weight": new_tensors["tok_embeddings.weight"], "output.scale": new_tensors["tok_embeddings.scale"]})
    return new_tensors

class Int8Embedding:
  def __init__(self, vocab_size:int, embed_size:int):
    self.vocab_sz, self.embed_sz = vocab_size, embed_size
    self.weight, self.scale = Tensor.ones(vocab_size, embed_size, dtype=dtypes.int8), Tensor.ones(vocab_size, dtype=dtypes.half)

  def __call__(self, idx:Tensor) -> Tensor:
    if not hasattr(self, 'arange'): self.arange = Tensor.arange(self.vocab_sz, requires_grad=False, device=self.weight.device).unsqueeze(-1)
    big_shp = idx.shape+(self.vocab_sz, self.embed_sz)
    arange, idx, vals = self.arange.expand(big_shp), idx.reshape(idx.shape+(1, 1)).expand(big_shp), (self.weight.cast(self.scale.dtype).T*self.scale).T
    return (arange == idx).mul(vals).sum(-2, dtype=vals.dtype)

def NF4Linear(block_size):
  _CODE = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
  ]
  CODE = Tensor.stack(*[Tensor(c, dtype=dtypes.float16) for c in _CODE])
  class _NF4Linear:
    def __init__(self, in_features, out_features, bias=False):
      assert not bias, "bias not supported"
      self.in_features, self.out_features = in_features, out_features
      self.weight = Tensor.empty(int(out_features * in_features / 2), dtype=dtypes.uint8)
      self.scale = Tensor.empty(int(out_features * in_features / block_size), 1, dtype=dtypes.float16)

    def __call__(self, x: Tensor) -> Tensor:
      high_bits = self.weight
      low_bits = (self.weight * 2 ** 4).contiguous()
      unpacked = Tensor.stack(high_bits, low_bits, dim=-1).idiv(2 ** 4)
      unscaled = CODE[unpacked].to(x.device).reshape(-1, block_size) * self.scale
      return x.linear(unscaled.reshape(self.out_features, self.in_features).T)

    @staticmethod
    def quantize(state_dict: dict[str, Tensor], device, scale_dtype=dtypes.float16, quantize_embeds=False) -> dict[str, Tensor]:
      assert not quantize_embeds  # TODO: support this?
      new_state_dict = {}
      for k, v in state_dict.items():
        if "feed_forward" in k or "attention.w" in k:
          grouped = v.reshape(-1, block_size)
          scale = (grouped.abs().max(axis=1, keepdim=True))
          coded = ((grouped / scale).unsqueeze(-1) - CODE.to(v.device)).abs().argmin(axis=-1).cast(dtypes.uint8).flatten()
          new_state_dict[k] = coded[::2] * 2 ** 4 + coded[1::2]
          new_state_dict[k.replace(".weight", ".scale")] = scale.cast(scale_dtype)
          if isinstance(device, tuple):
            new_state_dict[k].shard_(device, axis=-1)
            new_state_dict[k.replace('weight', 'scale')].shard_(device, axis=None)
        else:
          new_state_dict[k] = v
      return new_state_dict
  return _NF4Linear

def quantize_to_fp8(x: Tensor, dtype=dtypes.fp8e4m3):
  fp8_min = -448.0 if dtype == dtypes.fp8e4m3 else -57344.0
  fp8_max = 448.0 if dtype == dtypes.fp8e4m3 else 57344.0
  scale = fp8_max / x.abs().max()
  x_scl_sat = (x * scale).clamp(fp8_min, fp8_max)
  return x_scl_sat.cast(dtype), scale.float().reciprocal()

class FP8Linear:
  def __init__(self, in_features, out_features, bias=True):
    self.weight = Tensor.empty(out_features, in_features, dtype=dtypes.fp8e4m3)
    self.bias = Tensor.empty(out_features, dtype=dtypes.float16) if bias else None
    self.weight_scale = Tensor.empty((), dtype=dtypes.float16)

  def __call__(self, x:Tensor):
    y = x.dot(self.weight.T.cast(dtypes.float32)) * self.weight_scale
    if self.bias is not None: y = y + self.bias.cast(y.dtype)
    return y.cast(x.dtype)

  @staticmethod
  def quantize(tensors, device, scale_dtype=dtypes.float16, quantize_embeds=False):
    assert not quantize_embeds
    new_tensors = {}
    for name,v in tensors.items():
      if "feed_forward" in name or "attention.w" in name:
        assert "weight" in name, name
        fp8_weight, scale = quantize_to_fp8(v)
        new_tensors[name] = fp8_weight
        new_tensors[name.replace('weight', 'weight_scale')] = scale.cast(scale_dtype)
        if isinstance(device, tuple):
          new_tensors[name].shard_(device, axis=-1)
          new_tensors[name.replace('weight', 'weight_scale')].shard_(device, axis=None)
      else:
        new_tensors[name] = v
    return new_tensors

MODEL_PARAMS = {
  "1B": {
    "args": {"dim": 2048, "n_heads": 32, "n_kv_heads": 8, "n_layers": 16, "norm_eps": 1e-5, "rope_theta": 500000, "vocab_size": 128256, "hidden_dim": 8192},
  },
}
def build_transformer(model_size="1B", quantize=None, scale_dtype=dtypes.float16, device=None, max_context=8192, load_weights=True):
  # build model
  linear, embedding, quantize_embeds = nn.Linear, nn.Embedding, False
  model = Transformer(**MODEL_PARAMS[model_size]["args"], linear=linear, embedding=embedding, max_context=max_context, jit=True)

  if not load_weights: return model

  print(f"Model parameters: {sum(p.numel() for p in get_parameters(model)):,}")

  # load weights
  with WallTimeEvent(BenchEvent.LOAD_WEIGHTS):
    # This is just for fixing bf16 weights to float16
    # weights = fix_bf16(weights)
    derandomize_model(model)
  return model

# default settings
TEMPERATURE = 0.95
TOP_K = 0
TOP_P = 0.0
ALPHA_F = 0.0
ALPHA_P = 0.0

last_seen_toks = []
def prefill(model, toks, start_pos=0):
  global last_seen_toks

  # we can skip part of the prompt if it is the same as last and start_pos=0
  if start_pos == 0:
    for i, (a, b) in enumerate(zip(toks, last_seen_toks)):
      if a != b: break
    else: i = min(len(toks), len(last_seen_toks))
    start_pos += i
    last_seen_toks = toks
    toks = toks[i:]

  # prefill the model
  for tok in tqdm(toks):
    GlobalCounters.reset()
    model(Tensor([[tok]], device=Device.DEFAULT), start_pos, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P).realize()
    start_pos += 1
  return start_pos

global_mem_used = 0
def helper_test(nm, gen, model, max_memory_allowed, max_kernels_allowed, all_jitted=False):
  with Context(JIT=2):
    tms = []
    for _ in range(4):
      early_gen = [x.realize() if isinstance(x, Tensor) else x for x in gen()]
      GlobalCounters.reset()
      Device[Device.DEFAULT].synchronize()
      st = time.perf_counter_ns()
      model(*early_gen)
      Device[Device.DEFAULT].synchronize()
      tms.append(time.perf_counter_ns() - st)
    mem_used = GlobalCounters.mem_used - global_mem_used

    # TODO: jit should expose this correctly with graph
    kernels_used = len(model.jit_cache) if hasattr(model, "jit_cache") else None
    print(f"{nm}: used {mem_used/1e9:.2f} GB and {kernels_used} kernels in {min(tms)/1e6:.2f} ms")
    # assert mem_used/1e9 < max_memory_allowed, f"{nm} used more than {max_memory_allowed:.2f} GB - {mem_used/1e9:.2} GB used"
    # assert not kernels_used or kernels_used <= max_kernels_allowed, f"{nm} used more than {max_kernels_allowed} kernels, it used {kernels_used}"
    # if all_jitted:
    #   assert kernels_used > 0 and kernels_used == GlobalCounters.kernel_count or (kernels_used <= GlobalCounters.kernel_count and getattr(Device[Device.DEFAULT], "graph", None)), f"only {kernels_used} out of {GlobalCounters.kernel_count} were jitted"  # noqa: E501

if __name__ == "__main__":
  Device.DEFAULT = "CPU"
  model = build_transformer(model_size='1B', device=Device.DEFAULT)
  param_bytes = sum(x.uop.size * x.dtype.itemsize for x in get_parameters(model))
  
  @TinyJit
  def test(t): return model(t, 0).realize()
  # TODO: test first token vs rest properly
  helper_test("test_llama", lambda: (Tensor([[1,2,3,4]]),), test, 0.27, 168, all_jitted=True)
  
  from transformers import LlamaConfig, LlamaModel
  torch_config = LlamaConfig(
    torch_dtype=torch.float16,
    **MODEL_PARAMS['1B']['args']
  )
  torch_model = LlamaModel(torch_config)
  print(torch_model)
  torch_model.eval()
  with torch.no_grad():
    torch_model(torch.tensor([[1,2,3,4]]))
  
  