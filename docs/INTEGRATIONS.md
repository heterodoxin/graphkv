# GraphKV Integrations

## Transformers

GraphKV can read and write modern Hugging Face cache objects:

- legacy tuple/list caches
- `DynamicCache`
- `StaticCache`
- `SlidingWindowCache`
- `EncoderDecoderCache` self and cross roles

The export path dequantizes into a normal Hugging Face cache. This is deliberate:
it makes correctness tests easy and lets users adopt GraphKV without modifying
model internals. For runtime speedups, a fused low-bit attention path is needed.

## vLLM

vLLM has native FP8 KV cache support through `kv_cache_dtype`. Its docs describe
default scales, warmup scale calculation, and dataset calibration. GraphKV exposes
small helpers:

```python
from graphkv import vllm_kwargs

kwargs = vllm_kwargs("vllm-fp8-calibrated")
```

GraphKV int2/int4 is not injected into vLLM today. A real vLLM implementation
would need a cache backend plus attention kernels that consume GraphKV's packed
representation instead of dequantizing first.

## llama.cpp

llama.cpp exposes native KV cache type flags:

```bash
--cache-type-k q8_0 --cache-type-v q8_0
```

GraphKV exposes `llama_cpp_args`, `llama_cpp_env`, and `llama_cpp_command` so
applications can wire this into launchers without hard-coding flags. Custom
GraphKV int2/int4 in llama.cpp would require a C/C++ implementation in ggml and
attention kernels that understand the packed layout.

## Engine Support Matrix

| Engine | Works now | What it does |
| --- | --- | --- |
| Transformers | Yes | GraphKV compress/decompress plus HF cache adapters |
| vLLM | Recipe | Native FP8 KV flags |
| llama.cpp | Recipe | Native `--cache-type-k/v` flags |
| Custom fused kernels | Planned | Consume GraphKV packed tensors directly |
