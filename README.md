# GraphKV

GraphKV is a small, publishable toolkit for graph-guided low-bit KV cache
compression. It started as a custom cache path for Mneme-style context routing:
keep recent, salient, or graph-relevant tokens high precision, and compress the
rest aggressively.

Current status: alpha. The Python implementation is correctness-first and works
with Hugging Face/Transformers cache objects. vLLM and llama.cpp support is
provided as native engine recipes today, because custom GraphKV int2/int4
execution inside those engines requires their attention/cache kernel paths.

## Install

```bash
pip install -e .
```

For Hugging Face model tests:

```bash
pip install -e ".[transformers]"
```

## Quick Start: Transformers

```python
from graphkv import quantize_hf_cache

# past_key_values from a Hugging Face model forward pass
compressed_cache = quantize_hf_cache(
    past_key_values,
    profile="graphkv-int4-balanced",
    output="dynamic",
    model_config=model.config,
)

next_out = model(next_ids, past_key_values=compressed_cache, use_cache=True)
```

To keep the compressed representation:

```python
from graphkv import QuantizedKvCache, get_profile

profile = get_profile("graphkv-int2-max")
cache = QuantizedKvCache.from_hf_cache(past_key_values, profile.config)
print(cache.compression_ratio())
```

## Profiles

GraphKV tensor profiles:

- `graphkv-int2-max`: aggressive 2-bit affine packing.
- `graphkv-int4-balanced`: 4-bit symmetric, tuned for grouped-query models.
- `graphkv-int4-safe`: 4-bit with residual tail and outlier retention.

Native engine recipes:

- `vllm-fp8`: vLLM native FP8 KV cache.
- `vllm-fp8-calibrated`: vLLM native FP8 with warmup scale calculation.
- `llamacpp-q8`: llama.cpp native Q8_0 KV cache.
- `llamacpp-q4`: llama.cpp native Q4_0 KV cache.

## vLLM Recipe

vLLM currently exposes FP8 KV cache settings such as `kv_cache_dtype="fp8"` and
`calculate_kv_scales=True`. GraphKV gives you helpers for those native settings:

```python
from vllm import LLM
from graphkv import vllm_kwargs

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    **vllm_kwargs("vllm-fp8-calibrated"),
)
```

CLI:

```bash
graphkv-recipes --profile vllm-fp8-calibrated
```

## llama.cpp Recipe

llama.cpp exposes KV cache type flags for K and V:

```bash
graphkv-recipes --profile llamacpp-q8 --model model.gguf
```

That prints a command with:

```bash
--cache-type-k q8_0 --cache-type-v q8_0
```

Use `llamacpp-q4` for a more aggressive native llama.cpp cache.

## Benchmark

Synthetic KV benchmark:

```bash
graphkv-benchmark --profile graphkv-int4-balanced --tokens 8192 --layers 32 --heads 8 --head-dim 128
```

Estimate context size from a codebase, then benchmark a capped synthetic KV cache:

```bash
graphkv-benchmark --scan-dir /path/to/big/repo --cap-tokens 32768
```

## Why These Axes?

GraphKV follows the practical KIVI-style observation that keys and values prefer
different grouping: keys are grouped over token blocks so each channel gets its
own scale over time, while values are grouped over channels for per-token scales.
This repository also exposes retention hooks for semantic or graph-derived token
importance scores.

Useful references:

- KIVI: https://arxiv.org/abs/2402.02750
- GEAR: https://arxiv.org/abs/2403.05527
- ZipCache: https://arxiv.org/abs/2405.14256
- vLLM FP8 KV cache docs: https://docs.vllm.ai/en/v0.22.0/features/quantization/quantized_kvcache/
- llama.cpp: https://github.com/ggml-org/llama.cpp

## Publishing Checklist

- Update `project.urls` in `pyproject.toml`.
- Replace the placeholder author if desired.
- Run `python -m unittest discover -s tests -v`.
- Run at least one benchmark on the target GPU.
- Tag a release, then publish with `python -m build` and `twine upload dist/*`.
