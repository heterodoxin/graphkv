# GraphKV

GraphKV is a custom graph-memory KV cache system for low-bit transformer
inference. It keeps recent, salient, or graph-relevant tokens high precision,
then compresses the rest aggressively with packed int2/int4 cache tensors.

Current status: alpha. The Python implementation is correctness-first and works
with Hugging Face/Transformers cache objects. vLLM and llama.cpp support is
provided as native engine recipes today, because custom GraphKV int2/int4
execution inside those engines requires their attention/cache kernel paths.

## Local Results

Measured on an NVIDIA GeForce RTX 4070 Ti SUPER with PyTorch CUDA.

| Test | Profile | Cache bytes | Compression | Quality |
| --- | --- | ---: | ---: | --- |
| Tiny GPT-2 actual next-token forward | `graphkv-int2-max` | `15,840 / 122,880` | `7.76x` | cosine `0.999949`, top10 `1.00` |
| Qwen2.5-0.5B actual next-token forward | `graphkv-int4-balanced` | `110,592 / 393,216` | `3.56x` | cosine `0.993159`, top10 `0.90` |
| Codebase-shaped synthetic KV, 16k tokens, 16 layers | `graphkv-int4-balanced` | `286,261,248 / 1,073,741,824` | `3.75x` | attention cosine `0.98845` |
| Synthetic pressure test, 8k tokens, 16 layers | `graphkv-int2-max` | `73,924,608 / 536,870,912` | `7.26x` | attention cosine `0.78807` |

Read the numbers as profile guidance: `graphkv-int4-balanced` is the current
default for fidelity, while `graphkv-int2-max` is a memory pressure profile that
needs stronger graph retention or model-specific tuning for harder workloads.

## Install

```bash
pip install -e .
```

For Hugging Face/Transformers adapters:

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

## Latest Push

The current public push includes the custom GraphKV compression core, package
metadata under `heterodoxin`, Transformers cache adapters, vLLM and llama.cpp
native recipe helpers, README statistics, and GitHub Actions CI. Plain `.txt`
files are ignored by default and were not included in the repository.

## Why These Axes?

GraphKV uses custom graph-memory retention plus asymmetric KV grouping: keys are
grouped over token blocks so each channel gets its own scale over time, while
values are grouped over channels for per-token scales. The package exposes
retention hooks for semantic or graph-derived token importance scores.

Useful references:

- KIVI: https://arxiv.org/abs/2402.02750
- GEAR: https://arxiv.org/abs/2403.05527
- ZipCache: https://arxiv.org/abs/2405.14256
- vLLM FP8 KV cache docs: https://docs.vllm.ai/en/v0.22.0/features/quantization/quantized_kvcache/
- llama.cpp: https://github.com/ggml-org/llama.cpp

## Publishing Checklist

- Run `python -c "import graphkv; print(graphkv.__version__)"`.
- Run `graphkv-recipes --profile llamacpp-q8 --model model.gguf`.
- Tag a release, then publish with `python -m build` and `twine upload dist/*`.
