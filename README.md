# GraphKV

[![CI](https://github.com/heterodoxin/graphkv/actions/workflows/ci.yml/badge.svg)](https://github.com/heterodoxin/graphkv/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Model bundled](https://img.shields.io/badge/bundled%20model-mneme--graph--17l-purple.svg)](src/graphkv/models/mneme-graph-17l)

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
| Qwen2.5-7B actual next-token forward, NF4 weights, 128-token cache | `graphkv-qwen7-nf4` | `2,618,112 / 7,340,032` | `2.80x` | cosine `0.99386`, top10 `0.90` |
| Codebase-shaped synthetic KV, 16k tokens, 16 layers | `graphkv-int4-balanced` | `286,261,248 / 1,073,741,824` | `3.75x` | attention cosine `0.98845` |
| Synthetic pressure test, 8k tokens, 16 layers | `graphkv-int2-max` | `73,924,608 / 536,870,912` | `7.26x` | attention cosine `0.78807` |

Read the numbers as profile guidance: `graphkv-int4-balanced` is the current
default for fidelity, while `graphkv-int2-max` is a memory pressure profile that
needs stronger graph retention or model-specific tuning for harder workloads.

## Bundled Graph Model

GraphKV includes the custom graph-memory model artifact in the repo:

```text
src/graphkv/models/mneme-graph-17l/latest.json
src/graphkv/models/mneme-graph-17l/latest.safetensors
```

The `.safetensors` file is stored directly in the repository. Users do not need
the old local `rustembed` folder to load the graph model:

```python
from graphkv import build_candidate_bank, build_query_bank, load_bundled_graph_model

model = load_bundled_graph_model(device="auto")
cfg = model.cfg
query = build_query_bank(0, 7, [(0, 7, 1), (1, 9, 2)], cfg)
candidates = [
    build_candidate_bank(1, [(1, 7, 3), (3, 9, 4)], cfg),
    build_candidate_bank(2, [(2, 4, 5), (5, 6, 6)], cfg),
]
scores = model.score(query, candidates)
```

## Qwen 7B NF4 Comparison

Local GraphKV was run on `Qwen/Qwen2.5-7B` loaded with bitsandbytes NF4 weights
on the RTX 4070 Ti SUPER. TurboQuant and KVarN are listed as public reported
baselines because their runnable backends were not installed in this environment.

| System | Benchmark status | Model/setup | Result |
| --- | --- | --- | --- |
| GraphKV | Local run | Qwen2.5-7B NF4, 128-token cache, next-token logits | `2.80x` KV compression, cosine `0.99386`, top10 `0.90` |
| TurboQuant | Reported by vLLM study | Qwen3-30B-A3B on 2xH100, not local Qwen7 NF4 | capacity `2.3-3.7x`, with `40-52%` throughput reduction for TurboQuant variants in the vLLM study |
| TurboQuant calculator | Reported formula estimate | Qwen2.5-7B KV shape at 128k context | FP16 `7.00 GB`, TQ 3-bit `1.31 GB` |
| KVarN | Reported by upstream repo | vLLM fork, Qwen3-32B/agentic workloads, not local Qwen7 NF4 | `3-5x` more KV capacity, up to `~1.3x` FP16 throughput, FP16-level accuracy |

Sources: [vLLM TurboQuant study](https://vllm.ai/blog/2026-05-11-turboquant),
[TurboQuant Qwen2.5-7B calculator](https://turbo-quant.com/kv-cache-calculator),
and [KVarN upstream](https://github.com/huawei-csl/KVarN).

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
- `graphkv-qwen7-nf4`: Qwen2.5-7B NF4 local profile, tuned for fidelity.

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
native recipe helpers, bundled graph model weights, README statistics, badges,
and GitHub Actions CI. Plain `.txt` files are ignored by default and were not
included in the repository.

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
