# GraphKV

[![CI](https://github.com/heterodoxin/graphkv/actions/workflows/ci.yml/badge.svg)](https://github.com/heterodoxin/graphkv/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Model bundled](https://img.shields.io/badge/bundled%20model-mneme--graph--17l-purple.svg)](src/graphkv/models/mneme-graph-17l)

GraphKV is a custom Graph Mneme-guided KV cache system for low-bit transformer
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
| Qwen2.5-7B NF4, 1k-token cache, next-token decode | `graphkv-qwen7-nf4` | `43,352,064 / 58,720,256` | `1.35x` | cosine `0.827394`, top10 `0.80`, argmax match |
| Qwen2.5-7B NF4, 4k-token cache, next-token decode | `graphkv-qwen7-nf4` | `95,993,856 / 234,881,024` | `2.45x` | cosine `0.830570`, top10 `0.70`, argmax match |
| Qwen2.5-7B NF4, 16k-token cache, next-token decode | `graphkv-qwen7-nf4` | `292,454,400 / 939,524,096` | `3.21x` | cosine `0.998599`, top10 `1.00`, argmax match |
| Qwen2.5-7B NF4, 16k-token Mneme semantic-tail retest | `graphkv` | `255,437,952 / 939,524,096` | `3.68x` | cosine `0.999018`, top10 `0.90`, argmax match |
| Qwen2.5-7B NF4, 32k-token cache, next-token decode | `graphkv-qwen7-nf4` | `558,530,560 / 1,879,048,192` | `3.36x` | cosine `0.990316`, top10 `1.00`, argmax match |

The Qwen2.5-7B runs use real chunked prefill, then compare a one-token decode
from the original KV cache against the GraphKV-compressed cache exported back to
Hugging Face `DynamicCache`. `graphkv-qwen7-nf4` keeps a 128-token prompt sink
and 512-token recent tail, uses affine int4 below 8k tokens, and switches to
symmetric int4 for longer caches.

## Bundled Graph Mneme Model

GraphKV includes the custom Graph Mneme 17L model artifact in the repo:

```text
src/graphkv/models/mneme-graph-17l/latest.json
src/graphkv/models/mneme-graph-17l/latest.safetensors
```

The `.safetensors` file is stored directly in the repository. Users do not need
the old local `rustembed` folder to load the graph model:

```python
from graphkv import (
    GraphMemoryRequest,
    GraphRetentionChunk,
    bundled_graph_model_metadata,
    quantize_hf_cache,
)

print(bundled_graph_model_metadata()["eval_metrics"]["wikikg2"])

graph_memory = GraphMemoryRequest(
    query_anchor=0,
    query_relation=7,
    query_facts=[(0, 7, 1), (1, 9, 2)],
    chunks=[
        GraphRetentionChunk(anchor=1, span=(0, 256), facts=[(1, 7, 3), (3, 9, 4)]),
        GraphRetentionChunk(anchor=2, span=(256, 512), facts=[(2, 4, 5), (5, 6, 6)]),
    ],
)

compressed_cache = quantize_hf_cache(
    past_key_values,
    graph_memory=graph_memory,
    output="compressed",
)
print(compressed_cache.stats())
```

The bundled model card records `0.54` MRR on WikiKG2 with
`num_global_relations=0`. GraphKV uses that graph-ranking signal both to
protect semantically important token spans and to route lowest-priority spans
into cheaper low-bit storage.

## GraphKV Retest

After adding the built-in Mneme retention API, GraphKV was retested locally on
`Qwen/Qwen2.5-7B` NF4 with a 16k-token code-shaped prompt. The target code span
was placed in the middle of the cache, outside the prompt sink and recent tail.
The default `graphkv` profile keeps a 128-token sink and a 256-token tail when
no graph signal is supplied. When Mneme or explicit importance scores are
available at 8k+ context lengths, it trusts that graph signal more: the recent
tail drops to 128 tokens, 1% of Mneme-ranked candidate tokens stay full
precision, and the lowest-priority 25% of compressed tokens move to 2-bit value
storage.

| Setup | Cache length | Cache bytes | Compression | Fidelity |
| --- | ---: | ---: | ---: | --- |
| GraphKV plain `graphkv-qwen7-nf4` | 16k | `292,454,400 / 939,524,096` | `3.21x` | cosine `0.979474`, top10 `0.80`, argmax match |
| Previous GraphKV Mneme tier, 256-token tail | 16k | `260,356,096 / 939,524,096` | `3.61x` | cosine `0.997761`, top10 `0.90`, argmax match |
| GraphKV default with Mneme semantic tail | 16k | `255,437,952 / 939,524,096` | `3.68x` | cosine `0.999018`, top10 `0.90`, argmax match |
| GraphKV 30% cold tier sweep, not default | 16k | `252,090,496 / 939,524,096` | `3.73x` | cosine `0.997708`, top10 `0.90`, argmax match |

The same sweep rejected an aggressive 6% plus 128-token minimum graph-retention
policy at 16k because it over-retained, dropped cosine to `0.906118`, and
changed argmax. The default `graphkv` profile uses the calibrated 1%
long-context graph policy when graph memory or importance scores are supplied.
It also uses Mneme to identify the least relevant compressed tokens for the
2-bit value tier. A 30% cold-tier sweep reached `3.73x`, but the default stays
at 25% for the stronger fidelity margin; pushing that tier to 35% or higher
reduced top10 quality.

## Qwen 7B NF4 Comparison

Local GraphKV and TurboQuant runs used `Qwen/Qwen2.5-7B` loaded with
bitsandbytes NF4 weights on the RTX 4070 Ti SUPER. Each row uses the same
code-shaped prompt length and compares one-token decode logits against an
uncompressed baseline. TurboQuant rows use the local source
`CompressedDynamicCache`; its byte counts are the wrapper-reported compressed
KV storage, matching the cache-storage accounting used for GraphKV.

| System | Cache length | Cache bytes | Compression | Fidelity |
| --- | ---: | ---: | ---: | --- |
| GraphKV `graphkv-qwen7-nf4` | 1k | `43,352,064 / 58,720,256` | `1.35x` | cosine `0.827394`, top10 `0.80`, argmax match |
| TurboQuant K4/V4 | 1k | `15,597,568 / 58,720,256` | `3.76x` | cosine `-0.393062`, top10 `0.50`, argmax match |
| TurboQuant K4/V3 | 1k | `22,937,600 / 58,720,256` | `2.56x` | cosine `-0.124455`, top10 `0.50`, argmax changed |
| GraphKV `graphkv-qwen7-nf4` | 4k | `95,993,856 / 234,881,024` | `2.45x` | cosine `0.830570`, top10 `0.70`, argmax match |
| TurboQuant K4/V4 | 4k | `62,390,272 / 234,881,024` | `3.76x` | cosine `-0.209976`, top10 `0.20`, argmax changed |
| TurboQuant K4/V3 | 4k | `91,750,400 / 234,881,024` | `2.56x` | cosine `-0.192223`, top10 `0.20`, argmax changed |
| GraphKV `graphkv` | 16k | `255,437,952 / 939,524,096` | `3.68x` | cosine `0.999018`, top10 `0.90`, argmax match |
| GraphKV `graphkv-qwen7-nf4` | 16k | `292,454,400 / 939,524,096` | `3.21x` | cosine `0.998599`, top10 `1.00`, argmax match |
| TurboQuant K4/V4 | 16k | `249,561,088 / 939,524,096` | `3.76x` | cosine `0.657976`, top10 `0.10`, argmax changed |
| TurboQuant K4/V3 | 16k | `367,001,600 / 939,524,096` | `2.56x` | cosine `0.649364`, top10 `0.10`, argmax changed |
| GraphKV `graphkv-qwen7-nf4` | 32k | `558,530,560 / 1,879,048,192` | `3.36x` | cosine `0.990316`, top10 `1.00`, argmax match |
| TurboQuant K4/V4 | 32k | `499,122,176 / 1,879,048,192` | `3.76x` | cosine `-0.401509`, top10 `0.10`, argmax changed |
| TurboQuant K4/V3 | 32k | `734,003,200 / 1,879,048,192` | `2.56x` | cosine `-0.472464`, top10 `0.10`, argmax changed |

External context: [vLLM TurboQuant study](https://vllm.ai/blog/2026-05-11-turboquant).

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

- `graphkv`: default profile with Mneme built in, tuned for long-context
  fidelity with a 128-token sink, 256-token normal tail, 128-token Mneme
  semantic tail, 64-token quant groups, and calibrated graph retention above 8k
  tokens when graph memory is supplied.
  Mneme also routes the lowest-priority 25% of compressed long-context tokens
  into 2-bit value storage for extra compression.
- `graphkv-int2-max`: aggressive 2-bit affine packing.
- `graphkv-int4-balanced`: 4-bit symmetric, tuned for grouped-query models.
- `graphkv-int4-safe`: 4-bit with sink/tail retention and outlier retention.
- `graphkv-qwen7-nf4`: conservative Qwen2.5-7B NF4 profile used for local
  TurboQuant comparison rows.

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
native recipe helpers, bundled graph model weights, real-model README
statistics, badges, and GitHub Actions CI. Plain `.txt` files are ignored by
default and were not included in the repository.

## Why These Axes?

GraphKV uses built-in Mneme retention plus asymmetric KV grouping: keys are
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
