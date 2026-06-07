# Benchmark Notes

Hardware used for the initial local smoke benchmark:

- GPU: NVIDIA GeForce RTX 4070 Ti SUPER
- VRAM: 16,376 MiB
- Runtime: PyTorch CUDA

## Codebase-Shaped Synthetic KV Cache

Command:

```bash
graphkv-benchmark \
  --profile graphkv-int4-balanced \
  --scan-dir "C:\Users\Levit\OneDrive\Documents\New project" \
  --cap-tokens 16384 \
  --layers 16 \
  --heads 8 \
  --head-dim 128 \
  --dtype float16 \
  --device cuda
```

Result:

```text
estimated_tokens=16384
shape=layers=16 tokens=16384 heads=8 head_dim=128
memory=286261248/1073741824
ratio=3.7509x
quality=key_cos=0.99427 value_cos=0.99427 attn_cos=0.98845
seconds=2.973
```

## Aggressive Int2 Synthetic KV Cache

Command:

```bash
graphkv-benchmark \
  --profile graphkv-int2-max \
  --tokens 8192 \
  --layers 16 \
  --heads 8 \
  --head-dim 128 \
  --dtype float16 \
  --device cuda
```

Result:

```text
shape=layers=16 tokens=8192 heads=8 head_dim=128
memory=73924608/536870912
ratio=7.2624x
quality=key_cos=0.89219 value_cos=0.89223 attn_cos=0.78807
seconds=2.316
```

Interpretation: `graphkv-int4-balanced` is the publishable default for fidelity.
`graphkv-int2-max` is a pressure-test profile for memory-constrained experiments
and should be paired with stronger retention policies or model-specific tuning.
