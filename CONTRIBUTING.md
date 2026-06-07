# Contributing

GraphKV is alpha. Keep changes measurable and easy to test.

## Local Setup

```bash
pip install -e ".[dev]"
python -m unittest discover -s tests -v
```

## Good First Areas

- More model-family cache shape tests.
- Better retention policies for graph/chunk importance scores.
- vLLM packed-cache backend experiments.
- llama.cpp/ggml packed-cache experiments.
- Long-context benchmarks with real prompts.

## Design Rule

Do not claim runtime savings unless the engine consumes packed KV directly.
The current Transformers adapter dequantizes back into an HF cache for
correctness and adoption ease.
