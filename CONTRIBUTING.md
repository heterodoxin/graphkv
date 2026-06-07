# Contributing

GraphKV is alpha. Keep changes measurable and easy to test.

## Local Setup

```bash
pip install -e .
python -c "import graphkv; print(graphkv.__version__)"
graphkv-recipes --profile llamacpp-q8 --model model.gguf
```

## Good First Areas

- More model-family cache shape tests.
- Better retention policies for graph/chunk importance scores.
- vLLM packed-cache backend experiments.
- llama.cpp/ggml packed-cache experiments.
- Long-context measurements that can be summarized in the README.

## Design Rule

Do not claim runtime savings unless the engine consumes packed KV directly.
The current Transformers adapter dequantizes back into an HF cache for
correctness and adoption ease.
