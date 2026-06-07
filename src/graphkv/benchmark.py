from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from .core import attention_report, quantize_kv_layer
from .profiles import get_profile


@dataclass
class BenchmarkResult:
    tokens: int
    layers: int
    heads: int
    head_dim: int
    original_bytes: int
    compressed_bytes: int
    ratio: float
    key_cosine: float
    value_cosine: float
    attention_cosine: float
    seconds: float


def estimate_tokens_from_codebase(path: str | os.PathLike[str]) -> int:
    root = Path(path)
    text_suffixes = {
        ".c",
        ".cc",
        ".cpp",
        ".cs",
        ".go",
        ".h",
        ".hpp",
        ".java",
        ".js",
        ".json",
        ".md",
        ".py",
        ".rs",
        ".toml",
        ".ts",
        ".tsx",
        ".yaml",
        ".yml",
    }
    total_bytes = 0
    for file in root.rglob("*"):
        if file.is_file() and file.suffix.lower() in text_suffixes:
            try:
                total_bytes += file.stat().st_size
            except OSError:
                pass
    return max(1, total_bytes // 4)


def run_synthetic_benchmark(
    *,
    profile_name: str = "graphkv-int4-balanced",
    tokens: int = 8192,
    layers: int = 32,
    heads: int = 8,
    head_dim: int = 128,
    dtype: torch.dtype = torch.float16,
    device: str | torch.device = "cpu",
    seed: int = 1234,
) -> BenchmarkResult:
    profile = get_profile(profile_name)
    if profile.config is None:
        raise ValueError(f"Profile {profile.name!r} is not a GraphKV tensor profile.")

    torch.manual_seed(seed)
    device = torch.device(device)
    original_bytes = 0
    compressed_bytes = 0
    key_cosines = []
    value_cosines = []
    attention_cosines = []
    start = time.perf_counter()

    for _ in range(layers):
        keys = torch.randn(1, heads, tokens, head_dim, device=device, dtype=dtype)
        values = torch.randn(1, heads, tokens, head_dim, device=device, dtype=dtype)
        query = torch.randn(1, heads, 1, head_dim, device=device, dtype=dtype)
        layer = quantize_kv_layer(keys, values, profile.config)
        report = layer.report(keys, values)
        attn = attention_report(query, keys, values, layer)
        original_bytes += layer.original_memory_bytes()
        compressed_bytes += layer.memory_bytes()
        key_cosines.append(report["key_cosine"])
        value_cosines.append(report["value_cosine"])
        attention_cosines.append(attn["attention_output_cosine"])

    seconds = time.perf_counter() - start
    return BenchmarkResult(
        tokens=tokens,
        layers=layers,
        heads=heads,
        head_dim=head_dim,
        original_bytes=original_bytes,
        compressed_bytes=compressed_bytes,
        ratio=original_bytes / max(compressed_bytes, 1),
        key_cosine=sum(key_cosines) / len(key_cosines),
        value_cosine=sum(value_cosines) / len(value_cosines),
        attention_cosine=sum(attention_cosines) / len(attention_cosines),
        seconds=seconds,
    )


def _parse_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError("dtype must be float16, bfloat16, or float32")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark GraphKV on synthetic KV caches.")
    parser.add_argument("--profile", default="graphkv-int4-balanced")
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--scan-dir", default=None, help="Estimate tokens from a codebase directory.")
    parser.add_argument("--cap-tokens", type=int, default=32768)
    args = parser.parse_args(argv)

    tokens = args.tokens
    if args.scan_dir:
        tokens = min(args.cap_tokens, estimate_tokens_from_codebase(args.scan_dir))
        print(f"estimated_tokens={tokens} scan_dir={args.scan_dir}")

    result = run_synthetic_benchmark(
        profile_name=args.profile,
        tokens=tokens,
        layers=args.layers,
        heads=args.heads,
        head_dim=args.head_dim,
        dtype=_parse_dtype(args.dtype),
        device=args.device,
    )
    print(f"profile={args.profile}")
    print(
        "shape="
        f"layers={result.layers} tokens={result.tokens} heads={result.heads} head_dim={result.head_dim}"
    )
    print(f"memory={result.compressed_bytes}/{result.original_bytes} ratio={result.ratio:.4f}x")
    print(
        "quality="
        f"key_cos={result.key_cosine:.5f} value_cos={result.value_cosine:.5f} "
        f"attn_cos={result.attention_cosine:.5f}"
    )
    print(f"seconds={result.seconds:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
