from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .core import KvQuantConfig


Engine = Literal["graphkv", "vllm", "llama.cpp"]


GRAPHKV_DEFAULT_CONFIG = KvQuantConfig(
    bits=4,
    group_size=64,
    residual_length=256,
    sink_length=128,
    quantizer="affine",
    long_context_threshold=8192,
    long_context_quantizer="symmetric",
    key_group_axis="token",
    value_group_axis="channel",
    semantic_protection_ratio=0.01,
    semantic_protection_min_context=8192,
)


QWEN7_NF4_COMPARISON_CONFIG = KvQuantConfig(
    bits=4,
    group_size=32,
    residual_length=512,
    sink_length=128,
    quantizer="affine",
    long_context_threshold=8192,
    long_context_quantizer="symmetric",
    key_group_axis="token",
    value_group_axis="channel",
    semantic_protection_ratio=0.02,
    semantic_protection_min_context=8192,
)


@dataclass(frozen=True)
class GraphKVProfile:
    name: str
    engine: Engine
    description: str
    config: KvQuantConfig | None = None
    vllm_kv_cache_dtype: str | None = None
    vllm_calculate_kv_scales: bool | None = None
    llama_cache_type_k: str | None = None
    llama_cache_type_v: str | None = None

    @property
    def is_native_graphkv(self) -> bool:
        return self.engine == "graphkv"


PROFILES: dict[str, GraphKVProfile] = {
    "graphkv": GraphKVProfile(
        name="graphkv",
        engine="graphkv",
        description="Default GraphKV profile with built-in Mneme-guided long-context retention.",
        config=GRAPHKV_DEFAULT_CONFIG,
    ),
    "graphkv-int2-max": GraphKVProfile(
        name="graphkv-int2-max",
        engine="graphkv",
        description="Maximum compression profile for Transformers cache experiments.",
        config=KvQuantConfig(
            bits=2,
            group_size=128,
            residual_length=0,
            quantizer="affine",
            key_group_axis="token",
            value_group_axis="channel",
        ),
    ),
    "graphkv-int4-balanced": GraphKVProfile(
        name="graphkv-int4-balanced",
        engine="graphkv",
        description="Balanced profile for grouped-query models such as Qwen2.",
        config=KvQuantConfig(
            bits=4,
            group_size=64,
            residual_length=0,
            quantizer="symmetric",
            key_group_axis="token",
            value_group_axis="channel",
        ),
    ),
    "graphkv-int4-safe": GraphKVProfile(
        name="graphkv-int4-safe",
        engine="graphkv",
        description="Safer 4-bit profile with a recent residual tail and outlier retention.",
        config=KvQuantConfig(
            bits=4,
            group_size=64,
            residual_length=96,
            sink_length=32,
            quantizer="symmetric",
            key_group_axis="token",
            value_group_axis="channel",
            outlier_protection_ratio=0.02,
            min_outlier_tokens=1,
        ),
    ),
    "graphkv-qwen7-nf4": GraphKVProfile(
        name="graphkv-qwen7-nf4",
        engine="graphkv",
        description="Conservative Qwen2.5-7B NF4 profile used for local comparison rows.",
        config=QWEN7_NF4_COMPARISON_CONFIG,
    ),
    "vllm-fp8": GraphKVProfile(
        name="vllm-fp8",
        engine="vllm",
        description="Native vLLM FP8 KV cache with default scales.",
        vllm_kv_cache_dtype="fp8",
        vllm_calculate_kv_scales=False,
    ),
    "vllm-fp8-calibrated": GraphKVProfile(
        name="vllm-fp8-calibrated",
        engine="vllm",
        description="Native vLLM FP8 KV cache with warmup scale calculation.",
        vllm_kv_cache_dtype="fp8",
        vllm_calculate_kv_scales=True,
    ),
    "llamacpp-q8": GraphKVProfile(
        name="llamacpp-q8",
        engine="llama.cpp",
        description="Native llama.cpp Q8_0 KV cache. Usually conservative.",
        llama_cache_type_k="q8_0",
        llama_cache_type_v="q8_0",
    ),
    "llamacpp-q4": GraphKVProfile(
        name="llamacpp-q4",
        engine="llama.cpp",
        description="Native llama.cpp Q4_0 KV cache. More aggressive.",
        llama_cache_type_k="q4_0",
        llama_cache_type_v="q4_0",
    ),
}


def get_profile(name: str | GraphKVProfile) -> GraphKVProfile:
    if isinstance(name, GraphKVProfile):
        return name
    try:
        return PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(PROFILES))
        raise KeyError(f"Unknown GraphKV profile {name!r}. Choices: {choices}") from exc


def list_profiles() -> tuple[GraphKVProfile, ...]:
    return tuple(PROFILES.values())
