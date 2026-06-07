from __future__ import annotations

import argparse
import shlex
from typing import Iterable

from .core import QuantizedKvCache
from .profiles import GraphKVProfile, get_profile, list_profiles


def quantize_hf_cache(
    past_key_values: object,
    profile: str | GraphKVProfile = "graphkv-int4-balanced",
    *,
    importance_scores=None,
    output: str = "dynamic",
    model_config: object | None = None,
    max_cache_len: int | None = None,
) -> object:
    """Quantize a Hugging Face cache and export it in the requested format.

    `output` may be `compressed`, `legacy`, `dynamic`, `static`, or
    `sliding_window`. The exported HF cache is correctness-first: it dequantizes
    before returning to the engine. For real speedups, use GraphKV's packed cache
    directly or add fused low-bit attention kernels.
    """

    selected = get_profile(profile)
    if selected.config is None:
        raise ValueError(f"Profile {selected.name!r} is an engine-native recipe, not a GraphKV profile.")
    compressed = QuantizedKvCache.from_hf_cache(
        past_key_values,
        selected.config,
        importance_scores=importance_scores,
    )
    if output == "compressed":
        return compressed
    return compressed.to_hf_cache(output, config=model_config, max_cache_len=max_cache_len)


def vllm_kwargs(
    profile: str | GraphKVProfile = "vllm-fp8-calibrated",
    **overrides,
) -> dict[str, object]:
    """Return keyword arguments for `vllm.LLM(...)`.

    vLLM currently exposes native FP8 KV cache settings. GraphKV int2/int4
    kernels would require a vLLM attention/cache backend patch, so this helper
    returns the best native vLLM settings rather than pretending otherwise.
    """

    selected = get_profile(profile)
    if selected.engine != "vllm":
        raise ValueError(f"Profile {selected.name!r} is not a vLLM profile.")
    kwargs: dict[str, object] = {
        "kv_cache_dtype": selected.vllm_kv_cache_dtype,
        "calculate_kv_scales": selected.vllm_calculate_kv_scales,
    }
    kwargs.update(overrides)
    return kwargs


def vllm_cli_args(profile: str | GraphKVProfile = "vllm-fp8-calibrated") -> list[str]:
    selected = get_profile(profile)
    kwargs = vllm_kwargs(selected)
    args = ["--kv-cache-dtype", str(kwargs["kv_cache_dtype"])]
    if kwargs.get("calculate_kv_scales"):
        args.append("--calculate-kv-scales")
    return args


def llama_cpp_args(profile: str | GraphKVProfile = "llamacpp-q8") -> list[str]:
    selected = get_profile(profile)
    if selected.engine != "llama.cpp":
        raise ValueError(f"Profile {selected.name!r} is not a llama.cpp profile.")
    return [
        "--cache-type-k",
        str(selected.llama_cache_type_k),
        "--cache-type-v",
        str(selected.llama_cache_type_v),
    ]


def llama_cpp_env(profile: str | GraphKVProfile = "llamacpp-q8") -> dict[str, str]:
    selected = get_profile(profile)
    if selected.engine != "llama.cpp":
        raise ValueError(f"Profile {selected.name!r} is not a llama.cpp profile.")
    return {
        "LLAMA_ARG_CACHE_TYPE_K": str(selected.llama_cache_type_k),
        "LLAMA_ARG_CACHE_TYPE_V": str(selected.llama_cache_type_v),
    }


def llama_cpp_command(
    model_path: str,
    *,
    profile: str | GraphKVProfile = "llamacpp-q8",
    binary: str = "llama-cli",
    extra_args: Iterable[str] = (),
) -> str:
    parts = [binary, "-m", model_path, *llama_cpp_args(profile), *extra_args]
    return " ".join(shlex.quote(part) for part in parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print GraphKV integration recipes.")
    parser.add_argument("--profile", default=None, help="Profile name. Omit to list all profiles.")
    parser.add_argument("--engine", choices=["vllm", "llama.cpp"], default=None)
    parser.add_argument("--model", default="model.gguf", help="Model path for llama.cpp command output.")
    args = parser.parse_args(argv)

    if args.profile is None:
        for profile in list_profiles():
            print(f"{profile.name:24} {profile.engine:10} {profile.description}")
        return 0

    profile = get_profile(args.profile)
    if args.engine == "vllm" or profile.engine == "vllm":
        print("vLLM kwargs:", vllm_kwargs(profile))
        print("vLLM CLI:", " ".join(vllm_cli_args(profile)))
    elif args.engine == "llama.cpp" or profile.engine == "llama.cpp":
        print("llama.cpp args:", " ".join(llama_cpp_args(profile)))
        print("llama.cpp env:", llama_cpp_env(profile))
        print("llama.cpp command:", llama_cpp_command(args.model, profile=profile))
    else:
        print(profile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
