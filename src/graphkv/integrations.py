from __future__ import annotations

import argparse
import shlex
from typing import Iterable

from .core import QuantizedKvCache, hf_cache_to_tuple
from .graph_model import GraphMemory17L, GraphMemoryRequest, graph_mneme_request_token_scores
from .profiles import GraphKVProfile, get_profile, list_profiles


def quantize_hf_cache(
    past_key_values: object,
    profile: str | GraphKVProfile = "graphkv",
    *,
    importance_scores=None,
    graph_memory: GraphMemoryRequest | None = None,
    graph_mneme: GraphMemoryRequest | None = None,
    graph_mneme_model: GraphMemory17L | None = None,
    graph_mneme_device: str = "auto",
    output: str = "dynamic",
    model_config: object | None = None,
    max_cache_len: int | None = None,
) -> object:
    """Quantize a Hugging Face cache and export it in the requested format.

    `output` may be `compressed`, `legacy`, `dynamic`, `static`, or
    `sliding_window`. The exported HF cache is correctness-first: it dequantizes
    before returning to the engine. For real speedups, use GraphKV's packed cache
    directly or add fused low-bit attention kernels.

    When `graph_memory` is supplied, GraphKV scores it with the bundled Mneme
    model and uses those scores for token retention and low-priority bit
    routing. Explicit `importance_scores` take precedence over graph memory.
    """

    selected = get_profile(profile)
    if selected.config is None:
        raise ValueError(f"Profile {selected.name!r} is an engine-native recipe, not a GraphKV profile.")
    graph_request = graph_memory if graph_memory is not None else graph_mneme
    if importance_scores is None and graph_request is not None:
        importance_scores = graph_mneme_request_token_scores(
            _hf_cache_seq_len(past_key_values),
            graph_request,
            model=graph_mneme_model,
            device=graph_mneme_device,
        )
    compressed = QuantizedKvCache.from_hf_cache(
        past_key_values,
        selected.config,
        importance_scores=importance_scores,
    )
    if output == "compressed":
        return compressed
    return compressed.to_hf_cache(output, config=model_config, max_cache_len=max_cache_len)


def _hf_cache_seq_len(past_key_values: object) -> int:
    layers = hf_cache_to_tuple(past_key_values)
    if not layers:
        raise ValueError("past_key_values must contain at least one non-empty KV layer.")
    return int(layers[0][0].shape[2])


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
