"""GraphKV: graph-guided low-bit KV cache compression."""

from .core import (
    KvQuantConfig,
    QuantizedKvCache,
    QuantizedKvLayer,
    attention_from_kv,
    attention_report,
    build_retention_mask,
    chunk_scores_to_token_scores,
    hf_cache_to_tuple,
    quantize_kv_layer,
)
from .graph_model import (
    GraphFactBank,
    GraphMemory17L,
    GraphMemoryConfig,
    GraphMnemeRequest,
    GraphQueryBank,
    GraphRetentionChunk,
    build_candidate_bank,
    build_fact_bank,
    build_graph_mneme_token_scores,
    build_query_bank,
    bundled_graph_model_metadata,
    bundled_graph_model_dir,
    graph_mneme_chunk_scores,
    graph_mneme_request_token_scores,
    graph_mneme_token_scores,
    load_bundled_graph_model,
    normalize_graph_scores,
)
from .profiles import GraphKVProfile, get_profile, list_profiles

_INTEGRATION_EXPORTS = {
    "llama_cpp_args",
    "llama_cpp_command",
    "llama_cpp_env",
    "quantize_hf_cache",
    "vllm_cli_args",
    "vllm_kwargs",
}

__all__ = [
    "GraphKVProfile",
    "GraphFactBank",
    "GraphMemory17L",
    "GraphMemoryConfig",
    "GraphMnemeRequest",
    "GraphQueryBank",
    "GraphRetentionChunk",
    "KvQuantConfig",
    "QuantizedKvCache",
    "QuantizedKvLayer",
    "attention_from_kv",
    "attention_report",
    "build_retention_mask",
    "build_candidate_bank",
    "build_fact_bank",
    "build_graph_mneme_token_scores",
    "build_query_bank",
    "bundled_graph_model_metadata",
    "bundled_graph_model_dir",
    "chunk_scores_to_token_scores",
    "get_profile",
    "graph_mneme_chunk_scores",
    "graph_mneme_request_token_scores",
    "graph_mneme_token_scores",
    "hf_cache_to_tuple",
    "list_profiles",
    "load_bundled_graph_model",
    "llama_cpp_args",
    "llama_cpp_command",
    "llama_cpp_env",
    "normalize_graph_scores",
    "quantize_hf_cache",
    "quantize_kv_layer",
    "vllm_cli_args",
    "vllm_kwargs",
]

__version__ = "0.1.0"


def __getattr__(name: str):
    if name in _INTEGRATION_EXPORTS:
        from . import integrations

        value = getattr(integrations, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
