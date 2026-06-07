from __future__ import annotations

from dataclasses import dataclass
from math import ceil, sqrt
from typing import Iterable, Literal, Sequence

import torch
from torch.nn import functional as F


def _tensor_nbytes(tensor: torch.Tensor | None) -> int:
    if tensor is None:
        return 0
    return tensor.numel() * tensor.element_size()


@dataclass(frozen=True)
class KvQuantConfig:
    """Configuration for group-wise KV-cache quantization.

    Expected KV tensor layout is [batch, heads, tokens, head_dim]. The defaults
    follow the robust first-pass pattern from KV-cache papers: quantize keys
    over token groups and values over channel/head-dim groups, while keeping a
    recent residual tail in full precision.
    """

    bits: int = 4
    group_size: int = 64
    residual_length: int = 128
    quantizer: str = "symmetric"
    key_group_axis: str = "token"
    value_group_axis: str = "channel"
    semantic_protection_ratio: float = 0.0
    min_protected_tokens: int = 0
    outlier_protection_ratio: float = 0.0
    min_outlier_tokens: int = 0
    pack_int4: bool = True
    scale_dtype: torch.dtype = torch.float16
    skip_if_not_smaller: bool = True

    def validate(self) -> None:
        if self.bits not in {2, 4, 8}:
            raise ValueError("bits must be 2, 4, or 8 for this implementation.")
        if self.group_size < 1:
            raise ValueError("group_size must be >= 1.")
        if self.residual_length < 0:
            raise ValueError("residual_length must be >= 0.")
        if self.quantizer not in {"symmetric", "affine"}:
            raise ValueError("quantizer must be 'symmetric' or 'affine'.")
        if self.key_group_axis not in {"token", "channel"}:
            raise ValueError("key_group_axis must be 'token' or 'channel'.")
        if self.value_group_axis not in {"token", "channel"}:
            raise ValueError("value_group_axis must be 'token' or 'channel'.")
        if not 0.0 <= self.semantic_protection_ratio <= 1.0:
            raise ValueError("semantic_protection_ratio must be in [0, 1].")
        if self.min_protected_tokens < 0:
            raise ValueError("min_protected_tokens must be >= 0.")
        if not 0.0 <= self.outlier_protection_ratio <= 1.0:
            raise ValueError("outlier_protection_ratio must be in [0, 1].")
        if self.min_outlier_tokens < 0:
            raise ValueError("min_outlier_tokens must be >= 0.")


@dataclass
class QuantizedTensor:
    q: torch.Tensor
    scales: torch.Tensor
    zero_points: torch.Tensor | None
    original_shape: tuple[int, ...]
    axis: int
    group_size: int
    bits: int
    padded: int
    q_shape: tuple[int, ...]
    original_dtype: torch.dtype
    packed: bool
    packed_offset: int

    @property
    def qmin(self) -> int:
        return -(1 << (self.bits - 1))

    def memory_bytes(self) -> int:
        return _tensor_nbytes(self.q) + _tensor_nbytes(self.scales) + _tensor_nbytes(self.zero_points)

    def dequantize(self) -> torch.Tensor:
        if self.packed:
            q = _unpack_lowbit(self.q, self.bits, self.q_shape, self.packed_offset)
        else:
            q = self.q.view(self.q_shape)

        group_shape = (*self.q_shape[:-1], self.q_shape[-1] // self.group_size, self.group_size)
        grouped_q = q.view(group_shape).to(torch.float32)
        if self.zero_points is not None:
            grouped_q = grouped_q - self.zero_points.to(torch.float32)
        grouped = grouped_q * self.scales.to(torch.float32)
        moved = grouped.reshape(self.q_shape)

        original_axis_len = self.original_shape[self.axis]
        if self.padded:
            moved = moved.narrow(-1, 0, original_axis_len)

        out = moved.movedim(-1, self.axis).contiguous()
        return out.to(self.original_dtype)


@dataclass
class QuantizedKvLayer:
    """One quantized layer of a transformer KV cache."""

    compressed_keys: QuantizedTensor | None
    compressed_values: QuantizedTensor | None
    compressed_indices: torch.Tensor
    retained_keys: torch.Tensor
    retained_values: torch.Tensor
    retained_indices: torch.Tensor
    original_shape: tuple[int, ...]
    seq_dim: int = 2
    passthrough_keys: torch.Tensor | None = None
    passthrough_values: torch.Tensor | None = None

    def memory_bytes(self) -> int:
        if self.passthrough_keys is not None and self.passthrough_values is not None:
            return _tensor_nbytes(self.passthrough_keys) + _tensor_nbytes(self.passthrough_values)
        return (
            (self.compressed_keys.memory_bytes() if self.compressed_keys is not None else 0)
            + (self.compressed_values.memory_bytes() if self.compressed_values is not None else 0)
            + _tensor_nbytes(self.compressed_indices)
            + _tensor_nbytes(self.retained_keys)
            + _tensor_nbytes(self.retained_values)
            + _tensor_nbytes(self.retained_indices)
        )

    def original_memory_bytes(self) -> int:
        dtype = self.retained_keys.dtype
        if self.passthrough_keys is not None:
            dtype = self.passthrough_keys.dtype
        if self.compressed_keys is not None:
            dtype = self.compressed_keys.original_dtype
        elements = 1
        for dim in self.original_shape:
            elements *= dim
        return 2 * elements * _dtype_element_size(dtype)

    def compression_ratio(self) -> float:
        return self.original_memory_bytes() / max(self.memory_bytes(), 1)

    def retained_token_count(self) -> int:
        if self.passthrough_keys is not None:
            return self.original_shape[self.seq_dim]
        return int(self.retained_indices.numel())

    def compressed_token_count(self) -> int:
        if self.passthrough_keys is not None:
            return 0
        return int(self.compressed_indices.numel())

    def dequantize(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.passthrough_keys is not None and self.passthrough_values is not None:
            return self.passthrough_keys, self.passthrough_values

        device = self.retained_keys.device
        dtype = self.retained_keys.dtype
        if self.compressed_keys is not None:
            device = self.compressed_keys.q.device
            dtype = self.compressed_keys.original_dtype

        keys = torch.empty(self.original_shape, dtype=dtype, device=device)
        values = torch.empty(self.original_shape, dtype=dtype, device=device)

        if self.compressed_keys is not None and self.compressed_indices.numel() > 0:
            keys.index_copy_(self.seq_dim, self.compressed_indices.long(), self.compressed_keys.dequantize())
        if self.compressed_values is not None and self.compressed_indices.numel() > 0:
            values.index_copy_(self.seq_dim, self.compressed_indices.long(), self.compressed_values.dequantize())
        if self.retained_indices.numel() > 0:
            retain_idx = self.retained_indices.long()
            keys.index_copy_(self.seq_dim, retain_idx, self.retained_keys)
            values.index_copy_(self.seq_dim, retain_idx, self.retained_values)

        return keys, values

    def report(self, original_keys: torch.Tensor, original_values: torch.Tensor) -> dict[str, float]:
        keys, values = self.dequantize()
        original_bytes = _tensor_nbytes(original_keys) + _tensor_nbytes(original_values)
        quantized_bytes = self.memory_bytes()
        return {
            "original_bytes": float(original_bytes),
            "quantized_bytes": float(quantized_bytes),
            "compression_ratio": original_bytes / max(quantized_bytes, 1),
            "retained_tokens": float(self.retained_token_count()),
            "compressed_tokens": float(self.compressed_token_count()),
            "key_mse": float(F.mse_loss(keys.float(), original_keys.float()).cpu()),
            "value_mse": float(F.mse_loss(values.float(), original_values.float()).cpu()),
            "key_cosine": _flat_cosine(keys, original_keys),
            "value_cosine": _flat_cosine(values, original_values),
        }

    def append(
        self,
        new_keys: torch.Tensor,
        new_values: torch.Tensor,
        config: KvQuantConfig,
        importance_scores: torch.Tensor | None = None,
    ) -> "QuantizedKvLayer":
        """Append new KV tokens and requantize.

        This is intentionally simple and correctness-first. A fused production
        path would avoid dequantizing old compressed blocks, but this gives us a
        reliable streaming interface to test policy decisions.
        """

        old_keys, old_values = self.dequantize()
        if old_keys.shape[: self.seq_dim] != new_keys.shape[: self.seq_dim]:
            raise ValueError("new_keys batch/head dimensions must match the cached layer.")
        if old_keys.shape[self.seq_dim + 1 :] != new_keys.shape[self.seq_dim + 1 :]:
            raise ValueError("new_keys head_dim dimensions must match the cached layer.")
        if new_keys.shape != new_values.shape:
            raise ValueError("new_keys and new_values must have the same shape.")
        keys = torch.cat([old_keys, new_keys], dim=self.seq_dim)
        values = torch.cat([old_values, new_values], dim=self.seq_dim)
        return quantize_kv_layer(keys, values, config, importance_scores=importance_scores)


@dataclass
class QuantizedKvCache:
    layers: list[QuantizedKvLayer]

    @classmethod
    def from_past_key_values(
        cls,
        past_key_values: Sequence[tuple[torch.Tensor, torch.Tensor]],
        config: KvQuantConfig,
        importance_scores: Sequence[torch.Tensor] | torch.Tensor | None = None,
    ) -> "QuantizedKvCache":
        layers = []
        for layer_idx, (keys, values) in enumerate(past_key_values):
            if isinstance(importance_scores, torch.Tensor) or importance_scores is None:
                scores = importance_scores
            else:
                scores = importance_scores[layer_idx]
            layers.append(quantize_kv_layer(keys, values, config, importance_scores=scores))
        return cls(layers)

    def dequantize(self) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        return tuple(layer.dequantize() for layer in self.layers)

    def memory_bytes(self) -> int:
        return sum(layer.memory_bytes() for layer in self.layers)

    def original_memory_bytes(self) -> int:
        return sum(layer.original_memory_bytes() for layer in self.layers)

    def compression_ratio(self) -> float:
        return self.original_memory_bytes() / max(self.memory_bytes(), 1)

    def append(
        self,
        new_past_key_values: Sequence[tuple[torch.Tensor, torch.Tensor]],
        config: KvQuantConfig,
        importance_scores: Sequence[torch.Tensor] | torch.Tensor | None = None,
    ) -> "QuantizedKvCache":
        if len(new_past_key_values) != len(self.layers):
            raise ValueError("new_past_key_values must have one (key, value) pair per cached layer.")
        layers = []
        for layer_idx, (layer, (new_keys, new_values)) in enumerate(zip(self.layers, new_past_key_values)):
            if isinstance(importance_scores, torch.Tensor) or importance_scores is None:
                scores = importance_scores
            else:
                scores = importance_scores[layer_idx]
            layers.append(layer.append(new_keys, new_values, config, importance_scores=scores))
        return QuantizedKvCache(layers)

    @classmethod
    def from_hf_cache(
        cls,
        cache: object,
        config: KvQuantConfig,
        importance_scores: Sequence[torch.Tensor] | torch.Tensor | None = None,
        cache_role: Literal["self", "cross"] = "self",
    ) -> "QuantizedKvCache":
        return cls.from_past_key_values(hf_cache_to_tuple(cache, cache_role=cache_role), config, importance_scores)

    def to_legacy_cache(self) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        """Return a legacy tuple cache for older generation code."""

        return self.dequantize()

    def to_hf_cache(
        self,
        implementation: Literal["dynamic", "static", "sliding_window", "legacy"] = "dynamic",
        *,
        config: object | None = None,
        max_cache_len: int | None = None,
    ) -> object:
        """Build a Hugging Face-compatible cache from dequantized tensors.

        The cache representation stores compressed tensors; this adapter is the
        correctness-first boundary for existing HF models. A fused low-bit
        attention path can later replace the dequantization inside this method.
        """

        if implementation == "legacy":
            return self.to_legacy_cache()
        if implementation == "dynamic":
            return self.to_hf_dynamic_cache(config=config)
        if implementation in {"static", "sliding_window"}:
            return self._to_hf_positioned_cache(implementation, config=config, max_cache_len=max_cache_len)
        raise ValueError("implementation must be 'dynamic', 'static', 'sliding_window', or 'legacy'.")

    def to_hf_dynamic_cache(self, *, config: object | None = None) -> object:
        try:
            from transformers.cache_utils import DynamicCache
        except ImportError as exc:
            raise RuntimeError("transformers is required to build a DynamicCache.") from exc

        cache = DynamicCache(config=config)
        for layer_idx, (keys, values) in enumerate(self.dequantize()):
            cache.update(keys, values, layer_idx)
        return cache

    def _to_hf_positioned_cache(
        self,
        implementation: Literal["static", "sliding_window"],
        *,
        config: object | None,
        max_cache_len: int | None,
    ) -> object:
        if config is None:
            raise ValueError("config is required for static or sliding_window HF cache export.")
        try:
            from transformers.cache_utils import SlidingWindowCache, StaticCache
        except ImportError as exc:
            raise RuntimeError("transformers is required to build a positioned cache.") from exc

        dequantized = self.dequantize()
        required_cache_len = max((keys.shape[2] for keys, _ in dequantized), default=0)
        if max_cache_len is None:
            max_cache_len = required_cache_len
        if max_cache_len < required_cache_len:
            raise ValueError("max_cache_len must be at least the longest KV sequence length.")
        cache_cls = StaticCache if implementation == "static" else SlidingWindowCache
        cache = cache_cls(config, max_cache_len=max_cache_len)
        for layer_idx, (keys, values) in enumerate(dequantized):
            positions = torch.arange(keys.shape[2], device=keys.device, dtype=torch.long)
            cache.update(keys, values, layer_idx, cache_kwargs={"cache_position": positions})
        return cache


def hf_cache_to_tuple(
    cache: object,
    *,
    cache_role: Literal["self", "cross"] = "self",
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Extract `(keys, values)` tuples from legacy or modern HF caches."""

    if hasattr(cache, "self_attention_cache") or hasattr(cache, "cross_attention_cache"):
        if cache_role == "self":
            selected = getattr(cache, "self_attention_cache", None)
        elif cache_role == "cross":
            selected = getattr(cache, "cross_attention_cache", None)
        else:
            raise ValueError("cache_role must be 'self' or 'cross'.")
        if selected is None:
            raise TypeError(f"HF encoder-decoder cache has no {cache_role!r} attention cache.")
        return hf_cache_to_tuple(selected)

    if isinstance(cache, (tuple, list)):
        layers = []
        for layer in cache:
            if layer is None:
                continue
            if not isinstance(layer, (tuple, list)) or len(layer) < 2:
                raise TypeError("Legacy cache layers must be tuples/lists with at least key and value tensors.")
            keys, values = layer[0], layer[1]
            if keys is None or values is None:
                continue
            layers.append((keys, values))
        return tuple(layers)

    layers = getattr(cache, "layers", None)
    if layers is not None:
        extracted = []
        for layer in layers:
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if keys is None or values is None:
                continue
            seq_len = _cache_layer_seq_length(layer, keys)
            if seq_len == 0:
                continue
            if seq_len < keys.shape[2]:
                keys = keys.narrow(2, 0, seq_len).contiguous()
                values = values.narrow(2, 0, seq_len).contiguous()
            extracted.append((keys, values))
        return tuple(extracted)

    key_cache = getattr(cache, "key_cache", None)
    value_cache = getattr(cache, "value_cache", None)
    if key_cache is not None and value_cache is not None:
        return tuple(
            (keys, values)
            for keys, values in zip(key_cache, value_cache)
            if keys is not None and values is not None
        )

    raise TypeError(f"Unsupported cache type: {type(cache)!r}")


def _cache_layer_seq_length(layer: object, keys: torch.Tensor) -> int:
    seq_len = None
    get_seq_length = getattr(layer, "get_seq_length", None)
    if callable(get_seq_length):
        try:
            seq_len = get_seq_length()
        except TypeError:
            seq_len = None
    if seq_len is None:
        seq_len = getattr(layer, "cumulative_length", None)
    if seq_len is None:
        return keys.shape[2]
    if isinstance(seq_len, torch.Tensor):
        if seq_len.numel() == 0:
            return 0
        seq_len = int(seq_len.max().item())
    else:
        seq_len = int(seq_len)
    return min(max(seq_len, 0), keys.shape[2])


def build_retention_mask(
    seq_len: int,
    config: KvQuantConfig,
    importance_scores: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return a token mask for full-precision retention.

    The recent residual tail is always retained. Older retained tokens are
    selected by importance score, which can come from Mneme-Graph retrieval or
    any other router.
    """

    config.validate()
    if device is None:
        device = importance_scores.device if importance_scores is not None else torch.device("cpu")

    mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    residual = min(config.residual_length, seq_len)
    if residual:
        mask[-residual:] = True

    older_len = seq_len - residual
    if importance_scores is None or older_len <= 0:
        return mask

    scores = _normalize_token_scores(seq_len, importance_scores, device)
    _apply_score_retention(
        mask,
        scores,
        candidate_len=older_len,
        ratio=config.semantic_protection_ratio,
        min_tokens=config.min_protected_tokens,
    )
    return mask


def quantize_kv_layer(
    keys: torch.Tensor,
    values: torch.Tensor,
    config: KvQuantConfig,
    importance_scores: torch.Tensor | None = None,
    retain_mask: torch.Tensor | None = None,
) -> QuantizedKvLayer:
    """Quantize one layer of KV tensors with layout [B, H, T, D]."""

    config.validate()
    if keys.shape != values.shape:
        raise ValueError("keys and values must have the same shape.")
    if keys.ndim != 4:
        raise ValueError("keys and values must use [batch, heads, tokens, head_dim] layout.")

    seq_dim = 2
    head_dim = 3
    seq_len = keys.size(seq_dim)
    device = keys.device

    if retain_mask is None:
        retain_mask = build_retention_mask(seq_len, config, importance_scores, device=device)
        if config.outlier_protection_ratio > 0.0 or config.min_outlier_tokens > 0:
            residual = min(config.residual_length, seq_len)
            older_len = seq_len - residual
            outlier_scores = _kv_outlier_scores(keys, values)
            _apply_score_retention(
                retain_mask,
                outlier_scores,
                candidate_len=older_len,
                ratio=config.outlier_protection_ratio,
                min_tokens=config.min_outlier_tokens,
            )
    else:
        retain_mask = retain_mask.to(device=device, dtype=torch.bool)
        if retain_mask.numel() != seq_len:
            raise ValueError("retain_mask must have one value per token.")

    all_indices = torch.arange(seq_len, device=device)
    retained_indices = all_indices[retain_mask].to(torch.int32)
    compressed_indices = all_indices[~retain_mask].to(torch.int32)

    retained_keys = keys.index_select(seq_dim, retained_indices.long()).contiguous()
    retained_values = values.index_select(seq_dim, retained_indices.long()).contiguous()

    compressed_keys = None
    compressed_values = None
    if compressed_indices.numel() > 0:
        key_src = keys.index_select(seq_dim, compressed_indices.long()).contiguous()
        value_src = values.index_select(seq_dim, compressed_indices.long()).contiguous()
        key_axis = seq_dim if config.key_group_axis == "token" else head_dim
        value_axis = seq_dim if config.value_group_axis == "token" else head_dim
        compressed_keys = _quantize_tensor(key_src, config, axis=key_axis)
        compressed_values = _quantize_tensor(value_src, config, axis=value_axis)

    layer = QuantizedKvLayer(
        compressed_keys=compressed_keys,
        compressed_values=compressed_values,
        compressed_indices=compressed_indices,
        retained_keys=retained_keys,
        retained_values=retained_values,
        retained_indices=retained_indices,
        original_shape=tuple(keys.shape),
        seq_dim=seq_dim,
    )
    if config.skip_if_not_smaller and layer.memory_bytes() >= layer.original_memory_bytes():
        return _full_precision_layer(keys, values, seq_dim)
    return layer


def chunk_scores_to_token_scores(
    seq_len: int,
    chunk_spans: Iterable[tuple[int, int]],
    chunk_scores: torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Spread chunk-level retrieval scores over token positions."""

    if device is None:
        device = chunk_scores.device
    token_scores = torch.full((seq_len,), -torch.inf, dtype=torch.float32, device=device)
    scores = chunk_scores.to(device=device, dtype=torch.float32).flatten()

    for idx, (start, end) in enumerate(chunk_spans):
        if idx >= scores.numel():
            break
        lo = max(0, int(start))
        hi = min(seq_len, int(end))
        if hi <= lo:
            continue
        token_scores[lo:hi] = torch.maximum(token_scores[lo:hi], scores[idx])

    return torch.nan_to_num(token_scores, neginf=0.0)


def attention_from_kv(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute scaled-dot attention for [B, H, Q, D] x [B, H, T, D]."""

    if query.ndim != 4 or keys.ndim != 4 or values.ndim != 4:
        raise ValueError("query, keys, and values must be rank-4 tensors.")
    if keys.shape != values.shape:
        raise ValueError("keys and values must have the same shape.")
    if query.shape[:2] != keys.shape[:2] or query.shape[-1] != keys.shape[-1]:
        raise ValueError("query and KV batch/head/head_dim dimensions must match.")

    scores = query.float().matmul(keys.float().transpose(-2, -1)) / sqrt(query.size(-1))
    if mask is not None:
        scores = scores.masked_fill(~mask.to(device=scores.device, dtype=torch.bool), -1.0e9)
    weights = F.softmax(scores, dim=-1)
    out = weights.matmul(values.float())
    return out, weights


def attention_report(
    query: torch.Tensor,
    original_keys: torch.Tensor,
    original_values: torch.Tensor,
    quantized_layer: QuantizedKvLayer,
    mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Compare attention outputs before and after KV quantization."""

    quant_keys, quant_values = quantized_layer.dequantize()
    original_out, original_weights = attention_from_kv(query, original_keys, original_values, mask)
    quant_out, quant_weights = attention_from_kv(query, quant_keys, quant_values, mask)
    kl = (
        original_weights.clamp_min(1.0e-9)
        * (original_weights.clamp_min(1.0e-9).log() - quant_weights.clamp_min(1.0e-9).log())
    ).sum(dim=-1).mean()

    return {
        "attention_output_mse": float(F.mse_loss(quant_out, original_out).cpu()),
        "attention_output_cosine": _flat_cosine(quant_out, original_out),
        "attention_output_max_abs": float((quant_out - original_out).abs().max().cpu()),
        "attention_weight_kl": float(kl.cpu()),
    }


def _quantize_tensor(x: torch.Tensor, config: KvQuantConfig, axis: int) -> QuantizedTensor:
    axis = axis % x.ndim
    moved = x.detach().movedim(axis, -1).contiguous()
    axis_len = moved.shape[-1]
    group_size = min(config.group_size, axis_len)
    padded = (-axis_len) % group_size
    if padded:
        moved = F.pad(moved, (0, padded))

    q_shape = tuple(moved.shape)
    grouped = moved.reshape(*q_shape[:-1], q_shape[-1] // group_size, group_size)
    grouped_f = grouped.to(torch.float32)

    if config.quantizer == "affine":
        qmax = (1 << config.bits) - 1
        qmin = 0
        mins = grouped_f.amin(dim=-1, keepdim=True)
        maxs = grouped_f.amax(dim=-1, keepdim=True)
        ranges = maxs - mins
        degenerate = ranges < 1.0e-8
        fallback_scales = grouped_f.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8) / qmax
        scales = torch.where(degenerate, fallback_scales, ranges.clamp_min(1.0e-8) / qmax)
        zero_points_f = torch.round(-mins / scales).clamp(qmin, qmax)
        degenerate_zero_points = torch.where(
            mins < 0,
            torch.full_like(zero_points_f, float(qmax)),
            torch.zeros_like(zero_points_f),
        )
        zero_points = torch.where(degenerate, degenerate_zero_points, zero_points_f).to(torch.uint8)
        q = torch.round(grouped_f / scales + zero_points.to(torch.float32)).clamp(qmin, qmax).to(torch.uint8)
        packed_offset = 0
    else:
        qmax = (1 << (config.bits - 1)) - 1
        qmin = -(1 << (config.bits - 1))
        scales = grouped_f.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8) / qmax
        zero_points = None
        q = torch.round(grouped_f / scales).clamp(qmin, qmax).to(torch.int8)
        packed_offset = qmin
    q = q.reshape(q_shape)

    packed = config.bits in {2, 4} and config.pack_int4
    if packed:
        q_store = _pack_lowbit(q, config.bits, packed_offset)
    else:
        q_store = q

    return QuantizedTensor(
        q=q_store,
        scales=scales.to(config.scale_dtype).contiguous(),
        zero_points=zero_points.contiguous() if zero_points is not None else None,
        original_shape=tuple(x.shape),
        axis=axis,
        group_size=group_size,
        bits=config.bits,
        padded=padded,
        q_shape=q_shape,
        original_dtype=x.dtype,
        packed=packed,
        packed_offset=packed_offset,
    )


def _full_precision_layer(keys: torch.Tensor, values: torch.Tensor, seq_dim: int) -> QuantizedKvLayer:
    empty_indices = torch.empty(0, dtype=torch.int32, device=keys.device)
    empty_shape = list(keys.shape)
    empty_shape[seq_dim] = 0
    empty_keys = keys.new_empty(tuple(empty_shape))
    empty_values = values.new_empty(tuple(empty_shape))
    return QuantizedKvLayer(
        compressed_keys=None,
        compressed_values=None,
        compressed_indices=empty_indices,
        retained_keys=empty_keys,
        retained_values=empty_values,
        retained_indices=empty_indices,
        original_shape=tuple(keys.shape),
        seq_dim=seq_dim,
        passthrough_keys=keys.detach().contiguous(),
        passthrough_values=values.detach().contiguous(),
    )


def _pack_lowbit(q: torch.Tensor, bits: int, offset: int) -> torch.Tensor:
    values_per_byte = 8 // bits
    encoded = (q.to(torch.int16) - offset).to(torch.uint8).flatten()
    pad = (-encoded.numel()) % values_per_byte
    if pad:
        encoded = F.pad(encoded, (0, pad))

    encoded = encoded.view(-1, values_per_byte)
    packed = torch.zeros((encoded.size(0),), dtype=torch.uint8, device=encoded.device)
    for idx in range(values_per_byte):
        packed = packed | (encoded[:, idx] << (idx * bits))
    return packed.contiguous()


def _unpack_lowbit(packed: torch.Tensor, bits: int, q_shape: tuple[int, ...], offset: int) -> torch.Tensor:
    values_per_byte = 8 // bits
    mask = (1 << bits) - 1
    values = []
    for idx in range(values_per_byte):
        values.append((packed >> (idx * bits)) & mask)
    encoded = torch.stack(values, dim=1).flatten()
    total = 1
    for dim in q_shape:
        total *= dim
    encoded = encoded[:total]
    q = encoded.to(torch.int16) + offset
    if offset == 0:
        return q.to(torch.uint8).view(q_shape)
    return q.to(torch.int8).view(q_shape)


def _flat_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    cosine = F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0)
    return float(cosine.cpu())


def _dtype_element_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _normalize_token_scores(
    seq_len: int,
    scores: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    scores = scores.to(device=device, dtype=torch.float32)
    if scores.numel() == seq_len:
        return scores.reshape(seq_len)
    if scores.ndim >= 1 and scores.shape[-1] == seq_len:
        return scores.reshape(-1, seq_len).mean(dim=0)
    if scores.ndim >= 1 and scores.shape[0] == seq_len:
        return scores.reshape(seq_len, -1).mean(dim=1)
    raise ValueError("importance_scores must have one score per token or a final token dimension.")


def _apply_score_retention(
    mask: torch.Tensor,
    scores: torch.Tensor,
    *,
    candidate_len: int,
    ratio: float,
    min_tokens: int,
) -> None:
    keep = min(candidate_len, max(ceil(candidate_len * ratio), min_tokens))
    if keep <= 0:
        return
    top = torch.topk(scores[:candidate_len], k=keep, largest=True).indices
    mask[top] = True


def _kv_outlier_scores(keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    key_score = keys.float().square().mean(dim=(0, 1, 3))
    value_score = values.float().square().mean(dim=(0, 1, 3))
    return key_score + value_score
