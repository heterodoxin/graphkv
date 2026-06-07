from __future__ import annotations

import json
import random
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Iterable, Literal, Sequence

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class GraphMemoryConfig:
    model_name: str = "mneme-graph-17l"
    num_entities: int = 128
    num_facts: int = 64
    num_cand_facts: int = 32
    num_candidates: int = 64
    num_relations: int = 512
    d_model: int = 256
    num_heads: int = 8
    encoder_layers: int = 6
    dropout: float = 0.05
    num_global_relations: int = 0

    @classmethod
    def from_mapping(cls, data: dict) -> "GraphMemoryConfig":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: data[key] for key in allowed if key in data})


@dataclass(frozen=True)
class GraphFactBank:
    src: torch.Tensor
    rel: torch.Tensor
    dst: torch.Tensor
    relation_map: dict[int, int]
    entity_map: dict[int, int]

    def as_batch(self, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.src.unsqueeze(0).to(device),
            self.rel.unsqueeze(0).to(device),
            self.dst.unsqueeze(0).to(device),
        )


@dataclass(frozen=True)
class GraphQueryBank:
    facts: GraphFactBank
    query_relation: int


@dataclass(frozen=True)
class GraphRetentionChunk:
    """A text/token span backed by local graph facts."""

    anchor: int
    span: tuple[int, int]
    facts: Sequence[tuple[int, int, int]]
    shuffle_seed: int | None = None


@dataclass(frozen=True)
class GraphMemoryRequest:
    """Graph query plus candidate chunks for automatic KV-retention scoring."""

    query_anchor: int
    query_relation: int
    query_facts: Sequence[tuple[int, int, int]]
    chunks: Sequence[GraphRetentionChunk]
    normalize: Literal["none", "minmax", "zscore"] = "minmax"
    shuffle_seed: int | None = None


GraphMnemeRequest = GraphMemoryRequest


class FactGraphAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.05):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=1e-5)
        self.attn_in = nn.Linear(d_model, d_model * 3)
        self.attn_out = nn.Linear(d_model, d_model)
        self.ff_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.ff1 = nn.Linear(d_model, d_model * 4)
        self.ff2 = nn.Linear(d_model * 4, d_model)
        self.drop = nn.Dropout(dropout)
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, d_model = x.shape
        head_dim = d_model // self.num_heads
        normed = self.norm(x)
        q, k, v = self.attn_in(normed).chunk(3, dim=-1)

        def split(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch, seq, self.num_heads, head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(split(q), split(k), split(v), dropout_p=0.0)
        out = out.transpose(1, 2).contiguous().view(batch, seq, d_model)
        x = x + self.drop(self.attn_out(out))
        return x + self.drop(self.ff2(F.gelu(self.ff1(self.ff_norm(x)))))


class GraphMemory17L(nn.Module):
    """Bundled custom graph-memory model used to score retention candidates."""

    def __init__(self, cfg: GraphMemoryConfig):
        super().__init__()
        self.cfg = cfg
        d_model = cfg.d_model
        self.entity_emb = nn.Embedding(cfg.num_entities, d_model)
        self.rel_emb = nn.Embedding(cfg.num_relations, d_model)
        self.src_role = nn.Parameter(torch.zeros(d_model))
        self.dst_role = nn.Parameter(torch.zeros(d_model))
        self.fact_mlp = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 3),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d_model * 3, d_model),
        )
        self.fact_pos = nn.Parameter(torch.zeros(1, max(cfg.num_facts, cfg.num_cand_facts), d_model))
        self.layers = nn.ModuleList(
            [FactGraphAttention(d_model, cfg.num_heads, cfg.dropout) for _ in range(cfg.encoder_layers)]
        )
        self.query_cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.query_rel_proj = nn.Linear(d_model, d_model, bias=False)
        self.query_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.query_head = nn.Linear(d_model, d_model, bias=False)
        self.entity_cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.entity_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.entity_head = nn.Linear(d_model, d_model, bias=False)
        self.log_temp = nn.Parameter(torch.zeros(1))

    def _build_content_and_facts(
        self,
        fact_src: torch.Tensor,
        fact_rel: torch.Tensor,
        fact_dst: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src = self.entity_emb(fact_src) + self.src_role
        rel = self.rel_emb(fact_rel)
        dst = self.entity_emb(fact_dst) + self.dst_role
        content = self.fact_mlp(torch.cat([src, rel, dst], dim=-1))
        return content, content + self.fact_pos[:, : fact_src.shape[1]]

    def encode_query(
        self,
        fact_src: torch.Tensor,
        fact_rel: torch.Tensor,
        fact_dst: torch.Tensor,
        query_relation: torch.Tensor,
    ) -> torch.Tensor:
        content, facts = self._build_content_and_facts(fact_src, fact_rel, fact_dst)
        rel_mask = (fact_rel == query_relation.unsqueeze(1)).float()
        rel_sum = (content * rel_mask.unsqueeze(-1)).sum(1)
        rel_count = rel_mask.sum(1, keepdim=True).clamp(min=1)
        rel_proto = self.query_rel_proj(rel_sum / rel_count)
        batch = facts.shape[0]
        seq = torch.cat([self.query_cls.expand(batch, -1, -1), rel_proto.unsqueeze(1), facts], dim=1)
        for layer in self.layers:
            seq = layer(seq)
        return self.query_head(self.query_norm(seq[:, 0]))

    def encode_entity(
        self,
        fact_src: torch.Tensor,
        fact_rel: torch.Tensor,
        fact_dst: torch.Tensor,
    ) -> torch.Tensor:
        _, facts = self._build_content_and_facts(fact_src, fact_rel, fact_dst)
        batch = facts.shape[0]
        seq = torch.cat([self.entity_cls.expand(batch, -1, -1), facts], dim=1)
        for layer in self.layers:
            seq = layer(seq)
        return self.entity_head(self.entity_norm(seq[:, 0]))

    @torch.no_grad()
    def score(
        self,
        query: GraphQueryBank,
        candidates: Sequence[GraphFactBank],
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        if len(candidates) == 0:
            return torch.empty(0, dtype=torch.float32, device=device)
        q_src, q_rel, q_dst = query.facts.as_batch(device)
        qrel = torch.tensor([query.query_relation], dtype=torch.long, device=device)
        query_vec = self.encode_query(q_src, q_rel, q_dst, qrel)
        c_src = torch.stack([candidate.src for candidate in candidates], dim=0).to(device)
        c_rel = torch.stack([candidate.rel for candidate in candidates], dim=0).to(device)
        c_dst = torch.stack([candidate.dst for candidate in candidates], dim=0).to(device)
        candidate_vecs = self.encode_entity(c_src, c_rel, c_dst)
        temp = self.log_temp.exp().clamp(0.1, 10.0)
        return (query_vec @ candidate_vecs.t()).squeeze(0) / temp


def bundled_graph_model_dir() -> Path:
    return Path(str(files("graphkv").joinpath("models", "mneme-graph-17l")))


def bundled_graph_model_metadata() -> dict[str, object]:
    """Return the bundled Graph Mneme model card/config JSON."""

    model_dir = bundled_graph_model_dir()
    return json.loads((model_dir / "latest.json").read_text())


def load_bundled_graph_model(device: str | torch.device = "auto") -> GraphMemory17L:
    from safetensors.torch import load_file

    model_dir = bundled_graph_model_dir()
    device = _resolve_graph_device(device)
    cfg = GraphMemoryConfig.from_mapping(json.loads((model_dir / "latest.json").read_text()))
    model = GraphMemory17L(cfg).to(device)
    model.load_state_dict(load_file(model_dir / "latest.safetensors", device=str(device)))
    model.eval()
    return model


def graph_mneme_chunk_scores(
    query: GraphQueryBank,
    candidates: Sequence[GraphFactBank],
    *,
    model: GraphMemory17L | None = None,
    device: torch.device | str | None = None,
    normalize: Literal["none", "minmax", "zscore"] = "minmax",
) -> torch.Tensor:
    """Score graph-backed chunks with the bundled Graph Mneme model."""

    if model is None:
        model = load_bundled_graph_model(device=device or "auto")
    resolved_device = _resolve_graph_device(device) if device is not None else next(model.parameters()).device
    model = model.to(resolved_device)
    scores = model.score(query, candidates, device=resolved_device)
    return normalize_graph_scores(scores, mode=normalize)


def graph_mneme_token_scores(
    seq_len: int,
    chunk_spans: Iterable[tuple[int, int]],
    query: GraphQueryBank,
    candidates: Sequence[GraphFactBank],
    *,
    model: GraphMemory17L | None = None,
    device: torch.device | str | None = None,
    normalize: Literal["none", "minmax", "zscore"] = "minmax",
) -> torch.Tensor:
    """Return one Graph Mneme importance score per token for KV retention."""

    from .core import chunk_scores_to_token_scores

    scores = graph_mneme_chunk_scores(
        query,
        candidates,
        model=model,
        device=device,
        normalize=normalize,
    )
    return chunk_scores_to_token_scores(seq_len, chunk_spans, scores, device=scores.device)


def build_graph_mneme_token_scores(
    seq_len: int,
    *,
    query_anchor: int,
    query_relation: int,
    query_facts: Iterable[tuple[int, int, int]],
    chunks: Sequence[GraphRetentionChunk],
    model: GraphMemory17L | None = None,
    device: torch.device | str | None = None,
    normalize: Literal["none", "minmax", "zscore"] = "minmax",
    shuffle_seed: int | None = None,
) -> torch.Tensor:
    """Build query/candidate banks, score chunks, and spread scores over tokens."""

    if model is None:
        model = load_bundled_graph_model(device=device or "auto")
    if device is not None:
        device = _resolve_graph_device(device)
    cfg = model.cfg
    query = build_query_bank(
        query_anchor,
        query_relation,
        query_facts,
        cfg,
        shuffle_seed=shuffle_seed,
    )
    candidates = [
        build_candidate_bank(
            chunk.anchor,
            chunk.facts,
            cfg,
            shuffle_seed=chunk.shuffle_seed,
        )
        for chunk in chunks
    ]
    return graph_mneme_token_scores(
        seq_len,
        [chunk.span for chunk in chunks],
        query,
        candidates,
        model=model,
        device=device,
        normalize=normalize,
    )


def graph_mneme_request_token_scores(
    seq_len: int,
    request: GraphMemoryRequest,
    *,
    model: GraphMemory17L | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Score a `GraphMemoryRequest` over a KV cache sequence length."""

    return build_graph_mneme_token_scores(
        seq_len,
        query_anchor=request.query_anchor,
        query_relation=request.query_relation,
        query_facts=request.query_facts,
        chunks=request.chunks,
        model=model,
        device=device,
        normalize=request.normalize,
        shuffle_seed=request.shuffle_seed,
    )


def normalize_graph_scores(
    scores: torch.Tensor,
    *,
    mode: Literal["none", "minmax", "zscore"] = "minmax",
) -> torch.Tensor:
    """Normalize graph scores while preserving their ranking."""

    scores = scores.to(dtype=torch.float32)
    if mode == "none" or scores.numel() == 0:
        return scores
    if mode == "minmax":
        lo = scores.min()
        hi = scores.max()
        if bool((hi - lo).abs() < 1.0e-8):
            return torch.ones_like(scores)
        return (scores - lo) / (hi - lo)
    if mode == "zscore":
        mean = scores.mean()
        std = scores.std(unbiased=False)
        if bool(std < 1.0e-8):
            return torch.zeros_like(scores)
        return (scores - mean) / std
    raise ValueError("mode must be 'none', 'minmax', or 'zscore'.")


def build_fact_bank(
    anchor: int,
    facts: Iterable[tuple[int, int, int]],
    *,
    num_facts: int,
    num_entities: int,
    num_relations: int,
    shuffle_seed: int | None = None,
) -> GraphFactBank:
    fact_list = list(facts)
    if shuffle_seed is not None:
        rng = random.Random(shuffle_seed)
        rng.shuffle(fact_list)
    entity_map: dict[int, int] = {int(anchor): 0}
    relation_map: dict[int, int] = {}
    for src, rel, dst in fact_list:
        _local_slot(entity_map, int(src), num_entities)
        _local_slot(entity_map, int(dst), num_entities)
        _local_slot(relation_map, int(rel), num_relations)
    src_out = torch.zeros(num_facts, dtype=torch.long)
    rel_out = torch.zeros(num_facts, dtype=torch.long)
    dst_out = torch.zeros(num_facts, dtype=torch.long)
    for idx, (src, rel, dst) in enumerate(fact_list[:num_facts]):
        src_out[idx] = entity_map.get(int(src), 0)
        rel_out[idx] = relation_map.get(int(rel), 0)
        dst_out[idx] = entity_map.get(int(dst), 0)
    return GraphFactBank(src=src_out, rel=rel_out, dst=dst_out, relation_map=relation_map, entity_map=entity_map)


def build_query_bank(
    anchor: int,
    query_relation: int,
    facts: Iterable[tuple[int, int, int]],
    cfg: GraphMemoryConfig,
    shuffle_seed: int | None = None,
) -> GraphQueryBank:
    bank = build_fact_bank(
        anchor,
        facts,
        num_facts=cfg.num_facts,
        num_entities=cfg.num_entities,
        num_relations=cfg.num_relations,
        shuffle_seed=shuffle_seed,
    )
    return GraphQueryBank(facts=bank, query_relation=bank.relation_map.get(int(query_relation), cfg.num_relations - 1))


def build_candidate_bank(
    anchor: int,
    facts: Iterable[tuple[int, int, int]],
    cfg: GraphMemoryConfig,
    shuffle_seed: int | None = None,
) -> GraphFactBank:
    return build_fact_bank(
        anchor,
        facts,
        num_facts=cfg.num_cand_facts,
        num_entities=cfg.num_entities,
        num_relations=cfg.num_relations,
        shuffle_seed=shuffle_seed,
    )


def _local_slot(mapping: dict[int, int], key: int, limit: int) -> int:
    if key not in mapping:
        mapping[key] = min(len(mapping), limit - 1)
    return mapping[key]


def _resolve_graph_device(device: str | torch.device) -> torch.device:
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
