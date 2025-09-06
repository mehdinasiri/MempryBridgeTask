# repo/memory/factory.py
from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Dict, Any

from utils.config import SETTINGS
from memory.base import DefaultEvaluationHooks, JSONLinesFileSink
from memory.vector_memory import VectorMemory
from memory.graph_memory import GraphMemory
from memory.keyword_baseline import KeywordBaseline
from utils.vector_adapters import InMemoryIndex  # type: ignore


# -----------------------------
# Embedding resolution (MB_BACKEND)
# -----------------------------
def _resolve_embed_fn() -> Callable[[List[str]], List[List[float]]]:
    """
    Select an embedding function based on MB_BACKEND.
    Default path uses an OpenAI-compatible endpoint via CustomProxyEmbedder,
    so you can point to Gemini embeddings through your proxy.
    Falls back to a tiny toy embedder if anything fails.
    """
    backend = (SETTINGS.backend or "openai").lower()

    # All backends route through the same OpenAI-compatible proxy by default.
    # If you later add distinct clients (e.g., Vertex, Azure), branch here.
    try:
        from utils.custom_embedder import CustomProxyEmbedder
        proxy = CustomProxyEmbedder(
            base_url=SETTINGS.base_url,
            api_key=SETTINGS.api_key,
            model=SETTINGS.embed_model,
        )

        def _embed(texts: List[str]) -> List[List[float]]:
            vecs = proxy.embed(texts)
            # Normalize to List[List[float]]
            if isinstance(vecs, list) and vecs and isinstance(vecs[0], (int, float)):
                return [vecs]  # single embedding returned
            return vecs  # type: ignore

        return _embed
    except Exception:
        pass

    # Final fallback: toy 26-dim bag-of-letters
    def _toy(texts: List[str]) -> List[List[float]]:
        import math
        out: List[List[float]] = []
        for t in texts:
            v = [0.0] * 26
            for ch in (t or "").lower():
                if "a" <= ch <= "z":
                    v[ord(ch) - 97] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out

    return _toy


# -----------------------------
# Vector store selection (MB_VECTOR_STORE)
# -----------------------------
def _resolve_vector_index(embed_fn):
    """
    Build a vector index adapter based on MB_VECTOR_STORE.
    Supported: chroma, lancedb. Fallback: in-memory.
    """
    store = (SETTINGS.vector_store or "chroma").lower()

    if store == "chroma":
        try:
            from utils.vector_adapters import ChromaIndex
            return ChromaIndex(embed_fn, collection_name=SETTINGS.collection)
        except Exception:
            # Will fall through to in-memory
            pass

    if store == "lancedb":
        try:
            from utils.vector_adapters import LanceDBIndex
            return LanceDBIndex(embed_fn, db_dir=".memdb/lancedb", table_name=SETTINGS.collection)
        except Exception:
            # Will fall through to in-memory
            pass

    # Fallback
    return InMemoryIndex(embed_fn)


# -----------------------------
# Public factories
# -----------------------------
def make_vector_memory(
        *,
        hooks: Optional[DefaultEvaluationHooks] = None,
        index_user_only: bool = True,
        dedup: bool = True,
        recency_half_life: float = 30.0,
        recency_weight: float = 0.10,
        rerank: bool = False,
) -> VectorMemory:
    """
    VectorMemory with embedding + vector store chosen via env:
      - MB_BACKEND controls embedder (CustomProxyEmbedder by default).
      - MB_VECTOR_STORE chooses chroma|lancedb|in-memory.
    """
    embed_fn = _resolve_embed_fn()
    index = _resolve_vector_index(embed_fn)
    hooks = hooks or DefaultEvaluationHooks(system_name="vector_memory")
    return VectorMemory(
        embed_fn=embed_fn,
        index=index,  # <-- adapter plugged in
        index_user_only=index_user_only,
        dedup=dedup,
        recency_half_life=recency_half_life,
        recency_weight=recency_weight,
        rerank=rerank,
        hooks=hooks,
    )


def make_graph_memory(
        *,
        hooks: Optional[DefaultEvaluationHooks] = None,
        db_path: Optional[str] = None,
        index_user_only: bool = True,
        confidence_threshold: float = 0.5,
        recency_half_life_days: float = 30.0,
        weights: Tuple[float, float, float, float] = (0.55, 0.25, 0.10, 0.10),
) -> GraphMemory:
    """
    GraphMemory (Mixed: facts+chunks+graph) with:
      - SQLite facts at DB_PATH (or provided db_path)
      - Embeddings via MB_BACKEND
      - Vector index via MB_VECTOR_STORE
    """
    embed_fn = _resolve_embed_fn()
    index = _resolve_vector_index(embed_fn)
    hooks = hooks or DefaultEvaluationHooks(system_name="graph_memory")
    return GraphMemory(
        db_path=db_path or SETTINGS.db_path,
        embed_fn=embed_fn,
        index=index,  # <-- adapter plugged in
        index_user_only=index_user_only,
        confidence_threshold=confidence_threshold,
        recency_half_life_days=recency_half_life_days,
        weights=weights,
        hooks=hooks,
    )


def make_keyword_baseline(
        *,
        hooks: Optional[DefaultEvaluationHooks] = None,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
) -> KeywordBaseline:
    """BM25 baseline (no vector store needed)."""
    hooks = hooks or DefaultEvaluationHooks(system_name="keyword_baseline")
    return KeywordBaseline(hooks=hooks, bm25_k1=bm25_k1, bm25_b=bm25_b)


def make_mem0_memory(*, hooks: Optional[DefaultEvaluationHooks] = None):
    from memory.third_party.mem0_adapter import Mem0Memory
    hooks = hooks or DefaultEvaluationHooks(system_name="mem0_memory")
    return Mem0Memory(hooks=hooks)


def make_llamaindex_memory(*, hooks: Optional[DefaultEvaluationHooks] = None, embed_model=None):
    from memory.third_party.llamaindex_adapter import LlamaIndexMemory
    hooks = hooks or DefaultEvaluationHooks(system_name="llamaindex_memory")
    return LlamaIndexMemory(hooks=hooks, embed_model=embed_model)


# -----------------------------
# Helper: logging hooks
# -----------------------------
def make_hooks(
        system_name: str,
        log_path: Optional[str] = None,
        base_meta: Optional[Dict[str, Any]] = None,
) -> DefaultEvaluationHooks:
    """
    Convenience for demos/benchmarks to create JSONL logging hooks.
    """
    sink = JSONLinesFileSink(log_path) if log_path else None
    return DefaultEvaluationHooks(system_name=system_name, sink=sink, base_meta=base_meta or {})
