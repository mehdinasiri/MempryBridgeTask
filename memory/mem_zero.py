from __future__ import annotations
from typing import Any, Dict, List
from datetime import datetime

from mem0 import Memory

from .base import AbstractMemory
from .config import SETTINGS
from .custom_embedder import CustomProxyEmbedder
from .reranker import rerank_with_llm


class MemZeroMemory(AbstractMemory):
    """Single-file mem0 backend: connect → add → search → optional rerank."""

    def __init__(self):
        self.memory: Memory | None = None

    # ------- connection -----------------------------------------------------
    def connect(self, **kwargs) -> None:
        s = SETTINGS.require()
        self.memory = Memory.from_config({
            "llm": {
                "provider": "openai",
                "config": {
                    "model": s.chat_model,
                    "api_key": s.api_key,
                    "openai_base_url": s.base_url,
                },
            },
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "path": ".mem0/chroma",
                    "collection_name": s.collection,
                },
            },
        })
        self.memory.embedding_model = CustomProxyEmbedder(
            base_url=s.base_url,
            api_key=s.api_key,
            model=s.emb_model,
            embedding_dims=s.emb_dims,
        )

    # ------- extract/store --------------------------------------------------
    def _add_turn(self, text: str, conv_id: str, turn_id: str, user_id: str):
        assert self.memory is not None
        meta = {"conv_id": conv_id, "turn_id": turn_id, "timestamp": datetime.utcnow().isoformat()}
        return self.memory.add(text, user_id=user_id, metadata=meta)

    def _add_messages(self, messages: List[Dict[str, str]], conv_id: str, user_id: str):
        assert self.memory is not None
        meta = {"conv_id": conv_id, "timestamp": datetime.utcnow().isoformat()}
        return self.memory.add(messages, user_id=user_id, metadata=meta)

    # ------- retrieve -------------------------------------------------------
    def _unwrap_hits(self, raw):
        if isinstance(raw, dict):
            for k in ("results", "data", "items"):
                if k in raw and isinstance(raw[k], list):
                    return raw[k]
            return []
        return raw or []

    def _text_from_hit(self, h: Dict[str, Any]) -> str:
        return h.get("memory") or h.get("data") or h.get("content") or h.get("text") or h.get("value") or ""

    def _retrieve(self, user_id: str, query: str, k: int = 5) -> List[Dict[str, Any]]:
        assert self.memory is not None
        try:
            raw = self.memory.search(query=query, user_id=user_id, limit=k)
        except TypeError:
            raw = self.memory.search(query=query, user_id=user_id, top_k=k)
        hits = self._unwrap_hits(raw)
        out: List[Dict[str, Any]] = []
        for h in hits:
            if isinstance(h, dict):
                out.append({
                    "fact_id": h.get("id") or h.get("_id") or h.get("uuid"),
                    "content": self._text_from_hit(h),
                    "score": h.get("score") or h.get("similarity") or h.get("confidence"),
                    "metadata": h.get("metadata") or {},
                    "created_at": h.get("created_at"),
                    "updated_at": h.get("updated_at"),
                    "raw": h,
                })
            else:
                out.append({"fact_id": None, "content": str(h), "score": None, "metadata": {}, "created_at": None, "updated_at": None, "raw": h})
        out.sort(key=lambda x: (x["score"] is None, -(x["score"] or 0.0)))
        return out[:k]

    # ------- public API -----------------------------------------------------
    def add_turn(self, *, text: str, conv_id: str, turn_id: str, user_id: str) -> Any:
        return self._add_turn(text, conv_id, turn_id, user_id)

    def add_conversation(self, *, messages: List[Dict[str, str]], conv_id: str, user_id: str) -> Any:
        return self._add_messages(messages, conv_id, user_id)

    def retrieve(self, *, query: str, user_id: str, k: int = 5) -> List[Dict[str, Any]]:
        return self._retrieve(user_id=user_id, query=query, k=k)

    def retrieve_reranked(self, *, query: str, user_id: str, k: int = 5, top_n: int = 3) -> List[Dict[str, Any]]:
        hits = self.retrieve(query=query, user_id=user_id, k=k)
        return rerank_with_llm(query, hits, top_n=top_n)