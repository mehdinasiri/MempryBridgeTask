from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from memory.base import MemorySystem, DefaultEvaluationHooks
from utils.models import make_chunk


class LlamaIndexMemory(MemorySystem):
    """
    Minimal adapter for LlamaIndex vector retrieval per conversation.

    Install:
        pip install llama-index-core
    """
    def __init__(
        self,
        name: str = "llamaindex_memory",
        *,
        hooks: Optional[DefaultEvaluationHooks] = None,
        embed_model: Optional[Any] = None,
    ):
        super().__init__(name=name, hooks=hooks)
        try:
            from llama_index.core import Document, VectorStoreIndex  # noqa: F401
            self._Document = Document
            self._VectorStoreIndex = VectorStoreIndex
        except Exception as e:
            raise ImportError(
                "LlamaIndex not installed. Try: pip install llama-index-core"
            ) from e

        self._embed_model = embed_model
        self._docs_by_conv: Dict[str, List[Any]] = {}
        self._index_by_conv: Dict[str, Any] = {}

    # ---------------- internal ----------------

    def _ensure_index(self, conversation_id: str) -> None:
        if conversation_id in self._index_by_conv:
            return
        docs = self._docs_by_conv.get(conversation_id, [])
        if not docs:
            docs = [self._Document(text="(init)")]
        if self._embed_model is not None:
            self._index_by_conv[conversation_id] = self._VectorStoreIndex.from_documents(
                docs, embed_model=self._embed_model
            )
        else:
            self._index_by_conv[conversation_id] = self._VectorStoreIndex.from_documents(docs)

    # ---------------- ingestion ----------------

    def add_turn(
        self,
        conversation_id: str,
        turn_id: int,
        role: str,
        content: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        role = (role or "").strip().lower()
        content = (content or "").strip()
        if not content:
            return

        # mirror your baselines: only index user messages
        if role != "user":
            if self.hooks:
                self.hooks.after_add(conversation_id, turn_id, False, {"reason": "not_user"})
            return

        if self.hooks:
            self.hooks.before_add(conversation_id, turn_id, role, content)

        try:
            Doc = self._Document
            doc = Doc(text=content, metadata={"source": f"{conversation_id}:{turn_id}"})
            self._docs_by_conv.setdefault(conversation_id, []).append(doc)

            if conversation_id in self._index_by_conv:
                self._index_by_conv[conversation_id].insert(doc)
            else:
                self._ensure_index(conversation_id)

            if self.hooks:
                self.hooks.after_add(conversation_id, turn_id, True, None)
        except Exception as e:
            if self.hooks:
                self.hooks.after_add(conversation_id, turn_id, False, {"exception": str(e)})

    # ---------------- retrieval ----------------

    def retrieve(
        self,
        conversation_id: str,
        query: str,
        top_k: int = 5,
        subject_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._ensure_index(conversation_id)

        if self.hooks:
            self.hooks.before_retrieve(conversation_id, None, query, top_k)

        t0 = time.perf_counter()
        try:
            retriever = self._index_by_conv[conversation_id].as_retriever(similarity_top_k=max(1, top_k))
            nodes = retriever.retrieve(query)

            items: List[Dict[str, Any]] = []
            for nd in nodes:
                text = getattr(nd, "text", "") or getattr(getattr(nd, "node", None), "text", "") or ""
                score = float(getattr(nd, "score", 0.0) or 0.0)
                meta = {}
                try:
                    meta = dict(getattr(getattr(nd, "node", None), "metadata", {}) or {})
                except Exception:
                    pass
                src = meta.get("source", conversation_id)
                ch = make_chunk(text=text, source=src)
                items.append(
                    {
                        "type": "chunk",
                        "id": ch.chunk_id,
                        "text": text,
                        "source": src,
                        "created_at": ch.created_at,
                        "score": score,
                        "scores": {"llamaindex": score},
                    }
                )

            latency_ms = (time.perf_counter() - t0) * 1000.0
            if self.hooks:
                self.hooks.after_retrieve(conversation_id, None, items, latency_ms, query)
            return {"results": items, "latency_ms": latency_ms}
        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            if self.hooks:
                self.hooks.after_retrieve(conversation_id, None, [], latency_ms, query)
            return {"results": [], "latency_ms": latency_ms}
