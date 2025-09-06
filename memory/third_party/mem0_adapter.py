from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from memory.base import MemorySystem, DefaultEvaluationHooks
from utils.models import make_chunk

# Settings & custom embedder
try:
    from utils.config import SETTINGS
except Exception:
    class _S:
        chat_model = os.getenv("MB_CHAT_MODEL", "gemini-flash")
        base_url = os.getenv("OPENAI_BASE_URL", "")
        api_key = os.getenv("OPENAI_API_KEY", "")
        emb_model = os.getenv("MB_EMBED_MODEL", "gemini-embedding")
        collection = os.getenv("MB_COLLECTION", "mem0_default")


    SETTINGS = _S()  # type: ignore

try:
    from utils.custom_embedder import CustomProxyEmbedder
except Exception as e:
    raise ImportError("CustomProxyEmbedder not found. Ensure utils/custom_embedder.py exists.") from e

# mem0 import kept local so the rest of the repo doesn't hard-require it
try:
    from mem0 import Memory
except Exception as e:
    raise ImportError("mem0ai not installed. Run: pip install mem0ai") from e


class Mem0Memory(MemorySystem):
    """
    Minimal, reliable Mem0 adapter:
      - Uses Memory.from_config with OpenAI-compatible Gemini gateway
      - Forces embedding model via CustomProxyEmbedder (prevents text-embedding-3-small)
      - Emits hooks with your positional signatures
      - Normalizes retrieval to your 'chunk' schema
    """

    def __init__(
            self,
            name: str = "mem0_memory",
            *,
            hooks: Optional[DefaultEvaluationHooks] = None,
            chroma_path: Optional[str] = None,
    ):
        super().__init__(name=name, hooks=hooks)

        s = SETTINGS  # convenience
        chroma_path = chroma_path or ".mem0/chroma"

        # 1) Build Mem0 with your gateway + chat model
        #    NOTE: mem0 expects 'openai_base_url' for the OpenAI provider config.
        self._client: Memory = Memory.from_config({
            "llm": {
                "provider": "openai",
                "config": {
                    "model": s.chat_model,  # e.g., "gemini-flash"
                    "api_key": s.api_key,
                    "openai_base_url": s.base_url,  # IMPORTANT key name
                },
            },
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "path": chroma_path,
                    "collection_name": getattr(s, "collection", "mem0_default"),
                },
            },
        })

        # 2) Force the embedding model to your Gemini embedding via the custom proxy.
        #    (This prevents mem0 from using text-embedding-3-small by default.)
        #    Only pass embedding_dims if present in SETTINGS.
        emb_kwargs: Dict[str, Any] = {
            "base_url": s.base_url,
            "api_key": s.api_key,
            "model": getattr(s, "emb_model", "gemini-embedding"),
        }

        self._client.embedding_model = CustomProxyEmbedder(**emb_kwargs)

        # Keep a short sliding window of turns per conversation
        self._turns: Dict[str, List[Dict[str, str]]] = {}

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

        self._turns.setdefault(conversation_id, []).append({"role": role, "content": content})
        window = self._turns[conversation_id][-6:]  # small sliding context

        if self.hooks:
            # before_add(conv_id, turn_id, role, text)
            self.hooks.before_add(conversation_id, turn_id, role, content)

        try:
            # Let mem0 infer/update memories from this short window
            self._client.add(window, user_id=conversation_id, metadata={"source": f"{conversation_id}:{turn_id}"})
            if self.hooks:
                # after_add(conv_id, turn_id, indexed, added_summary, info_dict)
                self.hooks.after_add(conversation_id, turn_id, True, content[:80], {"window_size": len(window)})
        except Exception as e:
            if self.hooks:
                self.hooks.after_add(conversation_id, turn_id, False, "", {"exception": str(e)})

    # ---------------- retrieval ----------------

    def retrieve(
            self,
            conversation_id: str,
            query: str,
            top_k: int = 5,
            subject_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        # correct hook signature: (conv_id, turn_id_or_none, query_text, top_k)
        if self.hooks:
            # meta dict expected by DefaultEvaluationHooks
            self.hooks.before_retrieve(
                conversation_id, None, {"query_preview": (query or "")[:200], "top_k": int(top_k)}
            )

        t0 = time.perf_counter()
        try:
            try:
                raw = self._client.search(query=query, user_id=conversation_id, limit=top_k)
            except TypeError:
                raw = self._client.search(query=query, user_id=conversation_id, top_k=top_k)

            results = raw or []
            items: List[Dict[str, Any]] = []
            for m in results[: max(1, top_k)]:
                text = (m.get("memory") or m.get("data") or m.get("content") or m.get("text") or "")
                score = float(m.get("score") or m.get("similarity") or m.get("confidence") or 0.0)
                src = (m.get("metadata") or {}).get("source", conversation_id)
                ch = make_chunk(text=text, source=str(src))
                items.append({
                    "type": "chunk",
                    "id": m.get("id") or m.get("_id") or m.get("uuid") or ch.chunk_id,
                    "text": text,
                    "source": src,
                    "created_at": ch.created_at,
                    "score": score,
                    "scores": {"mem0": score},
                })

            latency_ms = (time.perf_counter() - t0) * 1000.0

            if self.hooks:
                self.hooks.after_retrieve(
                    conversation_id, None,
                    {"top_k": int(top_k), "latency_ms": latency_ms, "query": query, "results": items}
                )

            return {"results": items, "latency_ms": latency_ms}

        except Exception:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            if self.hooks:
                self.hooks.after_retrieve(
                    conversation_id, None,
                    {"top_k": int(top_k), "latency_ms": latency_ms, "query": query, "results": []}
                )
            return {"results": [], "latency_ms": latency_ms}