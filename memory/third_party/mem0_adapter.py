# memory/mem0_memory.py
from __future__ import annotations

import os
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Union

from loguru import logger
from memory.base import MemorySystem

# ---- Load config from your src/config.py ----
from utils.config import load_config
cfg = load_config()

# ---- Custom embedder ----
try:
    from utils.custom_embedder import CustomProxyEmbedder
except Exception as e:
    raise ImportError("CustomProxyEmbedder not found. Ensure utils/custom_embedder.py exists.") from e

# ---- mem0 client ----
try:
    from mem0 import Memory
except Exception as e:
    raise ImportError("mem0 is not installed. Please `pip install mem0ai`.") from e


def _chroma_path(default_name: str = "mem0_chroma_store") -> str:
    path = os.getenv("MEM0_CHROMA_PATH", f".memdb/{default_name}")
    os.makedirs(path, exist_ok=True)
    return path


class _Mem0EmbedderAdapter:
    """
    Adapter for mem0 → wraps CustomProxyEmbedder so it exposes:
      - .embed(text) and .embed([texts])  (handles extra args/kwargs)
      - .embed_many / .embed_batch
      - .config.embedding_dims (for telemetry)
    """

    def __init__(self, proxy: CustomProxyEmbedder, embedding_dims: int = 3072):
        self._proxy = proxy
        self.config = SimpleNamespace(embedding_dims=embedding_dims)

    def embed(self, *args, **kwargs) -> Union[List[float], List[List[float]]]:
        text_or_texts = None
        if args:
            text_or_texts = args[0]
        else:
            for key in ("text", "texts", "input", "inputs"):
                if key in kwargs:
                    text_or_texts = kwargs[key]
                    break

        if text_or_texts is None:
            raise TypeError("embed() missing required text(s) argument")

        if isinstance(text_or_texts, str):
            return self._proxy.embed(text_or_texts)

        if isinstance(text_or_texts, Sequence):
            return [self._proxy.embed(t) for t in text_or_texts]

        return self._proxy.embed(str(text_or_texts))

    def embed_many(self, texts: Sequence[str], **_: Any) -> List[List[float]]:
        return [self._proxy.embed(t) for t in texts]

    def embed_batch(self, texts: Sequence[str], **_: Any) -> List[List[float]]:
        return [self._proxy.embed(t) for t in texts]


class Mem0Memory(MemorySystem):
    def __init__(
            self,
            name: str = "mem0_memory",
            *,
            collection: Optional[str] = None,
            chat_model: Optional[str] = None,
            embed_model: Optional[str] = None,
            base_url: Optional[str] = None,
            api_key: Optional[str] = None,
            restrict_to_conv: bool = True,
            chroma_path: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.restrict_to_conv = bool(restrict_to_conv)

        # ---- resolve from src.config.load_config(), overridden by kwargs ----
        cfg = load_config()

        self._collection = collection or cfg.get("MB_COLLECTION", "mem0_default")
        self._chat_model = chat_model or cfg.get("MB_CHAT_MODEL", "gemini-flash")
        self._embed_model = embed_model or cfg.get("MB_EMBED_MODEL", "gemini-embedding")
        self._base_url = (base_url or cfg.get("OPENAI_BASE_URL", "")).rstrip("/")
        self._api_key = api_key or cfg.get("OPENAI_API_KEY", "")

        # Ensure OpenAI SDK sees a key even if mem0 ignores passed config
        if self._api_key and not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = self._api_key

        chroma_path = chroma_path or _chroma_path()

        # ---- build mem0 client from explicit config ----
        from mem0 import Memory
        self._client: Memory = Memory.from_config({
            "llm": {
                "provider": "openai",
                "config": {
                    "model": self._chat_model,
                    "api_key": self._api_key,
                    # IMPORTANT: mem0 expects this key name
                    "openai_base_url": self._base_url,
                },
            },
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "path": chroma_path,
                    "collection_name": self._collection,
                },
            },
        })

        # ---- force your CustomProxyEmbedder via adapter ----
        from utils.custom_embedder import CustomProxyEmbedder
        proxy = CustomProxyEmbedder(
            base_url=self._base_url,
            api_key=self._api_key,
            model=self._embed_model,
        )

        # mem0 reads embedding_model.config.embedding_dims (telemetry)
        dims = None
        self._client.embedding_model = _Mem0EmbedderAdapter(proxy, embedding_dims=dims)

        from loguru import logger
        logger.info(
            f"Mem0Memory[{self.name}] ready "
            f"(collection={self._collection}, chat_model={self._chat_model}, "
            f"embed_model={self._embed_model}, path={chroma_path})"
        )

    # ---------------------------------------------------
    # Ingestion
    # ---------------------------------------------------
    def add_turn(self, conv_id: str, turn_id: int, role: str, text: Any) -> None:
        source = f"{conv_id}:turn_{turn_id}"
        if not isinstance(text, str) or not text.strip():
            return
        if role.lower() != "user":
            return

        meta = {"source": source, "role": "user"}
        try:
            self._client.add(text, user_id=conv_id, metadata=meta)  # type: ignore[arg-type]
            logger.info(f"[mem0.add_turn] stored: conv={conv_id} turn={turn_id} text={text!r}")
        except Exception as e:
            logger.warning(f"[mem0.add_turn] failed: {e}; source={source}")

    # ---------------------------------------------------
    # Retrieval
    # ---------------------------------------------------
    def retrieve(self, conv_id: str, query_text: str, top_k: int = 5, **kwargs) -> Dict[str, Any]:
        q = (query_text or "").strip()
        if not q:
            return {"query": "", "results": []}

        limit = max(1, int(top_k))
        try:
            hits_obj = self._client.search(query=q, user_id=conv_id, limit=limit)  # type: ignore[arg-type]
        except TypeError:
            hits_obj = self._client.search(q, user_id=conv_id, k=limit)  # type: ignore[call-arg]

        hits = self._coerce_hits(hits_obj)
        out = [self._to_result_item(h) for h in hits[:limit]]

        # 🔹 Extra logging of facts and memories
        logger.info(f"[mem0.retrieve] query={q!r} got {len(out)} result(s)")
        for i, item in enumerate(out, start=1):
            if item["type"] == "fact":
                logger.info(
                    f"  • FACT[{i}] subj={item['subject']!r}, pred={item['predicate']!r}, "
                    f"obj={item['object']!r}, score={item['score']:.3f}, src={item['source']}"
                )
            else:
                logger.info(
                    f"  • MEMORY[{i}] text={item['text']!r}, score={item['score']:.3f}, src={item['source']}"
                )

        return {"query": q, "results": out}


    def reset(self) -> None:
        try:
            if hasattr(self._client, "clear"):
                self._client.clear(collection=self._collection)  # type: ignore[attr-defined]
        except Exception:
            pass

    def close(self) -> None:
        pass

    @staticmethod
    def _coerce_hits(results_obj: Any) -> List[Dict[str, Any]]:
        if results_obj is None:
            return []
        if isinstance(results_obj, list):
            return [x for x in results_obj if isinstance(x, dict)]
        if isinstance(results_obj, dict):
            for key in ("results", "memories", "matches"):
                if key in results_obj and isinstance(results_obj[key], list):
                    return [x for x in results_obj[key] if isinstance(x, dict)]
        return []

    @staticmethod
    def _to_result_item(hit: Dict[str, Any]) -> Dict[str, Any]:
        hid = hit.get("id", hit.get("_id"))
        score = hit.get("score", hit.get("similarity", hit.get("dist")))
        meta = hit.get("metadata", {}) or {}
        source = meta.get("source") or hit.get("source")
        created_at = hit.get("created_at") or hit.get("ts") or meta.get("created_at")
        subj, pred, obj = hit.get("subject"), hit.get("predicate"), hit.get("object")
        text = (hit.get("text") or hit.get("value") or hit.get("memory") or "").strip()
        return {
            "type": "fact" if (subj and pred and obj) else "memory",
            "id": hid,
            "subject": subj,
            "predicate": pred,
            "object": obj,
            "text": text or meta.get("evidence") or "",
            "score": float(score) if isinstance(score, (int, float)) else 0.0,
            "source": source,
            "created_at": created_at,
        }
