# memory/llamaindex_adapter.py
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Union

from loguru import logger
from memory.base import MemorySystem

# ---- Config ----
from utils.config import load_config
_cfg = load_config()

# ---- Your custom embedder (vector-only) ----
try:
    from utils.custom_embedder import CustomProxyEmbedder
except Exception as e:
    raise ImportError("CustomProxyEmbedder not found. Ensure utils/custom_embedder.py exists.") from e

# ---- LlamaIndex & Chroma bindings (import defensively across versions) ----
try:
    # Core
    from llama_index.core import Document
except Exception as e:
    raise ImportError(
        "llama_index is required. `pip install llama-index chromadb`"
    ) from e

# Settings / ServiceContext (v0.9 vs v0.10+ compatibility)
_LI_Settings = None
_LI_ServiceContext = None
try:
    from llama_index.core import Settings as _LI_Settings  # v0.10+
except Exception:
    try:
        from llama_index.core import ServiceContext as _LI_ServiceContext  # v0.9
    except Exception:
        _LI_ServiceContext = None

# LLM (OpenAI-compatible; supports base_url + key)
_LI_OpenAI = None
try:
    from llama_index.llms.openai import OpenAI as _LI_OpenAI  # v0.10+
except Exception:
    try:
        from llama_index.llms.openai import OpenAI as _LI_OpenAI  # older
    except Exception:
        _LI_OpenAI = None

# Embedding base
try:
    from llama_index.core.embeddings import BaseEmbedding  # v0.10+
except Exception:
    from llama_index.embeddings import BaseEmbedding       # v0.9 fallback


# Vector store (Chroma)
try:
    import chromadb
    from llama_index.vector_stores.chroma import ChromaVectorStore
except Exception as e:
    raise ImportError(
        "ChromaVectorStore requires `chromadb`. Install: pip install chromadb llama-index-vector-stores-chroma"
    ) from e

# Index / Node APIs
_VectorStoreIndex = None
try:
    from llama_index.core import VectorStoreIndex as _VectorStoreIndex
except Exception:
    try:
        from llama_index import VectorStoreIndex as _VectorStoreIndex  # very old
    except Exception:
        _VectorStoreIndex = None

# Storage context (for injecting vector store)
try:
    from llama_index.core import StorageContext as _StorageContext
except Exception:
    try:
        from llama_index import StorageContext as _StorageContext
    except Exception:
        _StorageContext = None


def _chroma_path(default_name: str = "llamaindex_chroma_store") -> str:
    path = os.getenv("LLAMAINDEX_CHROMA_PATH", f".memdb/{default_name}")
    os.makedirs(path, exist_ok=True)
    return path


@dataclass
class _RetrievalItem:
    type: str
    id: Optional[str]
    subject: Optional[str]
    predicate: Optional[str]
    object: Optional[str]
    text: str
    score: float
    source: Optional[str]
    created_at: Optional[str]

class _LlamaIndexEmbedderAdapter(BaseEmbedding):
    def __init__(self, proxy: CustomProxyEmbedder):
        super().__init__()
        self._proxy = proxy
        # keep for optional internal logging if you want
        self._embedding_dims = int(os.getenv("MB_EMBED_DIMS", "3072"))

    # required sync hooks
    def _get_query_embedding(self, query: str) -> List[float]:
        return self._proxy.embed(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._proxy.embed(text)

    def _get_text_embeddings(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._proxy.embed(t) for t in texts]

    # optional async hooks (safe fallbacks)
    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    async def _aget_text_embeddings(self, texts: Sequence[str]) -> List[List[float]]:
        return self._get_text_embeddings(texts)



class LlamaIndexMemory(MemorySystem):
    """
    MemorySystem implemented on top of LlamaIndex + Chroma.

    - Stores only USER turns (like your other memories)
    - Uses CustomProxyEmbedder via an adapter class
    - Chroma persists embeddings at a local path
    - Retrieval returns normalized items similar to Vector/Graph/Mem0 adapters
    - Logs added turns and retrieval results (facts vs memories)
    """

    def __init__(
        self,
        name: str = "llamaindex_memory",
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

        # Resolve config (kwargs override env/config)
        self._collection = collection or _cfg.get("MB_COLLECTION", "mem0_default")
        self._chat_model = chat_model or _cfg.get("MB_CHAT_MODEL", "gemini-flash")
        self._embed_model = embed_model or _cfg.get("MB_EMBED_MODEL", "gemini-embedding")
        self._base_url = (base_url or _cfg.get("OPENAI_BASE_URL", "")).rstrip("/")
        self._api_key = api_key or _cfg.get("OPENAI_API_KEY", "")

        # Ensure OpenAI-compatible clients see a key (many clients still read env)
        if self._api_key and not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = self._api_key

        self._persist_path = chroma_path or _chroma_path()

        # --- Vector store (Chroma) ---
        self._chroma_client = chromadb.PersistentClient(path=self._persist_path)
        self._chroma_collection = self._chroma_client.get_or_create_collection(self._collection)
        self._vector_store = ChromaVectorStore(chroma_collection=self._chroma_collection)

        # --- Embeddings via your proxy embedder ---
        proxy = CustomProxyEmbedder(
            base_url=self._base_url,
            api_key=self._api_key,
            model=self._embed_model,
        )
        self._embedder = _LlamaIndexEmbedderAdapter(proxy)

        # --- LLM (optional) ---
        llm = None
        if _LI_OpenAI is not None:
            llm = _LI_OpenAI(model=self._chat_model, api_key=self._api_key or None, base_url=self._base_url or None)

        # --- Settings/ServiceContext wiring ---
        if _LI_Settings is not None:
            # v0.10+: global settings require BaseEmbedding instance (we have it now)
            _LI_Settings.embed_model = self._embedder
            if llm is not None:
                _LI_Settings.llm = llm
        elif _LI_ServiceContext is not None:
            # v0.9: build a ServiceContext
            from llama_index.core import ServiceContext
            self._service_context = ServiceContext.from_defaults(embed_model=self._embedder, llm=llm)
        else:
            self._service_context = None

        # --- Index on top of Chroma vector store ---
        if _StorageContext is None or _VectorStoreIndex is None:
            raise ImportError("Your LlamaIndex version is too old; please upgrade to >=0.9.")

        storage_ctx = _StorageContext.from_defaults(vector_store=self._vector_store)
        if _LI_ServiceContext is not None:
            self._index = _VectorStoreIndex.from_documents(
                documents=[], storage_context=storage_ctx, service_context=self._service_context
            )
        else:
            # v0.10 uses Settings; service_context arg is deprecated
            self._index = _VectorStoreIndex.from_documents(documents=[], storage_context=storage_ctx)

        logger.info(
            f"LlamaIndexMemory[{self.name}] ready "
            f"(collection={self._collection}, chat_model={self._chat_model}, "
            f"embed_model={self._embed_model}, path={self._persist_path})"
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

        # Create a document per user message; include key metadata
        doc = Document(text=text, metadata={"source": source, "role": "user", "conv_id": conv_id})
        try:
            # Insert into vector index (this will call our embedder)
            self._index.insert(doc)
            logger.info(f"[li.add_turn] stored: conv={conv_id} turn={turn_id} text={text!r}")
        except Exception as e:
            logger.warning(f"[li.add_turn] failed: {e}; source={source}")

    # ---------------------------------------------------
    # Retrieval
    # ---------------------------------------------------
    def retrieve(self, conv_id: str, query_text: str, top_k: int = 5, **kwargs) -> Dict[str, Any]:
        q = (query_text or "").strip()
        if not q:
            return {"query": "", "results": []}

        # If you want per-conversation scoping: filter nodes by conv_id at query time.
        # We’ll apply a simple metadata filter when building retriever (if supported).
        limit = max(1, int(top_k))

        try:
            # v0.10 retriever
            retriever = self._index.as_retriever(similarity_top_k=limit)
            # Some versions support metadata filtering via retriever; else we post-filter
            results = retriever.retrieve(q)
        except TypeError:
            # older API shape
            retriever = self._index.as_retriever(top_k=limit)
            results = retriever.retrieve(q)

        # results → list of NodeWithScore
        items: List[_RetrievalItem] = []
        for r in results:
            node = getattr(r, "node", None) or getattr(r, "node_with_score", None) or r
            score = float(getattr(r, "score", 0.0) or 0.0)

            meta = {}
            try:
                meta = dict(node.metadata or {})
            except Exception:
                pass

            # Per-conversation scoping: drop items with mismatched conv_id
            if self.restrict_to_conv and meta.get("conv_id") and meta.get("conv_id") != conv_id:
                continue

            text = ""
            try:
                text = (node.get_content() or "").strip()
            except Exception:
                # older versions
                text = (getattr(node, "text", "") or "").strip()

            item = _RetrievalItem(
                type="memory",               # LlamaIndex path stores free text; you can add your own fact extraction if needed
                id=str(getattr(node, "node_id", None) or getattr(node, "id_", None) or ""),
                subject=None,
                predicate=None,
                object=None,
                text=text,
                score=score,
                source=meta.get("source"),
                created_at=meta.get("created_at"),
            )
            items.append(item)

        # Logging like your mem0 adapter (facts vs memories)
        logger.info(f"[li.retrieve] query={q!r} got {len(items)} result(s)")
        for i, it in enumerate(items, start=1):
            logger.info(f"  • MEMORY[{i}] text={it.text!r}, score={it.score:.3f}, src={it.source}")

        return {
            "query": q,
            "results": [self._to_result_dict(it) for it in items[:limit]],
        }

    # ---------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------
    def reset(self) -> None:
        """Delete the Chroma collection (clears all stored memories for this adapter)."""
        try:
            # chroma client: delete + recreate collection
            self._chroma_client.delete_collection(self._collection)
            self._chroma_collection = self._chroma_client.get_or_create_collection(self._collection)
            self._vector_store = ChromaVectorStore(chroma_collection=self._chroma_collection)

            # rebuild empty index bound to new collection
            storage_ctx = _StorageContext.from_defaults(vector_store=self._vector_store)
            if _LI_ServiceContext is not None:
                self._index = _VectorStoreIndex.from_documents(documents=[], storage_context=storage_ctx, service_context=getattr(self, "_service_context", None))
            else:
                self._index = _VectorStoreIndex.from_documents(documents=[], storage_context=storage_ctx)
            logger.info("[li.reset] collection cleared and index rebuilt")
        except Exception as e:
            logger.warning(f"[li.reset] failed: {e}")

    def close(self) -> None:
        """Nothing to close explicitly; Chroma client persists on disk."""
        pass

    # ---------------------------------------------------
    # Helpers
    # ---------------------------------------------------
    @staticmethod
    def _to_result_dict(it: _RetrievalItem) -> Dict[str, Any]:
        return {
            "type": it.type,
            "id": it.id,
            "subject": it.subject,
            "predicate": it.predicate,
            "object": it.object,
            "text": it.text,
            "score": it.score,
            "source": it.source,
            "created_at": it.created_at,
        }
