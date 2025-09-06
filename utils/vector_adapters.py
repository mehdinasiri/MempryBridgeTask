# repo/memory/vector_adapters.py
from __future__ import annotations

from typing import Callable, Dict, List, Tuple, Optional

from .models import Chunk, make_chunk

EmbedFn = Callable[[List[str]], List[List[float]]]


# -----------------------------------------
# Minimal protocol-like base for vector indexes
# -----------------------------------------
class VectorIndex:
    def upsert(self, chunks: List[Chunk]) -> None: ...
    def search(self, query: str, k: int) -> List[Tuple[Chunk, float]]: ...
    def clear(self) -> None: ...


# -----------------------------------------
# Chroma adapter (persistent)
# -----------------------------------------
class ChromaIndex(VectorIndex):
    """
    Thin wrapper over ChromaDB.

    Notes
    -----
    - We provide our own `embed_fn` so embeddings stay consistent with the rest of
      the system (OpenAI-compatible, etc.).
    - We push embeddings explicitly on upsert; on query we pass query_embeddings.
    - Similarity is computed as 1 - distance (cosine distance -> cosine sim).
    """

    def __init__(
        self,
        embed_fn: EmbedFn,
        collection_name: str = "memorybridge",
        persist_dir: str = ".memdb/chroma",
    ):
        import chromadb  # pip install chromadb

        self.embed_fn = embed_fn
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=persist_dir)
        # No server-side embedding function — we pass embeddings manually.
        self.col = self.client.get_or_create_collection(name=self.collection_name)

    def upsert(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return
        ids = [c.chunk_id for c in chunks]
        docs = [c.text for c in chunks]
        metas = [{"source": c.source, "created_at": c.created_at, **(c.meta or {})} for c in chunks]
        embs = self.embed_fn(docs)
        # Upsert supports ids, documents, metadatas, embeddings
        self.col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)

    def search(self, query: str, k: int) -> List[Tuple[Chunk, float]]:
        qemb = self.embed_fn([query])  # [[...]]
        res = self.col.query(query_embeddings=qemb, n_results=max(1, k))

        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]

        out: List[Tuple[Chunk, float]] = []
        for _id, _doc, _meta, _dist in zip(ids, docs, metas, dists):
            sim = 1.0 - float(_dist) if _dist is not None else 0.0
            ch = make_chunk(
                _doc,
                source=_meta.get("source", "unknown"),
                chunk_id=_id,
                meta=_meta,
            )
            if _meta.get("created_at"):
                ch.created_at = _meta["created_at"]
            out.append((ch, sim))
        return out

    def clear(self) -> None:
        # Drop & recreate the collection
        try:
            self.client.delete_collection(self.collection_name)
        finally:
            self.col = self.client.get_or_create_collection(self.collection_name)


# -----------------------------------------
# LanceDB adapter (persistent)
# -----------------------------------------
class LanceDBIndex(VectorIndex):
    """
    LanceDB vector table:
      columns: id (str), text (str), source (str), created_at (str), vector (list[float])

    Notes
    -----
    - We compute vectors client-side using `embed_fn` and store them in a column.
    - Search uses cosine by default when available; falls back to backend default.
    """

    def __init__(
        self,
        embed_fn: EmbedFn,
        db_dir: str = ".memdb/lancedb",
        table_name: str = "memorybridge",
    ):
        import lancedb  # pip install lancedb
        from pathlib import Path

        self.embed_fn = embed_fn
        self.table_name = table_name

        Path(db_dir).mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(db_dir)

        existing = {t.name for t in self.db.table_names()} if hasattr(self.db, "table_names") else set()
        if table_name in existing:
            self.tbl = self.db.open_table(table_name)
        else:
            # Create an empty table; schema will adapt on first add
            self.tbl = self.db.create_table(table_name, data=[], mode="create")

    def upsert(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return

        texts = [c.text for c in chunks]
        vecs = self.embed_fn(texts)
        rows: List[Dict] = []
        for c, v in zip(chunks, vecs):
            row = {
                "id": c.chunk_id,
                "text": c.text,
                "source": c.source,
                "created_at": c.created_at,
                "vector": v,
            }
            rows.append(row)

        ids = [r["id"] for r in rows]
        # Best-effort delete then add (simple upsert behavior)
        try:
            # Some versions accept SQL-like predicate; others accept a list.
            self.tbl.delete(f"id in {tuple(ids)}")
        except Exception:
            try:
                for _id in ids:
                    self.tbl.delete(f"id == '{_id}'")
            except Exception:
                pass

        self.tbl.add(rows)

    def search(self, query: str, k: int) -> List[Tuple[Chunk, float]]:
        qv = self.embed_fn([query])[0]
        try:
            res = self.tbl.search(qv).metric("cosine").limit(max(1, k)).to_list()
        except Exception:
            res = self.tbl.search(qv).limit(max(1, k)).to_list()

        out: List[Tuple[Chunk, float]] = []
        for row in res:
            # Different versions expose 'distance' or '_distance'
            dist = row.get("distance", row.get("_distance"))
            sim = 1.0 - float(dist) if dist is not None else 0.0

            ch = make_chunk(
                row["text"],
                source=row.get("source", "unknown"),
                chunk_id=row["id"],
            )
            if row.get("created_at"):
                ch.created_at = row["created_at"]
            out.append((ch, sim))
        return out

    def clear(self) -> None:
        try:
            self.db.drop_table(self.table_name)
        finally:
            self.tbl = self.db.create_table(self.table_name, data=[], mode="create")
