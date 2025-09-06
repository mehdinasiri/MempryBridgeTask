# repo/memory/graph_memory.py
from __future__ import annotations

import json
import math
import re
import sqlite3
import time
from utils.vector_adapters import InMemoryIndex, VectorIndex

from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .base import MemorySystem
from utils.models import (
    AtomicFact,
    Chunk,
    RetrievalItem,
    RetrievalResult,
    cosine,
    make_chunk,
    make_fact,
    norm_text,
    norm_entity,
    now_iso,
)

# -------------------------------------------------------------
# Embedding: use your CustomProxyEmbedder by default if present
# -------------------------------------------------------------
try:
    from utils.custom_embedder import CustomProxyEmbedder
    _DEFAULT_EMBED = CustomProxyEmbedder().embed  # accepts str|List[str], returns vec or list[vec]
except Exception:
    _DEFAULT_EMBED = None


def _toy_embed(texts: List[str]) -> List[List[float]]:
    """ 26-dim character bag-of-letters embedder (fallback). """
    def emb(t: str) -> List[float]:
        v = [0.0] * 26
        for ch in t.lower():
            if "a" <= ch <= "z":
                v[ord(ch) - 97] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]
    return [emb(t) for t in texts]


def _normalize_embed_fn(embed_fn: Optional[Callable[[List[str]], List[List[float]]]]):
    """Wraps CustomProxyEmbedder to always return List[List[float]]."""
    if embed_fn is not None:
        return embed_fn
    if _DEFAULT_EMBED is None:
        return _toy_embed

    def _wrap(texts: List[str]) -> List[List[float]]:
        vecs = _DEFAULT_EMBED(texts)  # type: ignore
        # If single vector returned (for a single str path), normalize
        if isinstance(vecs, list) and vecs and isinstance(vecs[0], (int, float)):
            return [vecs]  # type: ignore
        return vecs  # type: ignore

    return _wrap


# -------------------------------------------------------------
# Simple in-memory vector index (dependency-free)
# -------------------------------------------------------------
class _VecIndex:
    def __init__(self, embed_fn: Callable[[List[str]], List[List[float]]]):
        self.embed_fn = embed_fn
        self._vecs: Dict[str, List[float]] = {}   # chunk_id -> embedding
        self._chunks: Dict[str, Chunk] = {}       # chunk_id -> chunk

    def upsert(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return
        embs = self.embed_fn([c.text for c in chunks])
        for c, v in zip(chunks, embs):
            self._chunks[c.chunk_id] = c
            self._vecs[c.chunk_id] = v

    def search(self, query: str, k: int) -> List[Tuple[Chunk, float]]:
        qv = self.embed_fn([query])[0]
        scored: List[Tuple[float, str]] = []
        for cid, v in self._vecs.items():
            s = cosine(qv, v)
            scored.append((s, cid))
        scored.sort(key=lambda x: x[0], reverse=True)
        out: List[Tuple[Chunk, float]] = []
        for s, cid in scored[:max(1, k)]:
            out.append((self._chunks[cid], float(s)))
        return out

    def clear(self) -> None:
        self._vecs.clear()
        self._chunks.clear()


# -------------------------------------------------------------
# Fact store (SQLite) with supersede semantics
# -------------------------------------------------------------
class FactStoreSQLite:
    """
    Persistent store for facts. 'Active' denotes the latest fact for a slot (subject, predicate).
    Schema is compact and auditable.
    """
    def __init__(self, db_path: str = ".memdb/memory.sqlite"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._init()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init(self):
        with self._conn() as c:
            c.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                fact_id TEXT PRIMARY KEY,
                subject TEXT,
                predicate TEXT,
                object TEXT,
                source TEXT,
                confidence REAL,
                created_at TEXT,
                supersedes TEXT,
                active INTEGER,
                meta TEXT
            )
            """)
            c.execute("CREATE INDEX IF NOT EXISTS idx_slot ON facts(subject, predicate)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_active ON facts(active)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_created ON facts(created_at)")

    def add(self, f: AtomicFact) -> None:
        with self._conn() as c:
            c.execute("""
            INSERT OR REPLACE INTO facts VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (
                f.fact_id, f.subject, f.predicate, f.object, f.source,
                f.confidence, f.created_at, f.supersedes, 1 if f.active else 0,
                json.dumps(f.meta, ensure_ascii=False)
            ))

    def get_active(self, subject: str, predicate: str) -> Optional[AtomicFact]:
        with self._conn() as c:
            row = c.execute("""
            SELECT * FROM facts WHERE lower(subject)=lower(?) AND lower(predicate)=lower(?) AND active=1
            ORDER BY datetime(created_at) DESC LIMIT 1
            """, (subject, predicate)).fetchone()
        return self._row_to_fact(row) if row else None

    def supersede_slot(self, subject: str, predicate: str) -> Optional[AtomicFact]:
        prev = self.get_active(subject, predicate)
        if not prev:
            return None
        with self._conn() as c:
            c.execute("UPDATE facts SET active=0 WHERE fact_id=?", (prev.fact_id,))
        return prev

    def list_recent(self, limit: int = 100) -> List[AtomicFact]:
        with self._conn() as c:
            rows = c.execute("""
            SELECT * FROM facts WHERE active=1 ORDER BY datetime(created_at) DESC LIMIT ?
            """, (limit,)).fetchall()
        return [self._row_to_fact(r) for r in rows]

    def search_subject(self, subject: str, limit: int = 50) -> List[AtomicFact]:
        with self._conn() as c:
            rows = c.execute("""
            SELECT * FROM facts WHERE active=1 AND lower(subject)=lower(?)
            ORDER BY datetime(created_at) DESC LIMIT ?
            """, (subject, limit)).fetchall()
        return [self._row_to_fact(r) for r in rows]

    @staticmethod
    def _row_to_fact(r) -> AtomicFact:
        return AtomicFact(
            fact_id=r[0], subject=r[1], predicate=r[2], object=r[3], source=r[4],
            confidence=r[5], created_at=r[6], supersedes=r[7], active=bool(r[8]),
            meta=json.loads(r[9]) if r[9] else {}
        )


# -------------------------------------------------------------
# Graph overlay
# -------------------------------------------------------------
class GraphStore:
    """
    Lightweight directed multi-graph of triples: (subject)-[predicate]->(object).
    Stores a summary payload: {fact_ids, confidence(max), created_at(max)}.
    """
    def __init__(self):
        self.nodes: set[str] = set()
        self.edges: Dict[Tuple[str, str, str], Dict[str, object]] = {}

    @staticmethod
    def _n(x: str) -> str:
        return norm_entity(x)

    def upsert(self, f: AtomicFact) -> None:
        u, v, p = self._n(f.subject), self._n(f.object), self._n(f.predicate)
        self.nodes.add(u); self.nodes.add(v)
        key = (u, v, p)
        payload = self.edges.get(key, {"fact_ids": set(), "confidence": 0.0, "created_at": f.created_at})
        payload["fact_ids"].add(f.fact_id)  # type: ignore
        payload["confidence"] = max(float(payload["confidence"]), float(f.confidence))  # type: ignore
        # keep most recent timestamp
        if f.created_at > str(payload["created_at"]):
            payload["created_at"] = f.created_at
        self.edges[key] = payload

    def remove(self, f: AtomicFact) -> None:
        u, v, p = self._n(f.subject), self._n(f.object), self._n(f.predicate)
        key = (u, v, p)
        if key not in self.edges:
            return
        payload = self.edges[key]
        if f.fact_id in payload["fact_ids"]:  # type: ignore
            payload["fact_ids"].remove(f.fact_id)  # type: ignore
        if not payload["fact_ids"]:  # type: ignore
            self.edges.pop(key, None)

    def neighbors(self, node: str, direction: str = "both") -> List[Tuple[str, str, str, Dict[str, object]]]:
        q = self._n(node)
        out: List[Tuple[str, str, str, Dict[str, object]]] = []
        for (u, v, p), data in self.edges.items():
            if direction == "out" and u == q:
                out.append((u, v, p, data))
            elif direction == "in" and v == q:
                out.append((u, v, p, data))
            elif direction == "both" and (u == q or v == q):
                out.append((u, v, p, data))
        return out

    def k_hop_expand(self, seeds: List[str], k: int = 2, max_edges: int = 64) -> List[Tuple[str, str, str, Dict[str, object]]]:
        seen_nodes = {self._n(s) for s in seeds if s}
        frontier = list(seen_nodes)
        seen_edges: set[Tuple[str, str, str]] = set()
        out: List[Tuple[str, str, str, Dict[str, object]]] = []
        for _ in range(max(1, k)):
            next_frontier: List[str] = []
            for n in frontier:
                for (u, v, p, data) in self.neighbors(n, direction="both"):
                    ek = (u, v, p)
                    if ek in seen_edges:
                        continue
                    seen_edges.add(ek)
                    out.append((u, v, p, data))
                    if len(out) >= max_edges:
                        return out
                    if v not in seen_nodes:
                        seen_nodes.add(v); next_frontier.append(v)
                    if u not in seen_nodes:
                        seen_nodes.add(u); next_frontier.append(u)
            frontier = next_frontier
            if not frontier:
                break
        return out


# -------------------------------------------------------------
# Simple rule-based extractor (swap with LLM later)
# -------------------------------------------------------------
class SimpleExtractor:
    """
    Heuristic extractor for demo/bench:
      - "X works at Y"    -> (X, works_at, Y)
      - "I live in Z"     -> (speaker, lives_in, Z)
      - "My name is N"    -> (speaker, name_is, N)
    Also emits a chunk for every user/assistant turn (we'll index user-only by default).
    """

    WORKS_AT = re.compile(r"(?i)\b([A-Z][\w .&-]{0,60})\s+works\s+at\s+([A-Z][\w .&-]{0,60})")
    LIVE_IN  = re.compile(r"(?i)\b(?:i|we)\s+live(?:s)?\s+in\s+([A-Z][\w .&-]{0,60})")
    NAME_IS  = re.compile(r"(?i)\bmy\s+name\s+is\s+([A-Z][\w .&-]{0,60})")

    def __init__(self, speaker_alias: str = "user"):
        self.speaker_alias = speaker_alias

    def extract(self, conv_id: str, turn_id: int, role: str, text: str) -> Tuple[List[AtomicFact], List[Chunk]]:
        facts: List[AtomicFact] = []
        chunks: List[Chunk] = []

        t = norm_text(text)
        src = f"{conv_id}:{turn_id}"

        # 1) chunks: always emit (vector recall)
        chunks.append(make_chunk(t, source=src))

        # 2) facts: only from user role
        role_l = (role or "").lower()
        if role_l == "user":
            # A) "X works at Y"
            for m in self.WORKS_AT.finditer(text):
                subj = norm_text(m.group(1))
                obj  = norm_text(m.group(2))
                facts.append(make_fact(subj, "works_at", obj, source=src, confidence=0.9))

            # B) "I live in Z"
            for m in self.LIVE_IN.finditer(text):
                obj = norm_text(m.group(1))
                facts.append(make_fact(self.speaker_alias, "lives_in", obj, source=src, confidence=0.85))

            # C) "My name is N"
            for m in self.NAME_IS.finditer(text):
                obj = norm_text(m.group(1))
                facts.append(make_fact(self.speaker_alias, "name_is", obj, source=src, confidence=0.8))

        return facts, chunks


# -------------------------------------------------------------
# GraphMemory (Mixed: facts + chunks + graph)
# -------------------------------------------------------------
class GraphMemory(MemorySystem):
    """
    Mixed Memory with:
      - Atomic facts in SQLite (supersede on (subject, predicate))
      - Semantic chunks in a vector index
      - Graph overlay for multi-hop retrieval
      - Hybrid ranking with semantic/confidence/recency/graph contributions

    Defaults:
      - index_user_only=True  (facts always user-only; chunks user-only if True)
      - confidence_threshold=0.5
      - recency_half_life_days=30.0
      - weights: alpha=0.55 (semantic), beta=0.25 (fact confidence/kw), gamma=0.10 (recency), delta=0.10 (graph)
    """

    def __init__(self, name: str = "graph_memory", db_path: str = ".memdb/memory.sqlite",
                 embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
                 extractor: Optional[SimpleExtractor] = None, *,
                 index_user_only: bool = True, confidence_threshold: float = 0.5,
                 recency_half_life_days: float = 30.0,
                 weights: Tuple[float, float, float, float] = (0.55, 0.25, 0.10, 0.10),
                 hooks=None, index: Optional[VectorIndex] = None):  # <-- NEW
        super().__init__(name=name, hooks=hooks)

        self.facts = FactStoreSQLite(db_path=db_path)
        self.graph = GraphStore()
        self.embed_fn = _normalize_embed_fn(embed_fn)
        self.vec: VectorIndex = index or InMemoryIndex(self.embed_fn)
        self.ext = extractor or SimpleExtractor()

        # Per-conv state for recency scoring on chunks
        self._docs: Dict[str, List[Chunk]] = {}
        self._seen_texts: Dict[str, set[int]] = {}

        # Config
        self.index_user_only = bool(index_user_only)
        self.conf_thr = float(confidence_threshold)
        self.half_life = float(recency_half_life_days)
        self.alpha, self.beta, self.gamma, self.delta = [float(x) for x in weights]

    # ---------------------------
    # Ingestion
    # ---------------------------
    def add_turn(self, conv_id: str, turn_id: int, role: str, text: str) -> None:
        self.before_add(conv_id, turn_id, role, text)

        # Extract
        facts, chunks = self.ext.extract(conv_id, turn_id, role, text)

        # Respect index_user_only for chunks
        role_l = (role or "").lower()
        if self.index_user_only and role_l != "user":
            chunks = []  # still allow facts from user only (already enforced in extractor)

        # Deduplicate chunks per conv
        kept_chunks: List[Chunk] = []
        if chunks:
            seen = self._seen_texts.setdefault(conv_id, set())
            for ch in chunks:
                h = hash(ch.text)
                if h in seen:
                    continue
                seen.add(h)
                kept_chunks.append(ch)

        # Upsert facts with supersede semantics + graph sync
        added_facts = 0
        for f in facts:
            prev = self.facts.supersede_slot(f.subject, f.predicate)
            if prev:
                f = f.copy_with(supersedes=prev.fact_id)
                # remove old edge summary (if any)
                self.graph.remove(prev)
            self.facts.add(f)
            self.graph.upsert(f)
            added_facts += 1

        # Upsert chunks (embed-on-write)
        if kept_chunks:
            self.vec.upsert(kept_chunks)
            self._docs.setdefault(conv_id, []).extend(kept_chunks)

        self.after_add(
            conv_id, turn_id, role, text,
            added_summary={
                "facts_added": added_facts,
                "chunks_added": len(kept_chunks),
            }
        )

    # ---------------------------
    # Retrieval
    # ---------------------------
    def retrieve(self, conv_id: str, query_text: str, top_k: int = 5, **kwargs) -> Dict[str, any]:
        t0 = time.perf_counter()
        self.before_retrieve(conv_id, query_text, top_k)

        # 1) Semantic over chunks
        sem_hits = self.vec.search(query_text, k=max(1, top_k * 4))

        # 2) Structured facts — try subject hint if present
        subject_hint: Optional[str] = kwargs.get("subject_hint")
        if subject_hint:
            fact_hits = self.facts.search_subject(subject_hint, limit=top_k * 2)
        else:
            fact_hits = self.facts.list_recent(limit=top_k * 4)

        # 3) Graph expansion — seed with hint and naive proper-noun heuristics
        seeds: List[str] = []
        if subject_hint:
            seeds.append(subject_hint)
        for tok in query_text.split():
            if tok[:1].isupper() and len(tok) > 2:
                seeds.append(tok)
        graph_edges = self.graph.k_hop_expand(list(set(seeds)), k=2, max_edges=top_k * 4)

        # Score fusion (alpha,beta,gamma,delta)
        scored: List[Tuple[float, RetrievalItem]] = []

        # Chunks: semantic + recency
        for ch, sem in sem_hits:
            rec = self._recency_score(conv_id, ch)
            s = self.alpha * sem + self.gamma * rec
            scored.append((
                s,
                RetrievalItem(
                    type="chunk",
                    id=ch.chunk_id,
                    text=ch.text,
                    source=ch.source,
                    created_at=ch.created_at,
                    score=round(float(s), 4),
                    scores={"semantic": float(sem), "recency": float(rec)},
                )
            ))

        # Facts: confidence + crude kw hit + recency
        ql = query_text.lower()
        for f in fact_hits:
            if not f.active or f.confidence < self.conf_thr:
                continue
            kw = 1.0 if (f.subject.lower() in ql or f.object.lower() in ql or f.predicate.lower() in ql) else 0.6
            rec = self._time_decay(f.created_at)
            base = 0.5 * kw + 0.5 * float(f.confidence)
            s = self.beta * base + self.gamma * rec
            scored.append((
                s,
                RetrievalItem(
                    type="fact",
                    id=f.fact_id,
                    subject=f.subject,
                    predicate=f.predicate,
                    object=f.object,
                    source=f.source,
                    created_at=f.created_at,
                    score=round(float(s), 4),
                    scores={"kw_conf": float(base), "recency": float(rec)},
                )
            ))

        # Graph edges: (u, p, v) with confidence + recency + mention hit
        for (u, v, p, data) in graph_edges:
            hit = 1.0 if (u in ql or v in ql or p in ql) else 0.7
            conf = float(data.get("confidence", 0.7))  # type: ignore
            rec = self._time_decay(str(data.get("created_at", now_iso())))  # type: ignore
            gscore = self.delta * (0.5 * hit + 0.5 * conf) + self.gamma * rec
            scored.append((
                gscore,
                RetrievalItem(
                    type="graph_edge",
                    u=u, v=v, predicate=p,
                    fact_ids=sorted(list(data.get("fact_ids", []))),  # type: ignore
                    created_at=str(data.get("created_at", now_iso())),  # type: ignore
                    score=round(float(gscore), 4),
                    scores={"graph": float(gscore), "recency": float(rec)},
                )
            ))

        # 4) Deduplicate & take top_k
        results = self._dedup_and_topk(scored, k=top_k)

        out = RetrievalResult(query=query_text, results=results)
        latency = (time.perf_counter() - t0) * 1000.0
        self.after_retrieve(conv_id, query_text, top_k, results_summary=out.to_dict(), latency_ms=latency)
        return out.to_dict()

    # ---------------------------
    # Helpers
    # ---------------------------
    def _recency_score(self, conv_id: str, ch: Chunk) -> float:
        """Position-based recency proxy for chunks."""
        arr = self._docs.get(conv_id, [])
        if not arr:
            return 0.5
        try:
            idx = arr.index(ch)
        except ValueError:
            idx = len(arr) - 1
        # Map position to pseudo-age in days (every 10 turns ~ 1 day)
        age_days = max(0.0, (len(arr) - 1 - idx) / 10.0)
        return self._exp_decay(age_days, self.half_life)

    @staticmethod
    def _exp_decay(age_days: float, half_life_days: float) -> float:
        half = max(half_life_days, 1e-6)
        return 0.5 ** (age_days / half)

    def _time_decay(self, created_at_iso: str) -> float:
        try:
            ts = datetime.strptime(created_at_iso, "%Y-%m-%dT%H:%M:%S")
        except Exception:
            return 0.5
        age_days = max(0.0, (datetime.utcnow() - ts).total_seconds() / 86400.0)
        return self._exp_decay(age_days, self.half_life)

    @staticmethod
    def _dedup_and_topk(scored: List[Tuple[float, RetrievalItem]], k: int) -> List[RetrievalItem]:
        scored.sort(key=lambda x: x[0], reverse=True)
        seen: set[str] = set()
        out: List[RetrievalItem] = []
        for s, item in scored:
            # normalize key per item type
            if item.type == "chunk":
                key = f"chunk::{(item.text or '').strip().lower()}"
            elif item.type == "fact":
                key = f"fact::{(item.subject or '').lower()}::{(item.predicate or '').lower()}::{(item.object or '').lower()}"
            else:  # graph_edge
                key = f"edge::{(item.u or '')}::{(item.predicate or '')}::{(item.v or '')}"
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
            if len(out) >= max(1, k):
                break
        return out

    # ---------------------------
    # Lifecycle & config
    # ---------------------------
    def reset(self) -> None:
        self.vec.clear()
        self._docs.clear()
        self._seen_texts.clear()
        # facts/graph persist unless you recreate the DB file; add truncate if desired

    def configure(self, **kwargs) -> None:
        if "index_user_only" in kwargs and kwargs["index_user_only"] is not None:
            self.index_user_only = bool(kwargs["index_user_only"])
        if "confidence_threshold" in kwargs and kwargs["confidence_threshold"] is not None:
            self.conf_thr = float(kwargs["confidence_threshold"])
        if "recency_half_life_days" in kwargs and kwargs["recency_half_life_days"] is not None:
            self.half_life = float(kwargs["recency_half_life_days"])
        if "weights" in kwargs and kwargs["weights"] is not None:
            a, b, g, d = kwargs["weights"]
            self.alpha, self.beta, self.gamma, self.delta = float(a), float(b), float(g), float(d)
