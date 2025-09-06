# repo/memory/keyword_baseline.py
from __future__ import annotations

import math
import re
import time
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

from .base import MemorySystem
from utils.models import Chunk, RetrievalItem, RetrievalResult, make_chunk, norm_text


# ---------------------------
# Tokenizer + stopwords
# ---------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

DEFAULT_STOPWORDS: Set[str] = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else",
    "for", "to", "of", "in", "on", "at", "by", "with", "from",
    "this", "that", "these", "those", "as", "is", "are", "was", "were",
    "it", "its", "be", "been", "being", "so", "not", "no", "do", "does", "did",
    "you", "your", "yours", "i", "me", "my", "we", "our", "ours",
    "he", "she", "they", "them", "their", "theirs",
}

def tokenize(text: str, stopwords: Set[str] = DEFAULT_STOPWORDS) -> List[str]:
    toks = [t.lower() for t in _TOKEN_RE.findall(text)]
    return [t for t in toks if t not in stopwords]


# ---------------------------
# Keyword baseline (BM25)
# ---------------------------

class KeywordBaseline(MemorySystem):
    """
    Dependency-free BM25 baseline over conversation turns.
    - Indexes only USER turns (ignores assistant/system).
    - Scores with BM25 (k1, b).
    """

    def __init__(self, name: str = "keyword_baseline", bm25_k1: float = 1.5, bm25_b: float = 0.75, hooks=None):
        super().__init__(name=name, hooks=hooks)
        self.k1 = float(bm25_k1)
        self.b = float(bm25_b)

        # Per-conversation stores
        self._docs: Dict[str, List[Chunk]] = defaultdict(list)
        self._tfs: Dict[str, List[Counter[str]]] = defaultdict(list)
        self._dfs: Dict[str, Counter[str]] = defaultdict(Counter)
        self._doc_lengths: Dict[str, List[int]] = defaultdict(list)
        self._seen_texts: Dict[str, Set[int]] = defaultdict(set)

    # ---------------------------
    # Ingestion
    # ---------------------------

    def add_turn(self, conv_id: str, turn_id: int, role: str, text: str) -> None:
        self.before_add(conv_id, turn_id, role, text)

        # Ignore non-user turns
        if role.lower() != "user":
            self.after_add(conv_id, turn_id, role, text, added_summary={"indexed": False, "reason": "not_user"})
            return

        norm = norm_text(text)
        if not norm:
            self.after_add(conv_id, turn_id, role, text, added_summary={"indexed": False, "reason": "empty"})
            return

        # Deduplicate
        h = hash(norm)
        if h in self._seen_texts[conv_id]:
            self.after_add(conv_id, turn_id, role, text, added_summary={"indexed": False, "reason": "duplicate"})
            return
        self._seen_texts[conv_id].add(h)

        # Store chunk + update stats
        ch = make_chunk(norm, source=f"{conv_id}:{turn_id}")
        self._docs[conv_id].append(ch)

        toks = tokenize(norm)
        tf = Counter(toks)
        self._tfs[conv_id].append(tf)
        self._doc_lengths[conv_id].append(sum(tf.values()))

        for term in tf.keys():
            self._dfs[conv_id][term] += 1

        self.after_add(conv_id, turn_id, role, text, added_summary={
            "indexed": True,
            "chunk_id": ch.chunk_id,
            "unique_terms": len(tf)
        })

    # ---------------------------
    # Retrieval
    # ---------------------------

    def retrieve(self, conv_id: str, query_text: str, top_k: int = 5, **kwargs) -> Dict[str, any]:
        t0 = time.perf_counter()
        self.before_retrieve(conv_id, query_text, top_k)

        docs = self._docs.get(conv_id, [])
        if not docs:
            latency = (time.perf_counter() - t0) * 1000.0
            result = RetrievalResult(query=query_text, results=[])
            self.after_retrieve(conv_id, query_text, top_k, result.to_dict(), latency_ms=latency)
            return result.to_dict()

        tfs = self._tfs[conv_id]; df = self._dfs[conv_id]; lengths = self._doc_lengths[conv_id]
        N = max(len(docs), 1); avgdl = (sum(lengths) / len(lengths)) if lengths else 0.0

        q_terms = tokenize(query_text)
        idf: Dict[str, float] = {t: math.log(((N - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5)) + 1.0) for t in set(q_terms)}

        scores: List[Tuple[float, int]] = []
        for i, tf in enumerate(tfs):
            dl = lengths[i] if i < len(lengths) else 0
            s = 0.0
            for t in q_terms:
                if t not in tf: continue
                f = tf[t]
                denom = f + self.k1 * (1.0 - self.b + self.b * (dl / (avgdl or 1.0)))
                s += idf[t] * (f * (self.k1 + 1.0)) / (denom or 1e-9)
            if s > 0: scores.append((s, i))

        scores.sort(key=lambda x: x[0], reverse=True)
        items: List[RetrievalItem] = [
            RetrievalItem(
                type="chunk", id=docs[i].chunk_id,
                text=docs[i].text, source=docs[i].source, created_at=docs[i].created_at,
                score=round(s, 4), scores={"bm25": s}
            )
            for s, i in scores[:max(1, top_k)]
        ]

        result = RetrievalResult(query=query_text, results=items)
        latency = (time.perf_counter() - t0) * 1000.0
        self.after_retrieve(conv_id, query_text, top_k, result.to_dict(), latency_ms=latency)
        return result.to_dict()

    # ---------------------------
    # Lifecycle
    # ---------------------------

    def reset(self) -> None:
        self._docs.clear(); self._tfs.clear(); self._dfs.clear()
        self._doc_lengths.clear(); self._seen_texts.clear()
