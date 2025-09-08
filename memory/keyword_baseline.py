# repo/memory/keyword_baseline.py
from __future__ import annotations

import math
import re
import time
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

from loguru import logger

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
    """Return lowercase tokens (alnum + underscore), removing simple stopwords."""
    toks = [t.lower() for t in _TOKEN_RE.findall(text or "")]
    return [t for t in toks if t not in stopwords]


# ---------------------------
# Keyword baseline (BM25)
# ---------------------------

class KeywordBaseline(MemorySystem):
    """
    Dependency-free BM25 baseline over conversation turns.

    Behavior:
      - Indexes only USER turns (ignores assistant/system).
      - Deduplicates exact same normalized user text per conversation.
      - Scores with BM25 (k1, b) using per-conversation statistics.
      - Returns top-K chunks that best match the query.

    This is a *baseline* to compare with VectorMemory/GraphMemory.
    """

    def __init__(self, name: str = "keyword_baseline", bm25_k1: float = 1.5, bm25_b: float = 0.75):
        super().__init__(name=name)
        self.k1 = float(bm25_k1)
        self.b = float(bm25_b)

        # Per-conversation stores
        self._docs: Dict[str, List[Chunk]] = defaultdict(list)
        self._tfs: Dict[str, List[Counter[str]]] = defaultdict(list)
        self._dfs: Dict[str, Counter[str]] = defaultdict(Counter)
        self._doc_lengths: Dict[str, List[int]] = defaultdict(list)
        self._seen_texts: Dict[str, Set[int]] = defaultdict(set)

        logger.info(f"KeywordBaseline[{name}] initialized (k1={self.k1}, b={self.b})")

    # ---------------------------
    # Ingestion
    # ---------------------------

    def add_turn(self, conv_id: str, turn_id: int, role: str, text: str) -> None:
        """Index a single USER turn as a document (dedup within conversation)."""
        logger.debug(f"[kb.add_turn] conv={conv_id} turn={turn_id} role={role} type=str")
        if (role or "").lower() != "user":
            logger.debug(f"[kb.add_turn] skipping non-user role={role}")
            return

        norm = norm_text(text)
        if not norm:
            logger.debug("[kb.add_turn] empty text; skip")
            return

        # Deduplicate identical normalized text in this conversation
        h = hash(norm)
        if h in self._seen_texts[conv_id]:
            logger.debug("[kb.add_turn] duplicate text; skip")
            return
        self._seen_texts[conv_id].add(h)

        # Store chunk and update stats
        ch = make_chunk(norm, source=f"{conv_id}:{turn_id}")
        self._docs[conv_id].append(ch)

        toks = tokenize(norm)
        tf = Counter(toks)
        self._tfs[conv_id].append(tf)
        self._doc_lengths[conv_id].append(sum(tf.values()))
        for term in tf.keys():
            self._dfs[conv_id][term] += 1

        logger.info(f"[kb.add_turn] indexed chunk id={ch.chunk_id} terms={len(tf)} conv={conv_id} turn={turn_id}")

    # ---------------------------
    # Retrieval
    # ---------------------------

    def retrieve(self, conv_id: str, query_text: str, top_k: int = 5, **kwargs) -> Dict[str, any]:
        """Return top-K documents from this conversation using BM25 scoring."""
        t0 = time.perf_counter()
        qnorm = norm_text(query_text)
        logger.debug(f"[kb.retrieve] conv={conv_id} top_k={top_k} query_preview={qnorm[:120]!r}")

        docs = self._docs.get(conv_id, [])
        if not qnorm or not docs:
            latency = (time.perf_counter() - t0) * 1000.0
            logger.info(f"[kb.retrieve] no docs or empty query; results=0 latency_ms={latency:.1f}")
            return RetrievalResult(query=qnorm, results=[]).to_dict()

        tfs = self._tfs[conv_id]
        df = self._dfs[conv_id]
        lengths = self._doc_lengths[conv_id]

        N = max(len(docs), 1)
        avgdl = (sum(lengths) / len(lengths)) if lengths else 0.0

        q_terms = tokenize(qnorm)
        # IDF with a small +1.0 stabilizer like in BM25+ variants
        idf: Dict[str, float] = {
            t: math.log(((N - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5)) + 1.0)
            for t in set(q_terms)
        }

        scores: List[Tuple[float, int]] = []
        for i, tf in enumerate(tfs):
            dl = lengths[i] if i < len(lengths) else 0
            s = 0.0
            for t in q_terms:
                if t not in tf:
                    continue
                f = tf[t]
                denom = f + self.k1 * (1.0 - self.b + self.b * (dl / (avgdl or 1.0)))
                s += idf[t] * (f * (self.k1 + 1.0)) / (denom or 1e-9)
            if s > 0:
                scores.append((s, i))

        scores.sort(key=lambda x: x[0], reverse=True)

        items: List[RetrievalItem] = []
        for s, i in scores[:max(1, top_k)]:
            ch = docs[i]
            items.append(
                RetrievalItem(
                    type="chunk",
                    id=ch.chunk_id,
                    text=ch.text,
                    source=ch.source,
                    created_at=ch.created_at,
                    score=round(float(s), 4),
                    scores={"bm25": float(s)},
                )
            )

        latency = (time.perf_counter() - t0) * 1000.0
        logger.info(f"[kb.retrieve] results={len(items)} conv={conv_id} latency_ms={latency:.1f}")
        return RetrievalResult(query=qnorm, results=items).to_dict()

    # ---------------------------
    # Lifecycle
    # ---------------------------

    def reset(self) -> None:
        """Clear all in-memory indices/statistics."""
        self._docs.clear()
        self._tfs.clear()
        self._dfs.clear()
        self._doc_lengths.clear()
        self._seen_texts.clear()
        logger.debug("[kb.reset] cleared all stores")
