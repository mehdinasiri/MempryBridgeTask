# repo/memory/models.py
from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import json
import math
import re
import uuid


# -------------------------------------------------------------------
# Time & IDs
# -------------------------------------------------------------------

def now_iso() -> str:
    """UTC timestamp in RFC3339/ISO8601 with Z suffix, second precision."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def new_id(prefix: str) -> str:
    """Short, unique, readable id."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# -------------------------------------------------------------------
# Normalization helpers
# -------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")


def _ensure_str(x: Any) -> str:
    return "" if x is None else str(x)


def norm_text(x: Any) -> str:
    """Whitespace collapse + trim; lossless for content."""
    s = _ensure_str(x).strip()
    return _WHITESPACE_RE.sub(" ", s)


def norm_entity(x: Any) -> str:
    """Entity normalization for keys/joins (lowercased)."""
    return norm_text(x).lower()


# -------------------------------------------------------------------
# Data Models
# -------------------------------------------------------------------

@dataclass
class AtomicFact:
    """
    Minimal, updatable, auditable fact unit.
    Use (subject, predicate) as an 'update slot' (latest fact supersedes earlier).
    """
    fact_id: str
    subject: str
    predicate: str
    object: str
    source: str                     # e.g., "conv_12:turn_3"
    confidence: float               # [0..1]
    created_at: str                 # ISO8601 (UTC, Z)
    supersedes: Optional[str] = None
    active: bool = True
    meta: Dict[str, Any] = field(default_factory=dict)

    # ---- convenience ----
    def key(self) -> Tuple[str, str]:
        return (norm_entity(self.subject), norm_entity(self.predicate))

    def copy_with(self, **updates) -> "AtomicFact":
        return replace(self, **updates)

    # ---- serialization ----
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "AtomicFact":
        return AtomicFact(**d)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @staticmethod
    def from_json(s: str) -> "AtomicFact":
        return AtomicFact.from_dict(json.loads(s))

    # ---- validation ----
    def validate(self) -> None:
        if not self.fact_id:
            raise ValueError("fact_id required")
        if not self.subject:
            raise ValueError("subject required")
        if not self.predicate:
            raise ValueError("predicate required")
        if self.object is None or len(str(self.object)) == 0:
            raise ValueError("object required")
        c = float(self.confidence)
        if not (0.0 <= c <= 1.0):
            raise ValueError("confidence must be in [0,1]")


@dataclass
class Chunk:
    """
    Semantically coherent text snippet for vector retrieval.
    Embedding is stored externally or inline as a list[float].
    """
    chunk_id: str
    text: str
    source: str                     # e.g., "conv_12:turn_2"
    created_at: str                 # ISO8601 (UTC, Z)
    embedding: Optional[List[float]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    # ---- serialization ----
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Chunk":
        return Chunk(**d)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @staticmethod
    def from_json(s: str) -> "Chunk":
        return Chunk.from_dict(json.loads(s))

    # ---- validation ----
    def validate(self) -> None:
        if not self.chunk_id:
            raise ValueError("chunk_id required")
        if self.text is None or len(self.text) == 0:
            raise ValueError("text required")
        if self.embedding is not None:
            if not isinstance(self.embedding, list) or not all(isinstance(x, (int, float)) for x in self.embedding):
                raise ValueError("embedding must be a list of numbers")


# -------------------------------------------------------------------
# Retrieval Types
# -------------------------------------------------------------------

@dataclass
class RetrievalItem:
    """
    Unified retrieval item envelope consumed by the benchmark/evaluator.
    One of: 'fact', 'chunk', 'graph_edge'.
    """
    type: str                       # "fact" | "chunk" | "graph_edge"
    id: Optional[str] = None        # fact_id or chunk_id
    text: Optional[str] = None      # populated for chunks (or formatted facts)
    subject: Optional[str] = None   # facts / graph_edge (u)
    predicate: Optional[str] = None # facts / graph_edge (p)
    object: Optional[str] = None    # facts / graph_edge (v)
    source: Optional[str] = None
    created_at: Optional[str] = None
    score: Optional[float] = None
    scores: Dict[str, float] = field(default_factory=dict)
    # graph specifics
    u: Optional[str] = None
    v: Optional[str] = None
    fact_ids: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalResult:
    """Standard retrieval response."""
    query: str
    results: List[RetrievalItem]

    def to_dict(self) -> Dict[str, Any]:
        return {"query": self.query, "results": [r.to_dict() for r in self.results]}


# -------------------------------------------------------------------
# Simple numeric utilities
# -------------------------------------------------------------------

def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    num = 0.0
    da = 0.0
    db = 0.0
    for x, y in zip(a, b):
        num += x * y
        da += x * x
        db += y * y
    if da <= 0.0 or db <= 0.0:
        return 0.0
    return num / (math.sqrt(da) * math.sqrt(db))


# -------------------------------------------------------------------
# Factory helpers
# -------------------------------------------------------------------

def make_fact(
    subject: str,
    predicate: str,
    obj: str,
    *,
    source: str,
    confidence: float = 0.85,
    supersedes: Optional[str] = None,
    active: bool = True,
    meta: Optional[Dict[str, Any]] = None,
    fact_id: Optional[str] = None,
    created_at: Optional[str] = None,
) -> AtomicFact:
    f = AtomicFact(
        fact_id=fact_id or new_id("fact"),
        subject=norm_text(subject),
        predicate=norm_text(predicate),
        object=norm_text(obj),
        source=norm_text(source),
        confidence=float(confidence),
        created_at=created_at or now_iso(),
        supersedes=supersedes,
        active=active,
        meta=meta or {},
    )
    f.validate()
    return f


def make_chunk(
    text: str,
    *,
    source: str,
    embedding: Optional[List[float]] = None,
    meta: Optional[Dict[str, Any]] = None,
    chunk_id: Optional[str] = None,
    created_at: Optional[str] = None,
) -> Chunk:
    c = Chunk(
        chunk_id=chunk_id or new_id("chunk"),
        text=norm_text(text),
        source=norm_text(source),
        created_at=created_at or now_iso(),
        embedding=embedding,
        meta=meta or {},
    )
    c.validate()
    return c
