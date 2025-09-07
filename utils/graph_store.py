# repo/utils/graph_store.py
from __future__ import annotations

from dataclasses import asdict
from typing import Iterable, List, Optional, Set, Tuple, Dict, Any

from sqlalchemy import (
    Boolean,
    Column,
    Float,
    Index,
    String,
    create_engine,
    or_,
    select, text,
)
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class Edge(Base):
    """
    One row per FACT (active snapshot).
    We store normalized (lowercased) subject/predicate/object for fast lookups.
    """
    __tablename__ = "edges"

    fact_id = Column(String, primary_key=True)
    conv_id = Column(String, index=True, nullable=False)

    subject = Column(String, index=True, nullable=False)
    predicate = Column(String, index=True, nullable=False)
    object = Column(String, index=True, nullable=False)

    confidence = Column(Float, nullable=False, default=0.0)
    created_at = Column(String, nullable=False)
    evidence = Column(String, nullable=True)
    active = Column(Boolean, nullable=False, default=True)


# Helpful composite indexes
Index("idx_edges_spo", Edge.subject, Edge.predicate, Edge.object)
Index("idx_edges_conv_active", Edge.conv_id, Edge.active)


class GraphStoreSQLAlchemy:
    """
    SQLite-backed graph store (via SQLAlchemy).
    - begin_conv_refresh(conv_id): clear rows for this conv (caller reinserts actives)
    - upsert_active_fact(...): write/replace current fact edge (active=1)
    - k_hop(...): BFS up to K hops over subject/object in both directions
    """

    def __init__(self, db_path: str = ".memdb/graph.sqlite"):
        self.engine = create_engine(f"sqlite:///{db_path}", future=True)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False, future=True)

    # ---------------------------
    # Write API
    # ---------------------------
    def begin_conv_refresh(self, conv_id: str) -> None:
        with self.SessionLocal.begin() as sess:
            sess.execute(select(Edge).where(Edge.conv_id == conv_id))  # touch for lock
            sess.execute(text("DELETE FROM edges WHERE conv_id=:cid"), {"cid": conv_id})

    def upsert_active_fact(
        self,
        *,
        conv_id: str,
        fact_id: str,
        subject_norm: str,
        predicate_norm: str,
        object_norm: str,
        confidence: float,
        created_at: str,
        evidence: Optional[str],
    ) -> None:
        with self.SessionLocal.begin() as sess:
            row = sess.get(Edge, fact_id)
            if row is None:
                row = Edge(
                    fact_id=fact_id,
                    conv_id=conv_id,
                    subject=subject_norm,
                    predicate=predicate_norm,
                    object=object_norm,
                    confidence=float(confidence),
                    created_at=created_at,
                    evidence=evidence,
                    active=True,
                )
                sess.add(row)
            else:
                row.conv_id = conv_id
                row.subject = subject_norm
                row.predicate = predicate_norm
                row.object = object_norm
                row.confidence = float(confidence)
                row.created_at = created_at
                row.evidence = evidence
                row.active = True

    # ---------------------------
    # Read API
    # ---------------------------
    def k_hop(
        self,
        *,
        conv_id: Optional[str],
        seeds: Iterable[str],
        k: int = 2,
        limit_edges: int = 256,
        only_active: bool = True,
    ) -> List[Tuple[str, str, str, Dict[str, Any]]]:
        """
        BFS neighborhood (both directions) up to K hops.
        Returns list of (u, v, p, data) where u -[p]-> v.
        """
        seeds_norm: Set[str] = {s for s in (s.strip().lower() for s in seeds) if s}
        if not seeds_norm:
            return []

        out: List[Tuple[str, str, str, Dict[str, Any]]] = []
        seen: Set[Tuple[str, str, str]] = set()
        frontier: Set[str] = set(seeds_norm)
        steps = 0

        with self.SessionLocal() as sess:
            while frontier and steps < max(1, k) and len(out) < limit_edges:
                steps += 1
                # Build base filter
                q = select(Edge)
                if conv_id:
                    q = q.where(Edge.conv_id == conv_id)
                if only_active:
                    q = q.where(Edge.active.is_(True))
                # out-edges or in-edges that touch any node in frontier
                # NOTE: we use OR(subject in frontier, object in frontier)
                q = q.where(
                    or_(Edge.subject.in_(frontier), Edge.object.in_(frontier))
                ).limit(limit_edges - len(out))

                rows = list(sess.execute(q).scalars())
                next_frontier: Set[str] = set()

                for r in rows:
                    key = (r.subject, r.object, r.predicate)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(
                        (
                            r.subject,
                            r.object,
                            r.predicate,
                            {
                                "fact_id": r.fact_id,
                                "confidence": float(r.confidence),
                                "created_at": r.created_at,
                                "evidence": r.evidence,
                                "active": bool(r.active),
                                "conv_id": r.conv_id,
                            },
                        )
                    )
                    if len(out) >= limit_edges:
                        break
                    # Grow frontier with both endpoints
                    if r.subject not in frontier:
                        next_frontier.add(r.subject)
                    if r.object not in frontier:
                        next_frontier.add(r.object)

                frontier = next_frontier

        return out

    # ---------------------------
    # Maintenance
    # ---------------------------
    def clear(self) -> None:
        with self.SessionLocal.begin() as sess:
            sess.execute("DELETE FROM edges")
