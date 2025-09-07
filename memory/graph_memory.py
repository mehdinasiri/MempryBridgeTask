# repo/memory/graph_memory.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from .vector_memory import VectorMemory
from utils.models import RetrievalItem, RetrievalResult, norm_entity
from utils.graph_store import GraphStoreSQLAlchemy


class GraphMemory(VectorMemory):
    """
    Graph-augmented memory that *reuses* VectorMemory’s:
      - LLM fact extraction + evidence capture
      - slotting (active current fact per (subject,predicate))
      - vector index & retrieval
    and adds a persisted SQL (SQLite/SQLAlchemy) graph built from *active* facts.

    Retrieval runs VectorMemory first, then lightly augments results using k-hop
    edges near entities mentioned in the query (and the conversation alias).
    """

    def __init__(
        self,
        *args,
        graph_db_path: str = ".memdb/graph.sqlite",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._graph = GraphStoreSQLAlchemy(db_path=graph_db_path)
        logger.info(f"GraphMemory[{self.name}] graph store initialized at {graph_db_path}")

    # ---------------------------
    # Ingestion: same as VectorMemory + sync graph for this conversation
    # ---------------------------
    def add_turn(self, conv_id: str, turn_id: int, role: str, text: Any) -> None:
        """Run normal VectorMemory ingestion, then refresh graph rows for this conversation."""
        super().add_turn(conv_id, turn_id, role, text)
        self._sync_graph_for_conv(conv_id)

    def _sync_graph_for_conv(self, conv_id: str) -> None:
        """Rebuild graph edges for this conversation from *active* facts only."""
        # Clear existing rows for this conv
        self._graph.begin_conv_refresh(conv_id)

        # Reinsert current active facts for this conv
        count = 0
        for f in self._facts.values():
            if not f.active:
                continue
            # Scope to this conversation (by source prefix "<conv_id>:turn_X")
            src_conv = (f.source or "").split(":")[0]
            if src_conv != conv_id:
                continue

            subj = norm_entity(f.subject)
            pred = norm_entity(f.predicate)
            obj = norm_entity(f.object)
            evidence = (f.meta or {}).get("evidence")

            self._graph.upsert_active_fact(
                conv_id=conv_id,
                fact_id=f.fact_id,
                subject_norm=subj,
                predicate_norm=pred,
                object_norm=obj,
                confidence=float(f.confidence),
                created_at=f.created_at,
                evidence=evidence,
            )
            count += 1

        logger.debug(f"[graph_sync] conv={conv_id} active_facts_indexed={count}")

    # ---------------------------
    # Retrieval: call parent first, then augment via graph k-hop neighbors
    # ---------------------------
    def retrieve(self, conv_id: str, query_text: str, top_k: int = 5, **kwargs) -> Dict[str, Any]:
        """
        Returns the same schema as VectorMemory, but we augment results by:
          - mining graph edges k-hop around seed entities (conv alias + capitalized tokens)
          - translating edges back into active facts (whenever possible) and appending.
        """
        base = super().retrieve(conv_id, query_text, top_k=top_k, **kwargs)
        # If nothing came back or we already filled top_k, nothing to do
        if not base.get("results") or len(base["results"]) >= top_k:
            return base

        # Seeds: conversation alias + capitalized tokens in the query
        seeds: List[str] = []
        alias = self._conv_subject.get(conv_id)
        if alias:
            seeds.append(alias)
        for tok in (query_text or "").split():
            if tok and tok[0].isupper() and len(tok) > 2:
                seeds.append(tok)

        if not seeds:
            return base

        # Query graph store (scope by conv if restrict_to_conv)
        edges = self._graph.k_hop(
            conv_id=conv_id if self.restrict_to_conv else None,
            seeds=seeds,
            k=2,
            limit_edges=max(32, top_k * 8),
            only_active=True,
        )

        if not edges:
            return base

        merged = list(base["results"])
        # Track duplicates using the same keying scheme as VectorMemory facts
        seen_fact_keys = {
            f"fact::{(it.get('subject') or '').lower()}::{(it.get('predicate') or '').lower()}::{(it.get('object') or '').lower()}"
            for it in merged if it.get("type") == "fact"
        }

        appended = 0
        for (u, v, p, data) in edges:
            # Translate edge back into a current active fact (prefer registry for richer metadata/evidence)
            fid = data.get("fact_id")
            f = self._facts.get(fid or "")
            if not f or not f.active:
                continue

            # If restrict_to_conv, keep only this conversation
            src_conv = (f.source or "").split(":")[0]
            if self.restrict_to_conv and src_conv != conv_id:
                continue

            key = f"fact::{f.subject.lower()}::{f.predicate.lower()}::{f.object.lower()}"
            if key in seen_fact_keys:
                continue

            merged.append({
                "type": "fact",
                "id": f.fact_id,
                "subject": f.subject,
                "predicate": f.predicate,
                "object": f.object,
                "source": f.source,
                "created_at": f.created_at,
                "score": round(float(data.get("confidence", 0.2)) * 0.2, 6),  # small booster
                "text": (f.meta or {}).get("evidence"),
            })
            seen_fact_keys.add(key)
            appended += 1

            if len(merged) >= top_k:
                break

        logger.debug(f"[graph_augment] conv={conv_id} appended={appended} total={len(merged)}")
        return RetrievalResult(query=base["query"], results=merged[:max(1, top_k)]).to_dict()

    # ---------------------------
    # Lifecycle
    # ---------------------------
    def reset(self) -> None:
        """Clear vector memory and the graph store."""
        super().reset()
        self._graph.clear()
        logger.debug("[reset] graph store cleared")
