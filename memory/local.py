# memory/local.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Iterable
from datetime import datetime
from pathlib import Path
import json
import re
from collections import defaultdict

import numpy as np
from openai import OpenAI

from .base import AbstractMemory
from .config import DB_PATH, CHAT_MODEL, BASE_URL, MB_API_KEY
from .custom_embedder import get_embedding

# --- SQLAlchemy imports
from sqlalchemy import (
    create_engine,
    Column,
    String,
    Float,
    LargeBinary,
    DateTime,
    Integer,
    Text,
    Index,
    event,
    select,
    func,
    and_,
)
from sqlalchemy.orm import declarative_base, Session


# ============================================================
# Minimal, domain-agnostic SPO event log (append-only)
# ============================================================

Base = declarative_base()


class FactEvent(Base):
    """
    Minimal immutable fact event about (subject, predicate, object).
    """
    __tablename__ = "fact_events"

    id = Column(Integer, primary_key=True, autoincrement=True)

    subject = Column(String, index=True, nullable=False)
    predicate = Column(String, index=True, nullable=False)
    object = Column("object", Text, nullable=False)

    assertion = Column(String, nullable=False)   # 'present' | 'absent' | 'uncertain'
    confidence = Column(Float, default=0.5, nullable=False)

    evidence_text = Column(Text, nullable=True)
    turn_id = Column(String, nullable=True)
    timestamp = Column(DateTime, nullable=False, index=True)

    embedding = Column(LargeBinary, nullable=True)


Index("ix_fact_events_spo_time", FactEvent.subject, FactEvent.predicate, FactEvent.object, FactEvent.timestamp.desc())
Index("ix_fact_events_sp_time", FactEvent.subject, FactEvent.predicate, FactEvent.timestamp.desc())


class LocalMemory(AbstractMemory):
    """
    Memory backend with:
      - Append-only store of FactEvent (subject, predicate, object) with assertion {present, absent, uncertain}
      - Canonical predicate normalization + small predicate families to handle updates consistently
      - Optional subject aliasing (e.g., 'She' alias_of 'Mary') applied in the current view
      - Retrieval via embedding similarity with optional hybrid scoring (cosine + confidence + recency)
      - Optional subject-aware boost and absent penalty (for include_absent=True)
    """

    # ----- canonical predicate map (normalize synonyms/aliases) -----
    CANONICAL_PREDICATE_MAP: Dict[str, str] = {
        # identity
        "name": "name",
        "display_name": "name",
        # residence
        "lives_in": "lives_in",
        "resides_in": "lives_in",
        "located_in": "lives_in",
        "now_in": "lives_in",
        "moved_to": "lives_in",
        "relocated_to": "lives_in",
        "moved": "lives_in",
        # work
        "works_at": "works_at",
        "employer": "works_at",
        "joined": "works_at",
        "now_at": "works_at",
        "employed_at": "works_at",
        # relations
        "has_friend": "has_friend",
        "friend": "has_friend",
        "is_friend_with": "has_friend",
        # profession
        "occupation": "occupation",
        "works_as": "occupation",
        "role": "occupation",
        "job_title": "occupation",
        # subject aliasing / reference correction
        "alias_of": "alias_of",
        "refers_to": "alias_of",
        "refers_to_in_turn": "alias_of",
    }

    # ----- predicate families (single-valued "slots") -----
    # Any new 'present' in a family supersedes the previous value for that same family.
    PREDICATE_FAMILIES: Dict[str, Iterable[str]] = {
        "lives_in": {"lives_in", "resides_in", "located_in", "now_in", "moved_to", "relocated_to", "moved"},
        "works_at": {"works_at", "employer", "joined", "now_at", "employed_at"},
        "name": {"name", "display_name"},
        "occupation": {"occupation", "works_as", "role", "job_title"},
    }

    # Predicates treated as single-valued in time (latest wins)
    SINGLE_VALUED: set[str] = set(PREDICATE_FAMILIES.keys())

    # Pronoun tokens we may see as subjects in extraction
    PRONOUN_SUBJECTS = {"she", "he", "they", "them", "her", "him"}

    def __init__(self):
        self.engine = None
        self.session: Session | None = None
        self.client = OpenAI(base_url=BASE_URL, api_key=MB_API_KEY)

    # ------------- connection -------------
    def connect(self, **kwargs) -> None:
        db_path = Path(kwargs.get("db_path", DB_PATH)).expanduser().resolve()
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(
            f"sqlite:///{db_path}",
            future=True,
            echo=kwargs.get("echo", False),
            connect_args={"check_same_thread": False},
        )

        @event.listens_for(self.engine, "connect")
        def _set_sqlite_pragma(dbapi_conn, connection_record):  # noqa: ANN001
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL;")
            cur.execute("PRAGMA synchronous=NORMAL;")
            cur.close()

        Base.metadata.create_all(self.engine)
        self.session = Session(self.engine)

    # ------------- instruction -------------
    BASE_INSTRUCTION = (
        "Extract stable, verifiable facts or knowledge from the input.\n"
        "Output ONLY a JSON array of objects. Each object MUST follow exactly:\n\n"
        "- subject (string): concrete entity/concept/item. For the speaker, ALWAYS use 'User'.\n"
        "- predicate (string): choose ONLY from the canonical set: "
        "  {'name','lives_in','works_at','has_friend','occupation','alias_of'} "
        "(avoid synonyms; e.g., use 'lives_in' not 'relocated_to', 'works_at' not 'joined').\n"
        "- object (string): the value or target of the relation (concise string).\n"
        "- assertion (string): one of {'present','absent','uncertain'}.\n"
        "- confidence (float): numeric 0.0..1.0.\n"
        "- evidence (string): short supporting quote from the input.\n"
        "- timestamp (string, optional): leave empty (system sets this).\n\n"
        "Rules:\n"
        "1) Output raw JSON array only (no extra text or markdown fences).\n"
        "2) Multi-hop: emit one object per fact.\n"
        "3) Changes: for phrases like 'moved/relocated/now/changed', emit 'present' for the NEW value and "
        "'absent' for the OLD value using the SAME canonical predicate (e.g., 'lives_in').\n"
        "4) Negations: 'no longer','not anymore' imply 'absent'.\n"
        "5) Pronouns: 'I','my','me','I'm' → subject='User'.\n"
        "6) Pronoun clarification: if a later sentence clarifies a prior pronoun (e.g., 'I meant Mary'), emit "
        "   an 'alias_of' fact mapping the pronoun subject to the named subject (e.g., 'She' alias_of 'Mary'), "
        "   and re-emit the corrected fact for the named subject as 'present'.\n"
    )

    # ------------- helpers -------------
    @staticmethod
    def _normalize_confidence(val) -> float:
        if val is None:
            return 0.5
        if isinstance(val, (int, float)):
            return max(0.0, min(1.0, float(val)))
        if isinstance(val, str):
            v = val.strip().lower()
            mapping = {"very low": 0.1, "low": 0.25, "medium": 0.5, "mid": 0.5, "high": 0.75, "very high": 0.9, "certain": 1.0}
            if v in mapping:
                return mapping[v]
            try:
                return max(0.0, min(1.0, float(v)))
            except Exception:
                return 0.5
        return 0.5

    @staticmethod
    def _to_snake_case(s: str) -> str:
        s = (s or "").strip()
        s = s.replace("-", " ").replace("/", " ")
        s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
        s = re.sub(r"\s+", "_", s.strip().lower())
        return s

    def _canonicalize_predicate(self, p: str) -> str:
        p = self._to_snake_case(p)
        return self.CANONICAL_PREDICATE_MAP.get(p, p)

    @staticmethod
    def _canonical_triple_text(subject: str, predicate: str, obj: str) -> str:
        return f"{subject}'s {predicate} is {obj}"

    def _embed_bytes(self, text: str) -> bytes:
        vec = get_embedding(text)
        return np.asarray(vec, dtype=np.float32).tobytes()

    # ------------- extraction -------------
    def _extract_items(self, text: str, turn_id: str) -> List[Dict]:
        messages = [
            {"role": "system", "content": self.BASE_INSTRUCTION},
            {"role": "user", "content": text},
        ]
        resp = self.client.chat.completions.create(model=CHAT_MODEL, messages=messages, temperature=0)
        raw = (resp.choices[0].message.content or "").strip()

        try:
            items = json.loads(raw)
        except Exception:
            m = re.search(r"```(?:json)?(.*?)```", raw, re.DOTALL | re.IGNORECASE)
            items = json.loads(m.group(1).strip()) if m else []

        now = datetime.utcnow().isoformat()
        out: List[Dict] = []
        for it in items:
            subj = (it.get("subject") or "").strip()
            pred = self._canonicalize_predicate(it.get("predicate") or "")
            obj = (it.get("object") or "").strip()
            if not subj or not pred or not obj:
                continue
            out.append({
                "subject": subj,
                "predicate": pred,
                "object": obj,
                "assertion": (it.get("assertion") or "present").strip().lower(),
                "confidence": self._normalize_confidence(it.get("confidence")),
                "evidence": it.get("evidence"),
                "turn_id": turn_id,
                "timestamp": now,
            })
        return out

    # ------------- family helpers -------------
    def _family_for(self, predicate: str) -> Tuple[str, set]:
        """Return (canonical, family_set) for a predicate."""
        canonical = self._canonicalize_predicate(predicate)
        fam = None
        for k, members in self.PREDICATE_FAMILIES.items():
            if canonical in members or canonical == k:
                fam = set(members) | {k}
                return k, fam
        return canonical, {canonical}

    def _latest_active_for_subject_family(self, subject: str, family_canonical: str, family_set: Iterable[str]) -> Optional[FactEvent]:
        """Latest present/uncertain event for subject across all predicates in a family."""
        assert self.session is not None
        q = (
            select(FactEvent)
            .where(
                FactEvent.subject == subject,
                FactEvent.predicate.in_(family_set),
                FactEvent.assertion.in_(("present", "uncertain")),
            )
            .order_by(FactEvent.timestamp.desc(), FactEvent.id.desc())
        )
        return self.session.execute(q).scalars().first()

    # ------------- event insertion -------------
    def _insert_event(
        self,
        *,
        subject: str,
        predicate: str,
        obj: str,
        assertion: str,
        confidence: float,
        evidence: Optional[str],
        turn_id: Optional[str],
    ) -> List[Dict]:
        """
        Insert one logical item, returning a list of created event summaries with status:
          - status='deleted' for auto-inserted 'absent' (superseded)
          - status='added'   for the requested event itself
        """
        assert self.session is not None
        now = datetime.utcnow()
        created: List[Dict] = []

        # Normalize predicate to canonical + find family
        canonical_pred = self._canonicalize_predicate(predicate)

        # alias_of is meta; store as-is (no single-valued behavior)
        if canonical_pred == "alias_of":
            triple_text = self._canonical_triple_text(subject, "alias_of", obj)
            self.session.add(FactEvent(
                subject=subject,
                predicate="alias_of",
                object=obj,
                assertion=assertion,
                confidence=confidence,
                evidence_text=evidence,
                turn_id=turn_id,
                timestamp=now,
                embedding=self._embed_bytes(triple_text),
            ))
            created.append({
                "status": "added",
                "subject": subject,
                "predicate": "alias_of",
                "object": obj,
                "assertion": assertion,
                "confidence": confidence,
                "evidence": evidence,
            })
            return created

        fam_key, fam_set = self._family_for(canonical_pred)

        # If single-valued family and asserting new present/uncertain → auto-absent previous value
        if fam_key in self.SINGLE_VALUED and assertion in ("present", "uncertain"):
            prev = self._latest_active_for_subject_family(subject, fam_key, fam_set)
            if prev and (prev.object or "").strip().lower() != (obj or "").strip().lower():
                prev_text = self._canonical_triple_text(subject, fam_key, prev.object)
                self.session.add(FactEvent(
                    subject=subject,
                    predicate=fam_key,  # store absent on canonical key
                    object=prev.object,
                    assertion="absent",
                    confidence=1.0,
                    evidence_text="system: superseded",
                    turn_id=turn_id,
                    timestamp=now,
                    embedding=self._embed_bytes(prev_text),
                ))
                created.append({
                    "status": "deleted",
                    "subject": subject,
                    "predicate": fam_key,
                    "object": prev.object,
                    "assertion": "absent",
                    "confidence": 1.0,
                    "evidence": "system: superseded",
                })

        # Insert the requested event (use canonical predicate)
        triple_text = self._canonical_triple_text(subject, fam_key, obj)
        self.session.add(FactEvent(
            subject=subject,
            predicate=fam_key,
            object=obj,
            assertion=assertion,
            confidence=confidence,
            evidence_text=evidence,
            turn_id=turn_id,
            timestamp=now,
            embedding=self._embed_bytes(triple_text),
        ))
        created.append({
            "status": "added",
            "subject": subject,
            "predicate": fam_key,
            "object": obj,
            "assertion": assertion,
            "confidence": confidence,
            "evidence": evidence,
        })
        return created

    # ------------- public ingestion API -------------
    def add_turn(self, *, text: str, conv_id: str, turn_id: str, user_id: str) -> Any:
        del conv_id, user_id
        items = self._extract_items(text, turn_id)

        assert self.session is not None
        inserted = 0
        changes: List[Dict] = []
        for it in items:
            created = self._insert_event(
                subject=it["subject"],
                predicate=it["predicate"],
                obj=it["object"],
                assertion=it["assertion"],
                confidence=it["confidence"],
                evidence=it.get("evidence"),
                turn_id=turn_id,
            )
            inserted += 1  # count logical inputs, not physical events
            changes.extend(created)

        self.session.commit()
        return {"inserted": inserted, "facts": changes}

    def add_conversation(self, *, messages: List[Dict[str, str]], conv_id: str, user_id: str) -> Any:
        inserted = 0
        t = 0
        for msg in messages:
            if msg.get("role") != "user":
                continue
            t += 1
            res = self.add_turn(text=msg.get("content", ""), conv_id=conv_id, turn_id=f"turn_{t}", user_id=user_id)
            inserted += res.get("inserted", 0)
        return {"inserted": inserted}

    # ------------- alias resolution for current view -------------
    @staticmethod
    def _build_alias_map(rows: List[FactEvent]) -> Dict[str, str]:
        """
        Build alias mapping from latest 'alias_of' present rows.
        Map source_subject -> target_subject (e.g., 'She' -> 'Mary').
        """
        # Take latest alias_of per (subject)
        latest: Dict[str, Tuple[datetime, str]] = {}
        for r in rows:
            if r.predicate != "alias_of":
                continue
            if r.assertion not in ("present", "uncertain"):
                continue
            ts = r.timestamp
            prev = latest.get(r.subject)
            if prev is None or ts > prev[0]:
                latest[r.subject] = (ts, r.object)

        # Resolve chains (A->B, B->C => A->C)
        raw_map = {src: dst for src, (_, dst) in latest.items()}

        def resolve(x: str) -> str:
            seen = set()
            cur = x
            while cur in raw_map and cur not in seen:
                seen.add(cur)
                cur = raw_map[cur]
            return cur

        alias_map = {src: resolve(src) for src in raw_map.keys()}
        # avoid identity/self loops
        alias_map = {k: v for k, v in alias_map.items() if k != v}
        return alias_map

    @staticmethod
    def _apply_alias(subject: str, alias_map: Dict[str, str]) -> str:
        return alias_map.get(subject, subject)

    # ------------- current view computation -------------
    def _current_view_rows(self, include_absent: bool = False) -> List[FactEvent]:
        """
        Current view over all facts WITH aliasing:
          - group by (subject, predicate, object), take the latest event (SQL).
          - build alias map from 'alias_of' rows.
          - map subjects via alias_map, then re-deduplicate in-memory by (subject, predicate, object)
            keeping the newest timestamp.
          - if include_absent=False, keep only assertion ∈ {'present','uncertain'}.
        """
        assert self.session is not None
        subq = (
            select(
                FactEvent.subject.label("s"),
                FactEvent.predicate.label("p"),
                FactEvent.object.label("o"),
                func.max(FactEvent.timestamp).label("ts"),
            )
            .group_by(FactEvent.subject, FactEvent.predicate, FactEvent.object)
            .subquery()
        )

        q = (
            select(FactEvent)
            .join(subq, and_(
                FactEvent.subject == subq.c.s,
                FactEvent.predicate == subq.c.p,
                FactEvent.object == subq.c.o,
                FactEvent.timestamp == subq.c.ts,
            ))
        )
        rows = self.session.execute(q).scalars().all()
        if not rows:
            return []

        # Build alias map from these latest rows
        alias_map = self._build_alias_map(rows)

        # Apply alias to subjects and collapse duplicates keeping the newest timestamp/id
        bucket: Dict[Tuple[str, str, str], FactEvent] = {}
        for r in rows:
            # Skip alias_of rows from the visible current view (they are meta)
            if r.predicate == "alias_of":
                continue
            canon_subj = self._apply_alias(r.subject, alias_map)
            key = (canon_subj, r.predicate, r.object)
            keep = bucket.get(key)
            if keep is None or (r.timestamp, r.id) > (keep.timestamp, keep.id):
                # Make a lightweight copy with subject rewritten
                # (we avoid mutating ORM objects tied to session identity)
                clone = FactEvent(
                    id=r.id,
                    subject=canon_subj,
                    predicate=r.predicate,
                    object=r.object,
                    assertion=r.assertion,
                    confidence=r.confidence,
                    evidence_text=r.evidence_text,
                    turn_id=r.turn_id,
                    timestamp=r.timestamp,
                    embedding=r.embedding,
                )
                bucket[key] = clone

        out = list(bucket.values())
        if not include_absent:
            out = [r for r in out if r.assertion in ("present", "uncertain")]
        return out

    # ------------- similarity & scoring -------------
    @staticmethod
    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
        return float(np.dot(a, b) / denom)

    def _similarity(
        self,
        a: np.ndarray,
        b: np.ndarray,
        *,
        metric: str = "cosine",
        alpha: float = 0.7,
        beta: float = 0.2,
        gamma: float = 0.1,
        confidence: float = 1.0,
        recency: float = 1.0,
    ) -> Tuple[float, float]:
        """
        Return (score, cosine) where:
          - metric='cosine': score=cosine
          - metric='hybrid' : score=alpha*cosine + beta*confidence + gamma*recency
        """
        cos = self._cos(a, b)
        if metric == "cosine":
            return cos, cos
        elif metric == "hybrid":
            score = alpha * cos + beta * float(confidence) + gamma * float(recency)
            return score, cos
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")

    @staticmethod
    def _subject_bonus(query_lower: str, subject: str) -> float:
        """
        Small, lexical and domain-agnostic subject prior:
          - If query uses first-person mentions, boost 'User'
          - If query explicitly mentions the subject string, boost it
        """
        q = f" {query_lower} "
        bonus = 0.0
        if any(tok in q for tok in (" i ", " my ", " me ", " mine ", " i'm ")):
            if subject.strip().lower() == "user":
                bonus += 0.2
        subj_l = subject.strip().lower()
        if subj_l != "user" and f" {subj_l} " in q:
            bonus += 0.2
        return bonus

    # ------------- retrieval -------------
    def refine_query_with_llm(self, query: str) -> str:
        """
        Optional: Use LLM to rewrite query into a fact-compatible form.
        """
        messages = [
            {"role": "system", "content": "Rewrite the user query into a short factual search form "
                                          "using subject/predicate/object terms likely to match stored facts. "
                                          "Keep it concise; prefer forms like 'User lives_in ?'."},
            {"role": "user", "content": query},
        ]
        resp = self.client.chat.completions.create(model=CHAT_MODEL, messages=messages, temperature=0)
        return (resp.choices[0].message.content or query).strip()

    def retrieve(
        self,
        *,
        query: str,
        user_id: str,
        k: int = 5,
        include_absent: bool = False,
        refine_query: bool = False,
        similarity_metric: str = "cosine",   # 'cosine' (default) or 'hybrid'
        alpha: float = 0.7,
        beta: float = 0.2,
        gamma: float = 0.1,
        subject_aware: bool = False,
        absent_penalty: float = 0.15,        # only applied when include_absent=True
        **kwargs,
    ) -> List[Dict[str, Any]]:
        del user_id, kwargs
        assert self.session is not None

        qtext = self.refine_query_with_llm(query) if refine_query else query
        qvec = np.asarray(get_embedding(qtext), dtype=np.float32)
        rows = self._current_view_rows(include_absent=include_absent)

        q_lower = qtext.lower()
        results: List[Tuple[float, Dict]] = []
        now = datetime.utcnow()
        for r in rows:
            emb = r.embedding
            if emb is None:
                txt = self._canonical_triple_text(r.subject, r.predicate, r.object)
                emb = self._embed_bytes(txt)
                r.embedding = emb

            vec = np.frombuffer(emb, dtype=np.float32)
            # recency: 1.0 for fresh, decays linearly across a year
            rec = max(0.0, 1.0 - min(365.0, (now - r.timestamp).days) / 365.0)
            score, cos = self._similarity(
                qvec,
                vec,
                metric=similarity_metric,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                confidence=float(r.confidence),
                recency=rec,
            )

            # optional subject-aware boost (domain-agnostic, purely lexical)
            if subject_aware:
                score += self._subject_bonus(q_lower, r.subject)

            # optional absent penalty (only meaningful if include_absent)
            if include_absent and r.assertion == "absent" and similarity_metric in ("cosine", "hybrid"):
                score *= max(0.0, 1.0 - float(absent_penalty))

            results.append((score, {
                "event_id": r.id,
                "subject": r.subject,
                "predicate": r.predicate,
                "object": r.object,
                "assertion": r.assertion,
                "confidence": float(r.confidence),
                "timestamp": r.timestamp.isoformat(),
                "cosine": float(cos),
                "score": float(score),
            }))

        results.sort(key=lambda x: x[0], reverse=True)
        top = results[: max(1, int(k))]
        return [f for _, f in top]

    def retrieve_reranked(
        self,
        *,
        query: str,
        user_id: str,
        k: int = 5,
        top_n: int = 3,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        from .reranker import rerank_with_llm
        hits = self.retrieve(query=query, user_id=user_id, k=k, **kwargs)
        skinny = [{
            "content": f"{h['subject']}'s {h['predicate']} is {h['object']}",
            "score": h.get("score"),
            "metadata": {"event_id": h.get("event_id")},
        } for h in hits]
        return rerank_with_llm(query, skinny, top_n=top_n)
