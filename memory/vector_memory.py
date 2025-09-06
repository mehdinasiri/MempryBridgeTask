# repo/memory/vector_memory.py
from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger
from openai import OpenAI

from .base import MemorySystem
from utils.models import (
    AtomicFact,
    RetrievalItem,
    RetrievalResult,
    make_fact,
    make_chunk,          # used as a generic container for vector stores
    norm_text,
)
from utils.vector_adapters import VectorIndex, ChromaIndex, LanceDBIndex
from utils.custom_embedder import CustomProxyEmbedder
from utils.config import load_config


class VectorMemory(MemorySystem):
    def __init__(
        self,
        name: str = "vector_memory",
        *,
        embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
        index_backend: str = "chroma",         # "chroma" | "lancedb"
        collection_or_table: str = "memorybridge_facts",
        persist_path: str = ".memdb/chroma",   # for LanceDB, pass ".memdb/lancedb"
        restrict_to_conv: bool = True,
        index: Optional[VectorIndex] = None,
    ):
        """Initialize a fact-centric vector memory with an embedding fn and a persistent index."""
        super().__init__(name=name)

        # Embedding
        self._embed = embed_fn if embed_fn is not None else CustomProxyEmbedder().embed

        # Vector index
        if index is not None:
            self._index = index
            logger.debug(f"VectorMemory[{name}] using injected index: {type(index).__name__}")
        else:
            if index_backend.lower() == "chroma":
                self._index = ChromaIndex(
                    embed_fn=self._embed,
                    collection_name=collection_or_table,
                    persist_dir=persist_path,
                )
            elif index_backend.lower() == "lancedb":
                self._index = LanceDBIndex(
                    embed_fn=self._embed,
                    db_dir=persist_path,
                    table_name=collection_or_table,
                )
            else:
                raise ValueError(f"Unknown index_backend: {index_backend}")
            logger.info(
                f"VectorMemory[{name}] index initialized "
                f"(backend={index_backend}, store={collection_or_table}, path={persist_path})"
            )

        self.restrict_to_conv = bool(restrict_to_conv)

        # Local registries
        self._facts: Dict[str, AtomicFact] = {}           # fact_id -> AtomicFact
        self._slots: Dict[Tuple[str, str], str] = {}      # (subject, predicate) -> current fact_id
        self._conv_subject: Dict[str, str] = {}           # conv_id -> canonical subject alias (e.g., "Alice")

    # ---------------------------
    # LLM-based fact extractor (evidence comes from LLM via source_span)
    # ---------------------------
    def _extract_facts_llm(self, conv_id: str, text: str, *, source: str) -> List[AtomicFact]:
        """Call the LLM (OpenAI-compatible) to extract facts from raw text; return list[AtomicFact]."""
        cfg = load_config()
        base_url = cfg.get("OPENAI_BASE_URL", "").rstrip("/")
        api_key = cfg.get("OPENAI_API_KEY", "")
        model = cfg.get("MB_CHAT_MODEL", "gpt-4o-mini")

        if not api_key or not base_url:
            raise RuntimeError("Missing OPENAI_BASE_URL / OPENAI_API_KEY in .env for LLM extraction")

        client = OpenAI(api_key=api_key, base_url=base_url)

        # Multi-hop aware, canonical, general-purpose prompt (NO 'time' field).
        # Evidence MUST be provided as source_span: an exact substring from the user message that supports the fact.
        system_prompt = (
            "You are a precise information extraction engine. Read ONE user message and output ONLY a JSON array.\n"
            "Each array item is an atomic fact (up to 2 reasoning hops), with this schema:\n"
            "REQUIRED: subject (string), predicate (string), object (string), confidence (0.0-1.0), source_span (string).\n"
            "OPTIONAL: negated (bool), hypothetical (bool), derived_from (array<string>), id (string), qualifiers (object), slot (string).\n"
            "\n"
            "CONSTRAINTS & NORMALIZATION\n"
            "1) predicate: snake_case, lowercase, no spaces. Prefer conventional relations (e.g., works_at, lives_in, likes, "
            "switched_to, manager_of, founded, spouse_of). Map variants internally to a canonical label.\n"
            "2) SUBJECT RESOLUTION: If first-person appears (I/me/my/myself), then:\n"
            "   - If SUBJECT_ALIAS is provided in context, use that literal alias as subject.\n"
            "   - Otherwise use the literal token \"<USER>\" (do not guess names).\n"
            "   Resolve simple pronouns to prior entities within the same message when clear.\n"
            "3) MULTI-HOP: Derive short chains only when both hops are explicitly supported. Do not invent missing links.\n"
            "4) UPDATES & NEGATION: If the message clearly updates a (subject,predicate) with a new object, emit only the current fact. "
            "   If the message negates a prior state, set negated=true for that item; emit a new state only if stated.\n"
            "5) UNCERTAINTY: Mark hypothetical=true for tentative/planned/unsure statements.\n"
            "6) DEDUP: Deduplicate equivalent items; prefer the most informative form.\n"
            "7) ENTITIES: Keep entities concise and consistent (e.g., \"OpenAI\", \"Anthropic\", \"San Francisco\"). "
            "   Do not emit trivial tautologies like {\"subject\":\"Alice\",\"predicate\":\"name\",\"object\":\"Alice\"} unless newly revealed.\n"
            "8) EVIDENCE: source_span MUST be an exact, minimal substring copied verbatim from the user message that best supports the fact "
            "(e.g., \"My name is Alice\", \"I work at Anthropic now\"). No paraphrasing.\n"
            "9) OUTPUT: Return ONLY a JSON array of objects. No prose or code fences.\n"
        )

        # Pass current conversation alias for subject resolution
        subject_alias = self._conv_subject.get(conv_id)
        context_msg = {
            "role": "system",
            "content": f"SUBJECT_ALIAS: {subject_alias if subject_alias else '(none)'}"
        }

        logger.debug(f"[facts_llm] source={source} model={model} text_preview={text[:120]!r}")
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                context_msg,
                {"role": "user", "content": text},
            ],
            temperature=0.0,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        content = (resp.choices[0].message.content or "").strip()
        logger.debug(f"[facts_llm] raw_json_preview={content[:200]!r} latency_ms={latency_ms:.1f}")

        # Parse JSON
        try:
            parsed = json.loads(content)
        except Exception as e:
            logger.error(f"[facts_llm] JSON parse failed: {e}; content_preview={content[:200]!r}")
            return []

        if not isinstance(parsed, list):
            logger.debug("[facts_llm] Parsed content is not a list; returning empty.")
            return []

        # Build AtomicFacts; tiny guardrails (predicate map + alias rewrite). Evidence comes only from LLM.
        out: List[AtomicFact] = []
        for idx, d in enumerate(parsed):
            try:
                s = d.get("subject")
                p = d.get("predicate")
                o = d.get("object")
                span = (d.get("source_span") or "").strip()
                if not (s and p and o and span):
                    # We strictly require evidence from LLM per user request
                    logger.debug(f"[facts_llm] skipping item without required fields or source_span: {d!r}")
                    continue

                conf = float(d.get("confidence", 0.85))
                p = self._canon_pred(p)  # tiny fallback if the model drifts

                # Map "<USER>"/pronouns to alias if we have one (fallback)
                if s and norm_text(s).lower() in {"<user>", "i", "me", "my", "myself", "user"}:
                    alias = self._conv_subject.get(conv_id)
                    if alias:
                        s = alias

                f = make_fact(s, p, o, source=source, confidence=conf)
                f.meta["evidence"] = span  # evidence from LLM

                logger.info(f"[facts_llm] fact EXTRACTED id={f.fact_id} ({f.subject}|{f.predicate}|{f.object}) conf={f.confidence:.2f}")
                out.append(f)

                # If name revealed, update alias for this conversation
                if f.predicate == "name":
                    alias = f.object.strip()
                    if alias:
                        self._conv_subject[conv_id] = alias
                        logger.debug(f"[facts_llm] conv_alias set: conv={conv_id} alias={alias!r}")

            except Exception as ex:
                logger.debug(f"[facts_llm] Skipping malformed item at index {idx}: {d!r} err={ex}")

        logger.info(f"[facts_llm] total_extracted={len(out)} source={source}")
        return out

    # ---------------------------
    # Turn ingestion (raw text -> LLM facts -> slot -> embed+store)
    # ---------------------------
    def add_turn(self, conv_id: str, turn_id: int, role: str, text: Any) -> None:
        """Accept raw turn text (user) or a facts payload; extract (if needed), slot, then embed + upsert."""
        source = f"{conv_id}:turn_{turn_id}"
        logger.debug(f"[add_turn] conv={conv_id} turn={turn_id} role={role} type={type(text).__name__}")

        facts: List[AtomicFact] = []  # ensure defined

        # Raw-text path (preferred)
        if isinstance(text, str):
            raw = norm_text(text)
            if not raw:
                logger.debug(f"[add_turn] empty text; skipping. source={source}")
                return
            if role.lower() != "user":
                logger.debug(f"[add_turn] non-user role ignored (role={role}). source={source}")
                return
            try:
                facts = self._extract_facts_llm(conv_id, raw, source=source)
            except Exception as e:
                logger.warning(f"[add_turn] LLM extraction failed: {e}; source={source}")
                facts = []
        else:
            # Structured-facts path (back-compat); NOTE: here we cannot guarantee evidence
            facts = self._coerce_to_facts(text, source=source)

        if not facts:
            logger.info(f"[add_turn] no facts to index. source={source}")
            return

        # Slotting: keep only the latest fact per (subject, predicate)
        chunks = []
        for f in facts:
            slot_key = (norm_text(f.subject).lower(), norm_text(f.predicate).lower())
            prev_id = self._slots.get(slot_key)

            if prev_id and prev_id in self._facts and prev_id != f.fact_id:
                prev_fact = self._facts[prev_id]
                if prev_fact.active:
                    prev_fact.active = False
                    logger.info(
                        f"[add_turn] fact DEACTIVATED id={prev_fact.fact_id} "
                        f"({prev_fact.subject}|{prev_fact.predicate}|{prev_fact.object})"
                    )
                f.supersedes = prev_fact.fact_id
                logger.info(
                    f"[add_turn] fact UPDATED id={f.fact_id} supersedes={prev_fact.fact_id} "
                    f"({f.subject}|{f.predicate}|{f.object}) conf={f.confidence:.2f}"
                )
            else:
                logger.info(
                    f"[add_turn] fact ADDED id={f.fact_id} "
                    f"({f.subject}|{f.predicate}|{f.object}) conf={f.confidence:.2f}"
                )

            # Save/replace in registries
            self._facts[f.fact_id] = f
            self._slots[slot_key] = f.fact_id

            # Convert to chunk with metadata (include evidence from LLM if present)
            fact_text = self._fact_text(f)
            ch = make_chunk(
                fact_text,
                source=source,
                meta={
                    "type": "fact",
                    "conv_id": conv_id,
                    "fact_id": f.fact_id,
                    "subject": f.subject,
                    "predicate": f.predicate,
                    "object": f.object,
                    "confidence": f.confidence,
                    "created_at": f.created_at,
                    "active": f.active,
                    "supersedes": f.supersedes,
                    "evidence": f.meta.get("evidence"),
                },
            )
            chunks.append(ch)
            logger.debug(f"[add_turn] chunk created id={ch.chunk_id} text={fact_text!r}")

        # Upsert all chunks into the vector index
        self._index.upsert(chunks)
        logger.info(f"[add_turn] upserted_chunks={len(chunks)} conv={conv_id} turn={turn_id}")

    # ---------------------------
    # Retrieval (current facts only)
    # ---------------------------
    def retrieve(self, conv_id: str, query_text: str, top_k: int = 5, **kwargs) -> Dict[str, Any]:
        """Embed query and return top-K nearest *current* (active) facts, optionally restricted to this conversation."""
        t0 = time.perf_counter()
        query_text = norm_text(query_text)
        logger.debug(f"[retrieve] conv={conv_id} top_k={top_k} query_preview={query_text[:120]!r}")

        if not query_text:
            logger.info("[retrieve] empty query; returning no results.")
            return RetrievalResult(query=query_text, results=[]).to_dict()

        # Over-fetch to allow filtering of inactive or cross-conv items
        hits = self._index.search(query_text, k=max(10, top_k * 5))
        logger.debug(f"[retrieve] raw_hits={len(hits)}")

        items: List[RetrievalItem] = []
        for ch, sim in hits:
            meta = ch.meta or {}
            if meta.get("type") != "fact":
                continue

            fid = meta.get("fact_id")
            f = self._facts.get(fid or "")

            # Skip if not in registry (shouldn't happen) or not active
            if not f or f.active is False:
                continue

            # Respect conversation scoping
            if self.restrict_to_conv and meta.get("conv_id") != conv_id:
                continue

            evidence = f.meta.get("evidence") or meta.get("evidence")

            it = RetrievalItem(
                type="fact",
                id=f.fact_id,
                subject=f.subject,
                predicate=f.predicate,
                object=f.object,
                source=ch.source,
                created_at=f.created_at,
                score=round(float(sim), 6),
                text=evidence,  # expose LLM-provided evidence phrase
            )
            items.append(it)
            if len(items) >= top_k:
                break

        latency_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(f"[retrieve] results={len(items)} conv={conv_id} latency_ms={latency_ms:.1f}")
        for it in items:
            reg = self._facts.get(it.id or "")
            logger.debug(
                f"[retrieve] ({it.subject} | {it.predicate} | {it.object}) "
                f"score={it.score:.4f} active={getattr(reg,'active',None)} supersedes={getattr(reg,'supersedes',None)} "
                f"evidence={it.text!r}"
            )

        return RetrievalResult(query=query_text, results=items).to_dict()

    # ---------------------------
    # Helpers
    # ---------------------------
    @staticmethod
    def _fact_text(f: AtomicFact) -> str:
        """Return canonical text used for embedding a fact."""
        return f"{f.subject} | {f.predicate} | {f.object}"

    @staticmethod
    def _coerce_to_facts(payload: Any, *, source: str) -> List[AtomicFact]:
        """Convert supported payloads (list[AtomicFact] | JSON | list[dict]) into list[AtomicFact]."""
        from utils.models import AtomicFact as AF, make_fact as mkf  # local alias

        # 1) Already a list of AtomicFact
        if isinstance(payload, list) and payload and isinstance(payload[0], AF):
            return payload

        # 2) JSON string
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                return []

        # 3) List of dicts
        if isinstance(payload, list) and all(isinstance(x, dict) for x in payload):
            facts: List[AF] = []
            for d in payload:
                s = d.get("subject")
                p = d.get("predicate")
                o = d.get("object")
                if not (s and p and o):
                    continue
                f = mkf(
                    s, p, o,
                    source=source,
                    confidence=float(d.get("confidence", 0.85)),
                    fact_id=d.get("fact_id"),
                    created_at=d.get("created_at"),
                    meta={k: v for k, v in d.items() if k not in {
                        "subject", "predicate", "object", "confidence", "fact_id", "created_at"
                    }},
                )
                facts.append(f)
            return facts

        # Anything else → no facts
        return []

    @staticmethod
    def _canon_pred(p: str) -> str:
        """Tiny fallback to normalize common predicate variants to canonical snake_case."""
        p = norm_text(p).lower().replace(" ", "_")
        mapping = {
            "has_name": "name",  # alias per your request
            "work_at": "works_at", "works_at": "works_at", "employed_at": "works_at",
            "live_in": "lives_in", "lives_in": "lives_in", "reside_in": "lives_in",
            "love": "likes", "likes": "likes", "enjoy": "likes",
            "switched_to": "switched_to", "change_to": "switched_to", "changed_to": "switched_to",
            "name": "name", "is_named": "name", "is_called": "name",
            # useful extras:
            "manages": "manager_of", "manager_of": "manager_of",
            "founded": "founded", "spouse_of": "spouse_of",
        }
        return mapping.get(p, p)

    # ---------------------------
    # Lifecycle
    # ---------------------------
    def reset(self) -> None:
        """Clear the underlying index and local registries."""
        try:
            self._index.clear()
            logger.info("[reset] index cleared")
        except Exception as e:
            logger.warning(f"[reset] index clear failed: {e}")
        self._facts.clear()
        self._slots.clear()
        self._conv_subject.clear()
        logger.debug("[reset] local facts, slots, conv_subject cleared")
