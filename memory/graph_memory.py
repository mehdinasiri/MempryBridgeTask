# memory/graph_memory.py
from __future__ import annotations

import os
import re
import time
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from collections import defaultdict

from loguru import logger

# ---- deps ----
try:
    import networkx as nx
except Exception:
    nx = None

try:
    # OpenAI-compatible client; works with your proxy via OPENAI_BASE_URL/OPENAI_API_KEY
    from openai import OpenAI
except Exception:
    OpenAI = None

# ---- your models ----
from utils.models import AtomicFact, make_fact, norm_text  # adjust import path if needed

# ---------------- utils ----------------

_JSON_ARRAY_RE = re.compile(r"(?s)```(?:json)?\s*(\[[\s\S]*?\])\s*```|(\[[\s\S]*\])")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _now() -> float:
    return time.time()


def _norm(s: str) -> str:
    s = (s or "").strip()
    return re.sub(r"\s+", " ", s)


def _snake_like(s: str) -> str:
    s = _norm(s)
    s = re.sub(r"[\s\-]+", "_", s.lower())
    s = re.sub(r"[^a-z0-9_]", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _tokens(s: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(s or "")]


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    A, B = set(a), set(b)
    if not A and not B:
        return 0.0
    return len(A & B) / max(1, len(A | B))


def _half_life_weight(age_sec: Optional[float], half_life_sec: float = 7 * 24 * 3600.0) -> float:
    if age_sec is None:
        return 1.0
    return 0.5 ** (max(0.0, age_sec) / max(1.0, half_life_sec))


def _safe_load_json_array(s: str) -> Optional[List[Any]]:
    if not s:
        return None
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            arr = json.loads(s)
            return arr if isinstance(arr, list) else None
        except Exception:
            pass
    m = _JSON_ARRAY_RE.search(s)
    if not m:
        return None
    candidate = (m.group(1) or m.group(2) or "").strip()
    try:
        arr = json.loads(candidate)
        return arr if isinstance(arr, list) else None
    except Exception:
        return None


# -------------- main class --------------

class GraphMemory:
    """
    NetworkX-based graph memory with alias canonicalization & retroactive relabeling.
      • Extracts SPO facts via strict prompt (supports up to 2-hop explicit facts).
      • Slots latest state per (conv_id, subject, predicate).
      • Persists active facts as edges in per-conversation MultiDiGraph (keyed by fact_id).
      • Retrieves via k-hop frontier + blended scoring.
      • Optional: extract a "query fact" first to steer seeds & scoring.
    """

    def __init__(
        self,
        name: str = "graph_memory_nx",
        *,
        restrict_to_conv: bool = True,
        max_hops: int = 2,
        max_expand_edges: int = 512,
        openai_base_url: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        chat_model: Optional[str] = None,
        enable_query_fact_extraction: bool = True,
    ):
        if nx is None:
            raise ImportError("networkx is not installed. Please `pip install networkx`.")
        self.name = name
        self.restrict_to_conv = bool(restrict_to_conv)
        self.max_hops = int(max_hops)
        self.max_expand_edges = int(max_expand_edges)
        self.enable_query_fact_extraction = bool(enable_query_fact_extraction)

        # registries
        self._facts: Dict[str, AtomicFact] = {}                        # fact_id -> fact
        self._slots: Dict[Tuple[str, str, str], str] = {}              # (conv_id, subj_l, pred_l) -> current fact_id
        self._conv_alias: Dict[str, str] = {}                          # conv -> alias (from predicate=name)
        self._entity_timeline: Dict[str, Dict[str, int]] = defaultdict(dict)  # conv -> ent_lc -> last_turn

        # per-conversation graphs
        self._G: Dict[str, "nx.MultiDiGraph"] = defaultdict(nx.MultiDiGraph)

        # LLM params
        self._base_url = (openai_base_url or os.getenv("OPENAI_BASE_URL", "")).rstrip("/")
        self._api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
        self._chat_model = chat_model or os.getenv("MB_CHAT_MODEL", "gpt-4o-mini")

        if not self._api_key or not self._base_url:
            logger.warning("GraphMemory: OPENAI_BASE_URL / OPENAI_API_KEY not set; extraction will fail.")

        logger.info(f"GraphMemory[{self.name}] ready (restrict_to_conv={self.restrict_to_conv}, hops={self.max_hops})")

    # ---------------- ingestion ----------------

    def add_turn(self, conv_id: str, turn_id: int, role: str, text: Any) -> None:
        """Ingest a USER turn: LLM extract → coref/alias repair → slot → graph update (with retroactive relabel)."""
        source = f"{conv_id}:turn_{turn_id}"
        logger.debug(f"[graph.add_turn] conv={conv_id} turn={turn_id} role={role} type={type(text).__name__}")

        if (role or "").lower() != "user" or not isinstance(text, str):
            logger.debug(f"[graph.add_turn] skip (empty or non-user). source={source}")
            return
        msg = _norm(text)
        if not msg:
            logger.debug(f"[graph.add_turn] empty text; skip. source={source}")
            return

        facts = self._extract_facts_llm(conv_id, msg, source=source)
        if not facts:
            logger.info(f"[graph.add_turn] no facts to index. source={source}")
            return

        # Canonicalize / repair subjects first (pronouns, <USER>, vague heads)
        facts = self._coref_repair(conv_id, facts, msg, turn_id)

        # Slot + graph insert; alias adoption + retroactive relabel when we encounter a name
        for f in facts:
            # alias adoption
            if f.predicate.strip().lower() == "name" and f.object.strip():
                self._adopt_alias_and_relabel(conv_id, f.object.strip())

            # after alias step, ensure subject is canon again (may switch <USER> → alias)
            f.subject = self._canon_subject(conv_id, f.subject)

            # slotting (latest per (conv, subj, pred))
            slot_key = (conv_id, f.subject.lower(), f.predicate.lower())
            prev_id = self._slots.get(slot_key)
            if prev_id and prev_id in self._facts and prev_id != f.fact_id:
                prev = self._facts[prev_id]
                if prev.active:
                    prev.active = False
                    # graph: remove previous edge
                    self._remove_edge_by_fact(conv_id, prev.fact_id)
                    logger.info(f"[graph.add_turn] DEACTIVATED {prev.subject}|{prev.predicate}|{prev.object}")
                # record lineage (optional; does not affect scoring)
                f.supersedes = prev.fact_id
                logger.info(
                    f"[graph.add_turn] UPDATED -> {f.subject}|{f.predicate}|{f.object} "
                    f"(supersedes={prev.fact_id})"
                )
            else:
                logger.info(f"[graph.add_turn] ADDED -> {f.subject}|{f.predicate}|{f.object}")

            # persist in registries + graph
            self._facts[f.fact_id] = f
            self._slots[slot_key] = f.fact_id
            self._insert_edge(conv_id, f)

            # timeline for recency-based nudges
            self._entity_timeline[conv_id][f.subject.lower()] = turn_id
            self._entity_timeline[conv_id][f.object.lower()] = turn_id

    # ---------------- retrieval ----------------

    def retrieve(self, conv_id: str, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        query = _norm(query_text)
        if not query or conv_id not in self._G:
            return {"query": query, "results": []}

        # Optional query-fact extraction to guide seeds & scoring
        q_fact = None
        if self.enable_query_fact_extraction:
            q_fact = self._extract_query_fact(conv_id, query)

        seeds = self._seeds_from_query(conv_id, query, q_fact=q_fact)
        logger.debug(f"[graph.retrieve] conv={conv_id} seeds={seeds}")
        if not seeds:
            return {"query": query, "results": []}

        G = self._G[conv_id]
        candidates: Dict[str, float] = {}
        expansions = 0

        # Use undirected view for hop-frontier; score with original directed edges.
        for s in seeds:
            if s not in G:
                continue
            layers = nx.single_source_shortest_path_length(G.to_undirected(as_view=True), s, cutoff=self.max_hops)
            for node, dist in layers.items():
                # out edges
                for _, v, key, data in G.out_edges(node, keys=True, data=True):
                    if expansions >= self.max_expand_edges:
                        break
                    f = self._facts.get(data.get("fact_id", ""))
                    if not f or not f.active:
                        continue
                    score = self._edge_path_score(query, f, hop_dist=dist, q_fact=q_fact)
                    fid = f.fact_id
                    if score > candidates.get(fid, 0.0):
                        candidates[fid] = score
                    expansions += 1
                if expansions >= self.max_expand_edges:
                    break
                # in edges
                for u, _, key, data in G.in_edges(node, keys=True, data=True):
                    if expansions >= self.max_expand_edges:
                        break
                    f = self._facts.get(data.get("fact_id", ""))
                    if not f or not f.active:
                        continue
                    score = self._edge_path_score(query, f, hop_dist=dist, q_fact=q_fact)
                    fid = f.fact_id
                    if score > candidates.get(fid, 0.0):
                        candidates[fid] = score
                    expansions += 1
            if expansions >= self.max_expand_edges:
                break

        if not candidates:
            return {"query": query, "results": []}

        # Re-rank with blended scorer
        qtok = _tokens(query)
        seed_lc = {s.lower() for s in seeds}
        ranked: List[Tuple[float, AtomicFact]] = []
        now = _now()
        for fid, path_rel in candidates.items():
            f = self._facts[fid]
            lex = _jaccard(qtok, _tokens(f"{f.subject} {f.predicate} {f.object}"))
            # Age (AtomicFact.created_at is ISO string in your models) → safe ignore if parse is needed; use rec = 1.0
            rec = 1.0
            touches = 1.0 if (f.subject.lower() in seed_lc or f.object.lower() in seed_lc) else 0.0
            qboost = 0.0
            if q_fact:
                if q_fact.get("predicate") and _jaccard(_tokens(q_fact["predicate"]), _tokens(f.predicate)) >= 0.5:
                    qboost += 0.05
                if q_fact.get("object") and _jaccard(_tokens(q_fact["object"]), _tokens(f.object)) >= 0.5:
                    qboost += 0.05
            score = 0.45 * path_rel + 0.30 * lex + 0.20 * rec + 0.05 * touches + qboost
            ranked.append((score, f))

        ranked.sort(key=lambda t: t[0], reverse=True)

        out: List[Dict[str, Any]] = []
        for score, f in ranked[:max(1, top_k)]:
            out.append({
                "type": "fact",
                "id": f.fact_id,
                "subject": f.subject,
                "predicate": f.predicate,
                "object": f.object,
                "source": f.source,
                "created_at": f.created_at,
                "score": float(score),
                "text": (f.meta or {}).get("evidence"),
            })
        return {"query": query, "results": out}

    # ---------------- graph ops ----------------

    def _insert_edge(self, conv_id: str, f: AtomicFact) -> None:
        G = self._G[conv_id]
        if f.subject not in G:
            G.add_node(f.subject)
        if f.object not in G:
            G.add_node(f.object)
        G.add_edge(
            f.subject, f.object,
            key=f.fact_id,
            fact_id=f.fact_id,
            predicate=f.predicate,
            confidence=float(f.confidence),
            evidence=(f.meta or {}).get("evidence"),
            active=bool(f.active),
            source=f.source,
        )

    def _remove_edge_by_fact(self, conv_id: str, fact_id: str) -> None:
        G = self._G[conv_id]
        for u, v, k in list(G.edges(keys=True)):
            if k == fact_id:
                G.remove_edge(u, v, key=k)

    # ---------------- scoring & seeds ----------------

    def _edge_path_score(self, query: str, f: AtomicFact, hop_dist: int, q_fact: Optional[Dict[str, str]]) -> float:
        """Label-agnostic relevance at a frontier hop distance, optionally guided by extracted query fact."""
        pred_sim = _jaccard(_tokens(query), _tokens(f.predicate))
        conf = float(f.confidence)
        hop_pen = 1.0 / (1.0 + 0.3 * max(0, hop_dist))  # gentle decay: 1.0, ~0.77, ~0.62 for hops 0,1,2
        guided = 0.0
        if q_fact:
            if q_fact.get("subject") and _jaccard(_tokens(q_fact["subject"]), _tokens(f.subject)) >= 0.5:
                guided += 0.1
            if q_fact.get("object") and _jaccard(_tokens(q_fact["object"]), _tokens(f.object)) >= 0.5:
                guided += 0.1
            if q_fact.get("predicate") and _jaccard(_tokens(q_fact["predicate"]), _tokens(f.predicate)) >= 0.5:
                guided += 0.1
        return (0.55 * pred_sim + 0.45 * conf) * hop_pen + guided

    def _seeds_from_query(self, conv_id: str, query: str, *, q_fact: Optional[Dict[str, str]] = None) -> List[str]:
        """Alias + robust TitleCase spans (handles possessives) + node overlap + query-fact hints."""
        seeds: List[str] = []
        alias = self._conv_alias.get(conv_id)
        if alias:
            seeds.append(alias)

        # Normalize possessives
        raw_tokens = query.split()
        cleaned: List[str] = []
        for tok in raw_tokens:
            t = tok.strip().strip('?:!"“”.,;()[]{}')
            t = re.sub(r"[’]", "'", t)
            if t.lower().endswith("'s"):
                t = t[:-2]
            if t:
                cleaned.append(t)
        cleaned_query = " ".join(cleaned)

        # TitleCase spans and tokens
        toks = cleaned_query.split()
        span: List[str] = []
        for t in toks:
            tt = t.strip().strip('?:!"“”.,;()[]{}')
            if tt and tt[0].isupper():
                span.append(tt)
            else:
                if span:
                    seeds.append(" ".join(span))
                span = []
        if span:
            seeds.append(" ".join(span))
        for t in toks:
            if t and t[0].isupper():
                seeds.append(t)

        # Overlap with known nodes
        qtok = set(_tokens(cleaned_query))
        G = self._G.get(conv_id)
        if G is not None:
            for node in G.nodes():
                ntok = set(_tokens(node))
                if not ntok:
                    continue
                if ntok.issubset(qtok) or (len(ntok & qtok) / len(ntok) >= 0.6):
                    seeds.append(node)
                elif len(ntok) == 1 and list(ntok)[0] in qtok:
                    seeds.append(node)

        # Query fact hints
        if q_fact:
            if q_fact.get("subject"):
                seeds.append(q_fact["subject"])
            if q_fact.get("object"):
                seeds.append(q_fact["object"])

        # dedupe case-insensitively
        out, seen = [], set()
        for s in seeds:
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
        return out[:32]

    # ---------------- aliasing & coref ----------------

    def _canon_subject(self, conv_id: str, subj: str) -> str:
        """Map pronouns/<USER> to alias if known; preserve cased alias."""
        alias = self._conv_alias.get(conv_id)
        low = (subj or "").strip().lower()
        if low in {"<user>", "i", "me", "my", "myself", "user"}:
            return alias or "<USER>"
        if alias and low == alias.lower():
            return alias
        return subj

    def _adopt_alias_and_relabel(self, conv_id: str, alias: str) -> None:
        """After we learn the user's name, rewrite prior <USER> → alias for this conversation."""
        alias = alias.strip()
        if not alias:
            return

        # If alias already set to the same, nothing to do
        if self._conv_alias.get(conv_id, "").strip().lower() == alias.lower():
            return

        self._conv_alias[conv_id] = alias

        # Relabel existing facts/edges whose SUBJECT is <USER>
        to_relabel: List[str] = []
        for fid, f in self._facts.items():
            if (f.source or "").split(":")[0] != conv_id:
                continue
            if (f.subject or "").strip().lower() == "<user>":
                to_relabel.append(fid)

        if not to_relabel:
            return

        G = self._G[conv_id]
        for fid in to_relabel:
            f = self._facts[fid]

            # Remove old slot entry
            old_slot = (conv_id, f.subject.lower(), f.predicate.lower())
            if self._slots.get(old_slot) == fid:
                del self._slots[old_slot]

            # Remove old edge
            self._remove_edge_by_fact(conv_id, fid)

            # Rewrite subject to alias
            f.subject = alias

            # Re-slot under new (subj,pred)
            new_slot = (conv_id, f.subject.lower(), f.predicate.lower())
            self._slots[new_slot] = fid

            # Re-insert edge with new subject node
            self._insert_edge(conv_id, f)

        # Timeline consistency
        tl = self._entity_timeline.get(conv_id, {})
        if "<user>" in tl:
            tl[alias.lower()] = tl.pop("<user>")
        self._entity_timeline[conv_id] = tl

    def _coref_repair(self, conv_id: str, facts: List[AtomicFact], raw_text: str, turn_id: int) -> List[AtomicFact]:
        """General, label-free subject repair using alias + recency (+ minimal overlap for vague heads)."""
        if not facts:
            return facts

        pronouns = {"he","him","his","she","her","hers","they","them","their","theirs","it","its","we","us","our","ours","i","me","my","myself","user","<user>"}
        vague_cues = {"office","team","department","hq","headquarters"}

        timeline = self._entity_timeline.get(conv_id, {})
        recent = sorted(timeline.items(), key=lambda kv: kv[1], reverse=True)
        rq = set(_tokens(raw_text))

        def _recover_entity_cased(ent_lc: str) -> str:
            # try to find original casing from known facts in this conv
            for f in self._facts.values():
                if (f.source or "").split(":")[0] != conv_id:
                    continue
                if f.subject.lower() == ent_lc:
                    return f.subject
                if f.object.lower() == ent_lc:
                    return f.object
            return ent_lc

        def pick_recent(overlap_needed: bool) -> Optional[str]:
            for ent_lc, _t in recent:
                ent = _recover_entity_cased(ent_lc)
                if not overlap_needed:
                    return ent
                if _jaccard(rq, _tokens(ent)) >= 0.2:
                    return ent
            return None

        repaired: List[AtomicFact] = []
        for f in facts:
            s_low = (f.subject or "").strip().lower()
            new_s = f.subject
            if s_low in pronouns or s_low == "<user>":
                # first prefer canonical alias
                alias = self._conv_alias.get(conv_id)
                if alias:
                    new_s = alias
                else:
                    cand = pick_recent(overlap_needed=False)
                    if cand:
                        new_s = cand
                    else:
                        new_s = "<USER>"
            elif any(cue in s_low for cue in vague_cues):
                cand = pick_recent(overlap_needed=True)
                if cand:
                    new_s = cand
            # finally apply canonicalizer
            new_s = self._canon_subject(conv_id, new_s)
            if new_s != f.subject:
                f.subject = new_s
            repaired.append(f)

        # update timeline
        for f in repaired:
            timeline[f.subject.lower()] = turn_id
            timeline[f.object.lower()] = turn_id
        self._entity_timeline[conv_id] = timeline
        return repaired

    # ---------------- extraction ----------------

    def _extract_facts_llm(self, conv_id: str, text: str, *, source: str) -> List[AtomicFact]:
        """Use strict prompt to parse one user message into SPO facts (returns AtomicFact list)."""
        if OpenAI is None:
            logger.error("openai package not installed.")
            return []
        if not self._api_key or not self._base_url:
            logger.error("Missing OPENAI_BASE_URL / OPENAI_API_KEY.")
            return []

        client = OpenAI(api_key=self._api_key, base_url=self._base_url)

        system_prompt = (
            "You are a precise information extraction engine. Read ONE user message and output ONLY a JSON array.\n"
            "Each array item is an atomic fact (allow up to 2 reasoning hops), with this schema:\n"
            "REQUIRED fields: subject (string), predicate (string), object (string), confidence (0..1), source_span (string).\n"
            "OPTIONAL fields: negated (bool), hypothetical (bool), derived_from (array<string>), id (string), "
            "qualifiers (object), slot (string).\n"
            "\n"
            "CONSTRAINTS & NORMALIZATION\n"
            "1) predicate: snake_case, lowercase, concise. Use conventional labels: works_at, lives_in, likes, switched_to, "
            "manager_of, founded, spouse_of, name (not has_name). Map variants internally to a canonical label.\n"
            "2) SUBJECT RESOLUTION: If first-person appears (I/me/my/myself), then:\n"
            "   - If SUBJECT_ALIAS is provided in context, use that literal alias as subject.\n"
            "   - Otherwise use the literal token \"<USER>\" (do not guess names).\n"
            "   Resolve simple pronouns within the same message when unambiguous.\n"
            "3) MULTI-HOP: Emit short chains only when both hops are explicitly supported by the message; do not invent links.\n"
            "4) UPDATES/NEGATION: If the message clearly updates a (subject,predicate) slot with a new object, emit only the current state. "
            "   If it negates a prior state, set negated=true; emit a new positive state only if explicitly stated.\n"
            "5) UNCERTAINTY: Mark hypothetical=true for tentative/planned/unsure statements.\n"
            "6) DEDUP: Remove equivalent items; keep the most informative.\n"
            "7) ENTITIES: Keep concise canonical strings (e.g., \"OpenAI\", \"Anthropic\", \"San Francisco\"). "
            "   Avoid trivial tautologies unless they are newly revealed and useful.\n"
            "8) EVIDENCE: source_span MUST be an exact, minimal substring copied verbatim from the user message that best supports the fact "
            "(e.g., \"My name is Alice\", \"I work at Anthropic now\"). No paraphrasing.\n"
            "9) OUTPUT: Return ONLY a JSON array of objects. No prose, no code fences, no surrounding text.\n"
        )

        alias = self._conv_alias.get(conv_id)
        context = f"SUBJECT_ALIAS: {alias if alias else '(none)'}"

        logger.debug(f"[graph.extract] source={source} model={self._chat_model} preview={text[:120]!r}")
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=self._chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": context},
                {"role": "user", "content": text},
            ],
            temperature=0.0,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        content = (resp.choices[0].message.content or "").strip()
        logger.debug(f"[graph.extract] raw_json_preview={content[:200]!r} latency_ms={latency_ms:.1f}")

        arr = _safe_load_json_array(content)
        if not isinstance(arr, list):
            return []

        out: List[AtomicFact] = []
        for d in arr:
            try:
                s = _norm(d.get("subject") or "")
                p = _snake_like(d.get("predicate") or "")
                o = _norm(d.get("object") or "")
                span = _norm(d.get("source_span") or "")
                if not (s and p and o and span):
                    continue
                conf = float(d.get("confidence", 0.85))
                f = make_fact(
                    s, p, o,
                    source=source,
                    confidence=conf,
                    meta={"evidence": span},
                )
                out.append(f)
                logger.info(f"[graph.extract] fact {f.fact_id} ({f.subject}|{f.predicate}|{f.object}) conf={conf:.2f}")
            except Exception:
                pass
        return out

    def _extract_query_fact(self, conv_id: str, query: str) -> Optional[Dict[str, str]]:
        """Lightweight extraction on the question to guide seeds & scoring. Returns {subject?, predicate?, object?}."""
        if OpenAI is None or not self._api_key or not self._base_url:
            return None
        client = OpenAI(api_key=self._api_key, base_url=self._base_url)

        system_prompt = (
            "Read ONE question and output ONLY a JSON array with 0 or 1 items, same schema as before: "
            "subject, predicate (snake_case), object (optional), confidence (0..1), source_span. "
            "If the question is vague, return []. No prose."
        )
        alias = self._conv_alias.get(conv_id)
        context = f"SUBJECT_ALIAS: {alias if alias else '(none)'}"

        try:
            resp = client.chat.completions.create(
                model=self._chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "system", "content": context},
                    {"role": "user", "content": query},
                ],
                temperature=0.0,
            )
            content = (resp.choices[0].message.content or "").strip()
            arr = _safe_load_json_array(content) or []
            if not arr:
                return None
            d = arr[0]
            subj = _norm(d.get("subject") or "")
            pred = _snake_like(d.get("predicate") or "")
            obj = _norm(d.get("object") or "")
            return {
                "subject": subj or "",
                "predicate": pred or "",
                "object": obj or "",
            }
        except Exception:
            return None

    # ---------------- lifecycle ----------------

    def reset(self) -> None:
        self._facts.clear()
        self._slots.clear()
        self._conv_alias.clear()
        self._entity_timeline.clear()
        self._G.clear()
        logger.debug("[graph.reset] cleared all stores")
