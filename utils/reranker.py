# repo/utils/reranker.py
from __future__ import annotations

import json
from typing import List, Any

from utils.config import SETTINGS

# Optional dependency: OpenAI python client
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


_SYS_PROMPT = (
    "You are a ranking model. For each candidate snippet, assign a relevance score "
    "for answering the user's query. Return STRICT JSON in the form:\n"
    '{"scores":[{"index":1,"score":<0-100>},{"index":2,"score":<0-100>}, ...]}.\n'
    "Only output JSON. No explanations."
)


def _prep_items(texts: List[str]) -> str:
    lines = []
    for i, t in enumerate(texts, start=1):
        cleaned = (t or "").strip().replace("\n", " ")
        lines.append(f"{i}. {cleaned}")
    return "\n".join(lines)


def _parse_scores(s: str, n: int) -> List[float]:
    try:
        start, end = s.find("{"), s.rfind("}")
        data = json.loads(s[start : end + 1])
        mapping = {int(x["index"]): float(x["score"]) for x in data.get("scores", [])}
        return [float(mapping.get(i + 1, 0.0)) for i in range(n)]
    except Exception:
        return [0.0] * n


def _keyword_fallback(query: str, texts: List[str], base_scores: List[float]) -> List[float]:
    """If LLM is unavailable, fall back to a simple lexical overlap booster."""
    q = set(query.lower().split())
    outs = []
    for t, b in zip(texts, base_scores):
        toks = set((t or "").lower().split())
        jacc = len(q & toks) / max(len(q | toks), 1)
        # Blend 80% original + 20% overlap (scaled to 100)
        outs.append(0.8 * b * 100.0 + 0.2 * jacc * 100.0)
    return outs


def rerank(query: str, texts: List[str], scores: List[float]) -> List[float]:
    """
    Return a new list of scores (higher = more relevant), same length/order as `texts`.
    If OpenAI client/env is not available, uses a cheap lexical fallback.
    """
    n = len(texts)
    if n == 0:
        return []

    # Fallback if client not available
    if OpenAI is None or not SETTINGS.api_key:
        return _keyword_fallback(query, texts, scores)

    client = OpenAI(base_url=(SETTINGS.base_url or None), api_key=SETTINGS.api_key)

    user_prompt = f"Query: {query}\nCandidates:\n{_prep_items(texts)}"
    try:
        resp = client.chat.completions.create(
            model=SETTINGS.chat_model,  # e.g., "gemini-flash" via OpenAI-compatible gateway
            messages=[
                {"role": "system", "content": _SYS_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        content = resp.choices[0].message.content or ""
        llm_scores = _parse_scores(content, n)
        # Normalize to 0..1-ish before returning; vector_memory will sort by magnitude anyway
        return [float(s) for s in llm_scores]
    except Exception:
        # Robust fallback
        return _keyword_fallback(query, texts, scores)
