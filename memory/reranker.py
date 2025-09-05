from typing import List, Dict, Any
import os, json
from openai import OpenAI
from .config import BASE_URL, MB_API_KEY, CHAT_MODEL


def _prep_facts(hits: List[Dict[str, Any]]) -> str:
    lines = []
    for i, h in enumerate(hits, start=1):
        txt = (h.get("content") or h.get("memory") or "").strip().replace("\n", " ")
        lines.append(f"{i}. {txt or '(blank)'}")
    return "\n".join(lines)


def _parse_scores(s: str) -> dict:
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        data = json.loads(s[start:end+1])
        return {int(x["index"]): float(x["score"]) for x in data.get("scores", [])}
    except Exception:
        return {}


def rerank_with_llm(query: str, hits: List[Dict[str, Any]], top_n: int = 3) -> List[Dict[str, Any]]:
    if not hits:
        return []
    client = OpenAI(base_url=BASE_URL, api_key=MB_API_KEY)
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "Score each fact 0-100 for direct relevance; return JSON: {\"scores\": [{\"index\": i, \"score\": s}]."},
            {"role": "user", "content": f"Question: {query}\nFacts:\n{_prep_facts(hits)}"},
        ],
        temperature=0,
    )
    idx2 = _parse_scores(resp.choices[0].message.content or "")
    if not idx2:
        return hits[:top_n]
    enriched = []
    for i, h in enumerate(hits, start=1):
        hh = dict(h)
        hh["llm_score"] = idx2.get(i, 0.0)
        enriched.append(hh)
    enriched.sort(key=lambda x: x.get("llm_score", 0.0), reverse=True)
    return enriched[:top_n]