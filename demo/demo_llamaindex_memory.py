# demo/demo_llamaindex_memory.py
"""
Demo: pass RAW user text to LlamaIndexMemory.add_turn(...).
LlamaIndexMemory uses LlamaIndex + Chroma for storage/retrieval and your CustomProxyEmbedder for embeddings.

Reads config via src.config.load_config():
  - OPENAI_BASE_URL
  - OPENAI_API_KEY
  - MB_CHAT_MODEL
  - MB_EMBED_MODEL
  - (optional) MB_COLLECTION, LOG_LEVEL, etc.

Usage:
  python demo_llamaindex_memory.py
"""

from __future__ import annotations

import os
import re
from typing import Dict, Any, List

from memory.third_party.llamaindex_adapter import LlamaIndexMemory
from utils.config import load_config

# --- Simple dataset mirroring your earlier demos ---
DATA: List[Dict[str, Any]] = [{
    "conversation_id": "SIMPLE_EASY_42_000",
    "scenario": "simple",
    "difficulty": "easy",
    "noise_tags": [],
    "turns": [
        {"role": "user", "content": "Hi, I'm really into music. My favorite genre is indie folk."},
        {"role": "assistant", "content": "That's interesting! Indie folk often has a very unique sound. Do you have any favorite artists?"},
        {"role": "user", "content": "Yes, I really like Fleet Foxes and Bon Iver. Their harmonies are incredible."},
        {"role": "assistant", "content": "Both excellent choices. Their music definitely evokes a certain atmosphere."},
        {"role": "user", "content": "Agreed. Speaking of music, what's a good way to discover new artists in that genre?"},
        {"role": "assistant", "content": "Try curated playlists, music blogs, and local live shows."},
        {"role": "user", "content": "That sounds good. By the way, what did I say my favorite music genre was?"}
    ],
    "eval": [
        {
            "question": "what did i say my favorite music genre was?",
            "expected_answer": "indie folk",
            "relevant_texts": ["my favorite genre is indie folk."],
            "subject_hint": "user's favorite music genre"
        }
    ]
}]


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _extract_answer_from_text(text: str) -> str:
    """
    LlamaIndexMemory returns free-text memories. Pull the answer with a tiny heuristic.
    """
    m = re.search(r"favorite (?:music )?genre is ([\w\s\-]+)[\.\! ]?", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()


def _pick_answer(res: Dict[str, Any]) -> str:
    if not res or not res.get("results"):
        return ""
    best = res["results"][0]
    # best is shaped like: {"type":"memory","text": "...", ...}
    return _extract_answer_from_text(best.get("text", ""))


def main():
    cfg = load_config()  # also initializes logging level and seed

    mem = LlamaIndexMemory(
        name="demo_llamaindex",
        collection=os.getenv("MB_COLLECTION", cfg.get("MB_COLLECTION", "mem0_benchmark_dev")),
        chat_model=os.getenv("MB_CHAT_MODEL", cfg.get("MB_CHAT_MODEL", "gemini-flash")),
        embed_model=os.getenv("MB_EMBED_MODEL", cfg.get("MB_EMBED_MODEL", "gemini-embedding")),
        base_url=os.getenv("OPENAI_BASE_URL", cfg.get("OPENAI_BASE_URL", "")),
        api_key=os.getenv("OPENAI_API_KEY", cfg.get("OPENAI_API_KEY", "")),
        restrict_to_conv=True,
    )

    for convo in DATA:
        conv_id = convo["conversation_id"]
        print(f"\n=== Conversation {conv_id} ===")
        for i, turn in enumerate(convo["turns"], start=1):
            mem.add_turn(conv_id, i, turn["role"], turn["content"])

        # Evaluate
        for q in convo.get("eval", []):
            query = q["question"]
            want = _normalize(q["expected_answer"])
            res = mem.retrieve(conv_id, query, top_k=3)
            got = _pick_answer(res)
            got_norm = _normalize(got)
            status = "PASS" if got_norm == want else "FAIL"
            print(f"Q: {query}\n   expected={q['expected_answer']} | got={got}  [{status}]")


if __name__ == "__main__":
    main()
