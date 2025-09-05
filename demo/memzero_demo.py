# demo/memzero_demo.py
"""
Minimal demo for the MemZeroMemory backend (mem0 + Chroma + custom embedder).

Make sure your .env is set with:
  OPENAI_BASE_URL, OPENAI_API_KEY, MB_CHAT_MODEL, MB_EMBED_MODEL, MB_EMBED_DIMS
and (optionally) MB_COLLECTION for mem0.

Usage:
  python -m demo.memzero_demo
  # or:
  python demo/memzero_demo.py --k 5
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

# Ensure env is loaded before importing the backend
load_dotenv()

from memory.mem_zero import MemZeroMemory  # noqa: E402


def banner(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def ensure_dir(path: str) -> str:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def print_hits(hits: List[Dict[str, Any]]) -> None:
    if not hits:
        print("  (no results)")
        return
    # Try to normalize mem0 hit shape into a readable triple-ish line
    for i, h in enumerate(hits, 1):
        # Common fields we may find:
        subj = h.get("subject") or h.get("entity") or h.get("who") or h.get("entity_id") or "subject"
        pred = h.get("predicate") or h.get("attribute") or h.get("relation") or "predicate"
        obj = (
            h.get("object")
            or h.get("value")
            or h.get("what")
            or h.get("content")
            or h.get("text")
            or "object"
        )
        sc = h.get("score") or h.get("similarity") or h.get("relevance") or 0.0
        conf = h.get("confidence", 1.0)
        asrt = h.get("assertion") or h.get("status") or "present"
        print(f" {i:>2}. {subj}'s {pred} → {obj}  | score={float(sc):.3f}, conf={float(conf):.2f} ({asrt})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5, help="top-k to retrieve")
    parser.add_argument("--collection", type=str, default=os.getenv("MB_COLLECTION", "mem0_demo"))
    args = parser.parse_args()

    mem = MemZeroMemory()
    # Pass through a collection name if your MemZeroMemory supports it
    mem.connect(collection=args.collection)

    user = "demo_user_mem0"
    conv = "conv_mem0_demo"

    conversation = [
        {"role": "user", "content": "I'm Bob. I'm based in Austin and I study at UT."},
        {"role": "assistant", "content": "Hook 'em!"},
        {"role": "user", "content": "I moved to Seattle and now work at Microsoft."},
        {"role": "assistant", "content": "Nice career step!"},
    ]

    banner("Ingesting conversation into MemZeroMemory …")
    t0 = time.time()
    res = mem.add_conversation(messages=conversation, conv_id=conv, user_id=user)
    elapsed = 1000 * (time.time() - t0)
    print(f"Ingest complete in {elapsed:.1f} ms")
    # mem0 may return various shapes; show something informative without crashing
    try:
        print(json.dumps(res, indent=2, default=str))
    except Exception:
        print(res)

    queries = [
        "What's his name?",
        "Where does he work now?",
        "Which city does he live in?",
    ]

    banner("Queries")
    for q in queries:
        print(f"\nQ: {q}")

        t1 = time.time()
        hits = mem.retrieve(query=q, user_id=user, k=args.k)
        print(f"Retrieved {len(hits)} in {1000*(time.time()-t1):.1f} ms")
        print_hits(hits[:args.k])

        t2 = time.time()
        reranked = mem.retrieve_reranked(query=q, user_id=user, k=args.k, top_n=min(3, args.k))
        print(f"Reranked top-{min(3, args.k)} in {1000*(time.time()-t2):.1f} ms")
        print(json.dumps(reranked, indent=2, default=str))


if __name__ == "__main__":
    main()
