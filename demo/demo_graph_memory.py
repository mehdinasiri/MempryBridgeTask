# repo/demo/demo_graph_memory.py
from __future__ import annotations

import os
from typing import List, Dict, Any

from loguru import logger

from utils.config import load_config  # ensures logger is initialized in your setup
from memory.graph_memory import GraphMemory

CONVERSATIONS: List[Dict[str, Any]] = [{
    "conversation_id": "SIMPLE_EASY_42_000",
    "scenario": "simple",
    "difficulty": "easy",
    "noise_tags": [],
    "turns": [
        {
            "role": "user",
            "content": "Hi, I'm really into music. My favorite genre is indie folk."
        },
        {
            "role": "assistant",
            "content": "That's interesting! Indie folk often has a very unique sound. Do you have any favorite artists?"
        },
        {
            "role": "user",
            "content": "Yes, I really like Fleet Foxes and Bon Iver. Their harmonies are incredible."
        },
        {
            "role": "assistant",
            "content": "Both excellent choices. Their music definitely evokes a certain atmosphere."
        },
        {
            "role": "user",
            "content": "Agreed. Speaking of music, what's a good way to discover new artists in that genre?"
        },
        {
            "role": "assistant",
            "content": "There are several ways! You could try curated playlists on streaming services, explore music blogs, or even check out live local performances."
        },
        {
            "role": "user",
            "content": "That sounds like a good plan. By the way, what did I say my favorite music genre was?"
        }
    ],
    "eval": [
        {
            "question": "what did i say my favorite music genre was?",
            "expected_answer": "indie folk",
            "relevant_texts": [
                "my favorite genre is indie folk."
            ],
            "subject_hint": "user's favorite music genre"
        }
    ]
}
]


# [
#     {
#         "conversation_id": "001",
#         "turns": [
#                 {"role": "user", "content": "My name is Alice and I work at OpenAI."},
#                 {"role": "assistant", "content": "Nice to meet you, Alice."},
#                 {"role": "user", "content": "Actually, I work at Anthropic now."},
#                 {"role": "assistant", "content": "Got it."}
#         ],
#         "eval": [
#             {"question": "Where does Alice work now?", "expected_answer": "anthropic"},
#         ],
#     },
#     {
#         "conversation_id": "002",
#         "turns": [
#                 {"role": "user", "content": "I live in San Francisco and love espresso."},
#                 {"role": "assistant", "content": "Espresso is great!"},
#                 {"role": "user", "content": "By the way I switched to oat milk."}
#         ],
#         "eval": [
#             {"question": "What city do I live in?", "expected_answer": "san francisco"},
#         ],
#     },
# ]


def pretty(item: Dict[str, Any]) -> str:
    """Tiny helper to stringify a retrieval item."""
    if item.get("type") == "fact":
        subj = item.get("subject")
        pred = item.get("predicate")
        obj = item.get("object")
        ev = item.get("text")
        return f"FACT: ({subj} | {pred} | {obj})  evidence={ev!r}  score={item.get('score')}"
    return f"{item.get('type')}: {item}"


def run_eval(mem: GraphMemory, conv: Dict[str, Any]) -> None:
    conv_id = conv["conversation_id"]

    print(f"\n=== Conversation {conv_id} ===")
    # Ingest each turn (only user turns are indexed for facts in Vector/GraphMemory)
    for i, turn in enumerate(conv["turns"], start=1):
        mem.add_turn(conv_id, i, turn["role"], turn["content"])

    # Ask evaluation questions
    for q in conv.get("eval", []):
        question = q["question"]
        expected = (q.get("expected_answer") or "").strip().lower()
        res = mem.retrieve(conv_id, question, top_k=1)
        got = ""
        evidence = ""
        if res["results"]:
            top = res["results"][0]
            # Heuristic: if it’s a fact, take the object as the answer
            if top.get("type") == "fact":
                got = str(top.get("object") or "").strip().lower()
                evidence = str(top.get("text") or "")
            else:
                # Fallback: try to read something meaningful
                got = str(top.get("text") or top).strip().lower()

        ok = "OK" if expected and expected in got else "FAIL"
        print(f"Q: {question}\n   expected={expected} got={got} [{ok}]")
        if evidence:
            print(f"   evidence: {evidence}")


def main():
    # 1) Load config (.env) and initialize logging
    cfg = load_config()
    logger.info("Config loaded for demo (GraphMemory)")

    # Quick check: make sure LLM credentials exist (demo will still run, but extraction will fail otherwise)
    base_url = cfg.get("OPENAI_BASE_URL", "")
    api_key = cfg.get("OPENAI_API_KEY", "")
    if not base_url or not api_key:
        logger.warning(
            "OPENAI_BASE_URL / OPENAI_API_KEY not set. "
            "The demo will try to run but fact extraction will likely fail."
        )

    # 2) Create GraphMemory (same args as VectorMemory; graph persists to SQLite)
    mem = GraphMemory(
        name="graph_memory_demo",
        index_backend="chroma",  # or "lancedb"
        collection_or_table="demo_facts_graph",
        persist_path=".memdb/chroma_graph_demo",  # for lancedb: ".memdb/lancedb"
        restrict_to_conv=True,
        graph_db_path=".memdb/graph_demo.sqlite",  # graph persistence
    )

    # 3) Run demo conversations
    for conv in CONVERSATIONS:
        run_eval(mem, conv)


if __name__ == "__main__":
    main()
