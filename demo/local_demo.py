# demo/local_multihop_demo.py
"""
Demo: Complex multi-hop conversation with canonicalized updates, aliasing, and retrieval options.

Conversation (9 turns):
1. My name is Alice. I work at OpenAI and live in San Francisco. I have a friend named Robert.
2. Robert moved to New York and works at Microsoft now.
3. I relocated to Los Angeles, but I'm still working at OpenAI.
4. I'm not friends with Robert anymore.
5. I changed my name to Kate.
6. I have a friend Sara.
7. I have another friend Mary.
8. She works as a nurse.
9. To be clear, I meant Mary works as a nurse.

Then we run queries under several retrieval options (all exclude 'absent' facts by default).
"""

from __future__ import annotations
import time
from pathlib import Path
from dotenv import load_dotenv

from memory.local import LocalMemory

load_dotenv()


def ensure_dir(path: str) -> str:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def banner(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_facts(changes):
    """
    Show per-turn fact changes; we still display [deleted] events to illustrate
    single-valued slot updates, but retrieval below never includes 'absent' facts.
    """
    if not changes:
        print("   (no facts extracted)")
        return
    for f in changes:
        subj, pred, obj = f["subject"], f["predicate"], f["object"]
        status, assertion = f["status"], f["assertion"]
        conf = f.get("confidence", 0.0)
        ev = f.get("evidence")
        marker = "−" if (status == "deleted" or assertion == "absent") else "•"
        human = "deleted" if (status == "deleted" or assertion == "absent") else "added"
        tail = f" (conf={conf:.2f}" + (f", evidence='{ev}')" if ev else ")")
        print(f"   {marker} {subj}'s {pred} → {obj}  [{human}]{tail}")


def print_hits(hits):
    if not hits:
        print("   (no results)")
        return
    for i, h in enumerate(hits, 1):
        subj, pred, obj = h["subject"], h["predicate"], h["object"]
        score, cos, conf = h["score"], h["cosine"], h["confidence"]
        print(f" {i:>2}. {subj}'s {pred} → {obj}  | score={score:.3f}, cos={cos:.3f}, conf={conf:.2f} ({h['assertion']})")


def run():
    db_path = ensure_dir(".memdb/multihop_demo.sqlite")
    mem = LocalMemory()
    mem.connect(db_path=db_path, echo=False)

    # --- Conversation turns (complex + multi-hop) ---
    turns = [
        # 1
        "My name is Alice. I work at OpenAI and live in San Francisco. I have a friend named Robert.",
        # 2
        "Robert moved to New York and works at Microsoft now.",
        # 3
        "I relocated to Los Angeles, but I'm still working at OpenAI.",
        # 4
        "I'm not friends with Robert anymore.",
        # 5
        "I changed my name to Kate.",
        # 6
        "I have a friend Sara.",
        # 7
        "I have another friend Mary.",
        # 8
        "She works as a nurse.",
        # 9
        "To be clear, I meant Mary works as a nurse.",
    ]

    banner("Ingesting multi-hop conversation into LocalMemory …")
    t0 = time.time()
    for i, text in enumerate(turns, start=1):
        print(f" Turn {i}: {text}")
        res = mem.add_turn(text=text, conv_id="conv_demo", turn_id=f"turn_{i}", user_id="demo_user")
        print(f"   → inserted {res['inserted']} facts:")
        print_facts(res["facts"])
    print(f"\nTotal time: {1000*(time.time()-t0):.1f} ms")

    # --- Queries to test ---
    queries = [
        "What is my current name?",
        "Where do I live now?",
        "Where did I live before?",
        "Where does Robert live now?",
        "Where does Robert work now?",
        "Who are my friends?",
        "What is Mary's occupation?",
        "Which company do I work at?",
    ]

    # --- Retrieval configurations to compare (all exclude 'absent') ---
    configs = [
        {"label": "Default (cosine)", "kwargs": {}},
        {"label": "Refine query", "kwargs": {"refine_query": True}},
        {"label": "Hybrid metric", "kwargs": {"similarity_metric": "hybrid"}},
        {"label": "Hybrid + refine", "kwargs": {"similarity_metric": "hybrid", "refine_query": True}},
        {"label": "Cosine + subject-aware", "kwargs": {"subject_aware": True}},
        {"label": "Hybrid + subject-aware", "kwargs": {"similarity_metric": "hybrid", "subject_aware": True}},
        # No include_absent variants in this demo run
    ]

    banner("Queries (CURRENT VIEW)")
    for q in queries:
        print(f"\nQ: {q}")
        for cfg in configs:
            print(f"--- {cfg['label']} ---")
            hits = mem.retrieve(query=q, user_id="demo_user", k=5, **cfg["kwargs"])
            print_hits(hits)

    banner("Done.")


if __name__ == "__main__":
    run()
