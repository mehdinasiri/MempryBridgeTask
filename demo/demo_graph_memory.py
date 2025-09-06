from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Any, Dict, List

from utils.factory import make_graph_memory, make_hooks


def load_dataset(path: str) -> List[Dict[str, Any]]:
    if path and os.path.exists(path):
        data = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    # Default toy set
    return [
        {
            "conversation_id": "001",
            "turns": [
                {"role": "user", "content": "My name is Alice and I work at OpenAI."},
                {"role": "assistant", "content": "Nice to meet you, Alice."},
                {"role": "user", "content": "Actually, I work at Anthropic now."},
                {"role": "assistant", "content": "Got it."}
            ],
            "eval": [
                {
                    "question": "Where does Alice work now?",
                    "expected_answer": "anthropic",
                    "subject_hint": "Alice"
                }
            ]
        },
        {
            "conversation_id": "002",
            "turns": [
                {"role": "user", "content": "I live in San Francisco and love espresso."},
                {"role": "assistant", "content": "Espresso is great!"},
                {"role": "user", "content": "By the way I switched to oat milk."}
            ],
            "eval": [
                {
                    "question": "What city do I live in?",
                    "expected_answer": "san francisco",
                    "subject_hint": "user"
                }
            ]
        }
    ]


WORK_AT_RE = re.compile(r"\bwork(?:s)?\s+at\s+([A-Za-z][A-Za-z0-9 .&-]+)", re.I)
LIVE_IN_RE = re.compile(r"\blive(?:s)?\s+in\s+([A-Za-z][A-Za-z0-9 .&-]+)", re.I)


def exact_match(pred: str, gold: str) -> int:
    return int((pred or "").strip().lower() == (gold or "").strip().lower())


def extract_answer(query: str, items: List[Dict[str, Any]]) -> str:
    ql = (query or "").lower()
    # Prefer facts → edges → chunks
    for it in items:
        if it.get("type") == "fact":
            pred = (it.get("predicate") or "").lower()
            obj = (it.get("object") or "").lower()
            if "work" in ql and pred in ("works_at", "work_at", "employer"):
                return obj
            if "city" in ql and pred in ("lives_in", "live_in", "location"):
                return obj
    for it in items:
        if it.get("type") == "graph_edge":
            p = (it.get("predicate") or "").lower()
            v = (it.get("v") or "").lower()
            if "work" in ql and p in ("works_at", "work_at", "employer"):
                return v
            if "city" in ql and p in ("lives_in", "live_in", "location"):
                return v
    for it in items:
        if it.get("type") == "chunk":
            txt = (it.get("text") or "")
            m = WORK_AT_RE.search(txt)
            if m and "work" in ql:
                return m.group(1).strip().rstrip(".").lower()
            m = LIVE_IN_RE.search(txt)
            if m and "city" in ql:
                return m.group(1).strip().rstrip(".").lower()
    return "unknown"


def run_demo(dataset: List[Dict[str, Any]], top_k: int, log_path: str | None, subject_hint_arg: str | None):
    hooks = make_hooks("graph_memory", log_path=log_path, base_meta={"demo": True})
    mem = make_graph_memory(hooks=hooks)  # <- uses MB_VECTOR_STORE + MB_BACKEND (+ SQLite facts)

    total_em = 0
    q_count = 0

    for conv in dataset:
        cid = conv["conversation_id"]
        for i, turn in enumerate(conv["turns"], start=1):
            mem.add_turn(cid, i, turn["role"], turn["content"])

        for q in conv.get("eval", []):
            subject_hint = subject_hint_arg or q.get("subject_hint")
            t0 = time.perf_counter()
            res = mem.retrieve(cid, q["question"], top_k=top_k, subject_hint=subject_hint)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            items = res["results"]

            ans = extract_answer(q["question"], items)
            em = exact_match(ans, q.get("expected_answer", ""))

            total_em += em
            q_count += 1

            print("\n=== Query ===")
            print(q["question"])
            if subject_hint:
                print(f"(subject_hint: {subject_hint})")
            print("--- Retrieved (top_k=%d, %.1f ms) ---" % (top_k, latency_ms))
            for r in items:
                t = r.get("type")
                if t == "fact":
                    print(f"[{r.get('score',0):.3f}] FACT  :: {r.get('subject','')} -[{r.get('predicate','')}]-> {r.get('object','')}  ({r.get('source','')})")
                elif t == "graph_edge":
                    print(f"[{r.get('score',0):.3f}] EDGE  :: {r.get('u','')} -[{r.get('predicate','')}]-> {r.get('v','')}  facts={r.get('fact_ids',[])}")
                else:
                    print(f"[{r.get('score',0):.3f}] CHUNK :: {r.get('source','')} :: {r.get('text','')[:120]}")
            print("--- Answer ---")
            print(f"pred: {ans} | gold: {q.get('expected_answer','')!r} | EM={em}")

    if q_count:
        print("\nOverall Exact Match: %.1f%% (%d/%d)" % (100.0 * total_em / q_count, total_em, q_count))


def main():
    ap = argparse.ArgumentParser(description="Demo: Graph (Mixed) Memory (factory-powered)")
    ap.add_argument("--data", type=str, default="", help="Path to conversations.jsonl (optional).")
    ap.add_argument("--top_k", type=int, default=5, help="Top-k to retrieve.")
    ap.add_argument("--log", type=str, default="", help="Optional JSONL log path for eval hooks.")
    ap.add_argument("--subject_hint", type=str, default="", help="Optional subject hint (overrides dataset).")
    args = ap.parse_args()

    dataset = load_dataset(args.data)
    run_demo(dataset, top_k=args.top_k, log_path=(args.log or None), subject_hint_arg=(args.subject_hint or None))


if __name__ == "__main__":
    main()
