# repo/demo/demo_keyword_baseline.py
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, Any, List

from memory.keyword_baseline import KeywordBaseline
from memory.base import DefaultEvaluationHooks, JSONLinesFileSink


def load_dataset(path: str) -> List[Dict[str, Any]]:
    """
    Reads a conversations.jsonl file:
    Each line: {
      "conversation_id": "...",
      "turns": [{"role":"user|assistant", "content":"..."} ...],
      "eval": [{"question":"...", "expected_answer":"...", "relevant_texts":["..."]} ...]
    }
    """
    data = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def toy_dataset() -> List[Dict[str, Any]]:
    """A minimal in-memory dataset for quick demo."""
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
                    "relevant_texts": ["i work at anthropic now", "alice work at anthropic now"]
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
                    "relevant_texts": ["i live in san francisco"]
                }
            ]
        }
    ]


def exact_match(ans: str, gold: str) -> int:
    return int((ans or "").strip().lower() == (gold or "").strip().lower())


def make_answer_from_items(query: str, items: List[Dict[str, Any]]) -> str:
    """
    Ultra-naive answerer to keep the demo self-contained:
    - If any retrieved chunk contains "work at X" -> return X.
    - If query mentions 'city' and a chunk has 'live in Y' -> return Y.
    Fallback 'unknown'.
    """
    ql = query.lower()
    for it in items:
        txt = (it.get("text") or "").lower()
        if "work at" in txt:
            try:
                return txt.split("work at", 1)[1].strip().strip(".")
            except Exception:
                pass
        if "city" in ql and "live in" in txt:
            try:
                return txt.split("live in", 1)[1].strip().strip(".")
            except Exception:
                pass
    return "unknown"


def run_demo(data: List[Dict[str, Any]], top_k: int, log_path: str | None):
    # Optional JSONL logging for evaluation events
    hooks = DefaultEvaluationHooks(
        system_name="keyword_baseline",
        sink=JSONLinesFileSink(log_path) if log_path else None,
        base_meta={"demo": True}
    )
    mem = KeywordBaseline(hooks=hooks)

    # Ingest & evaluate
    total_em = 0
    q_count = 0
    for conv in data:
        cid = conv["conversation_id"]
        for i, turn in enumerate(conv["turns"], start=1):
            mem.add_turn(cid, i, turn["role"], turn["content"])

        for q in conv.get("eval", []):
            t0 = time.perf_counter()
            res = mem.retrieve(cid, q["question"], top_k=top_k)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            items = res["results"]
            ans = make_answer_from_items(q["question"], items)
            em = exact_match(ans, q.get("expected_answer", ""))
            total_em += em; q_count += 1

            print("\n=== Query ===")
            print(q["question"])
            print("--- Retrieved (top_k=%d, %.1f ms) ---" % (top_k, latency_ms))
            for r in items:
                print(f"[{r.get('score', 0):.3f}] {r.get('source','')} :: {r.get('text','')[:120]}")
            print("--- Answer ---")
            print(f"pred: {ans} | gold: {q.get('expected_answer','')!r} | EM={em}")

    if q_count:
        print("\nOverall Exact Match: %.1f%% (%d/%d)" % (100.0 * total_em / q_count, total_em, q_count))


def main():
    ap = argparse.ArgumentParser(description="Demo: Keyword BM25 Baseline")
    ap.add_argument("--data", type=str, default="", help="Path to conversations.jsonl (optional).")
    ap.add_argument("--top_k", type=int, default=5, help="Top-k to retrieve.")
    ap.add_argument("--log", type=str, default="", help="Optional JSONL log path for eval hooks.")
    args = ap.parse_args()

    if args.data and os.path.exists(args.data):
        dataset = load_dataset(args.data)
    else:
        dataset = toy_dataset()

    run_demo(dataset, top_k=args.top_k, log_path=args.log or None)


if __name__ == "__main__":
    main()
