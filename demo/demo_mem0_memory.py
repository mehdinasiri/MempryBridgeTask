# repo/demo/demo_mem0_memory.py
from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from utils.factory import make_mem0_memory, make_hooks


# -----------------------------
# tiny loader (JSONL of dicts)
# -----------------------------
def load_dataset(path: str) -> List[Dict[str, Any]]:
    if path and os.path.exists(path):
        data: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except Exception as e:
                    print(f"[demo] bad JSON on line {i}: {e}", flush=True)
        return data
    # minimal fallback so demo always prints something
    return [
        {
            "conversation_id": "M0_001",
            "turns": [
                {"role": "user", "content": "My name is Alice and I work at OpenAI."},
                # {"role": "assistant", "content": "Nice to meet you, Alice."},
                # {"role": "user", "content": "Actually, I work at Anthropic now."},
                # {"role": "assistant", "content": "Got it."},
            ],
            "eval": [
                {
                    "question": "Where do I work now?",
                    "expected_answer": "anthropic",
                    "relevant_texts": ["actually, i work at anthropic now"],
                    "subject_hint": "user",
                }
            ],
        }
    ]


# -----------------------------
# naive answer extraction
# -----------------------------
WORK_AT_RE = re.compile(r"\bwork(?:s)?\s+at\s+([A-Za-z][A-Za-z0-9 .&-]+)", re.I)
LIVE_IN_RE = re.compile(r"\blive(?:s)?\s+in\s+([A-Za-z][A-Za-z0-9 .&-]+)", re.I)
NAME_IS_RE = re.compile(r"\b(?:my\s+name\s+is|people\s+call\s+me)\s+([A-Za-z][A-Za-z0-9 .&-]+)", re.I)


def extract_answer(query: str, items: List[Dict[str, Any]]) -> str:
    ql = (query or "").lower()
    for it in items:
        txt = (it.get("text") or "")
        if "work" in ql:
            m = WORK_AT_RE.search(txt)
            if m:
                return m.group(1).strip().rstrip(".").lower()
        if ("city" in ql or "live" in ql):
            m = LIVE_IN_RE.search(txt)
            if m:
                return m.group(1).strip().rstrip(".").lower()
        if "name" in ql:
            m = NAME_IS_RE.search(txt)
            if m:
                return m.group(1).strip().rstrip(".").lower()
    return "unknown"


def exact_match(pred: str, gold: str) -> int:
    return int((pred or "").strip().lower() == (gold or "").strip().lower())


# -----------------------------
# runner
# -----------------------------
def run_demo(
    dataset: List[Dict[str, Any]],
    top_k: int,
    log_path: Optional[str],
    subject_hint_arg: Optional[str],
) -> None:
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        print(f"[demo] hooks logging to: {log_path}", flush=True)

    hooks = make_hooks("mem0_memory", log_path=log_path, base_meta={"demo": True})

    try:
        mem = make_mem0_memory(hooks=hooks)
    except ImportError as e:
        print("ERROR: Mem0 not installed. Try: pip install mem0ai\n", e, flush=True)
        return

    n_convs = len(dataset)
    n_evals = sum(len(c.get("eval", [])) for c in dataset)
    print(f"[demo] loaded {n_convs} conversations with {n_evals} eval items.", flush=True)

    if n_convs == 0 or n_evals == 0:
        print("[demo] dataset had no evals → using built-in fallback.", flush=True)
        dataset = load_dataset("")  # fallback

    total_em, q_count = 0, 0

    for conv in dataset:
        cid = conv.get("conversation_id") or "conv"
        turns = conv.get("turns", [])
        evals = conv.get("eval", [])

        # Ingest conversation turns
        for i, turn in enumerate(turns, start=1):
            role = turn.get("role", "")
            content = turn.get("content", "")
            mem.add_turn(cid, i, role, content)

        # Evaluate
        for q in evals:
            question = q.get("question", "")
            expected = q.get("expected_answer", "")
            subject_hint = subject_hint_arg or q.get("subject_hint")

            t0 = time.perf_counter()
            res = mem.retrieve(cid, question, top_k=top_k, subject_hint=subject_hint)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            items = res.get("results", [])
            ans = extract_answer(question, items)
            em = exact_match(ans, expected)

            q_count += 1
            total_em += em

            print("\n=== Query ===", flush=True)
            print(question, flush=True)
            if subject_hint:
                print(f"(subject_hint: {subject_hint})", flush=True)

            print(f"--- Retrieved (top_k={top_k}, {latency_ms:.1f} ms) ---", flush=True)
            if not items:
                print("(no results)", flush=True)
            else:
                for r in items:
                    print(f"[{r.get('score',0):.3f}] {r.get('source','')} :: {r.get('text','')[:160]}", flush=True)

            print("--- Answer ---", flush=True)
            print(f"pred: {ans} | gold: {expected!r} | EM={em}", flush=True)

    print("\n[demo] done.", flush=True)
    if q_count:
        print("Overall Exact Match: %.1f%% (%d/%d)" % (100.0 * total_em / q_count, total_em, q_count), flush=True)
    else:
        print("No eval items were executed (nothing to print).", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Demo: Mem0 Adapter")
    ap.add_argument("--data", type=str, default="", help="Path to conversations.jsonl (optional).")
    ap.add_argument("--top_k", type=int, default=5, help="Top-k to retrieve.")
    ap.add_argument("--log", type=str, default="", help="Optional JSONL log path for eval hooks.")
    ap.add_argument("--subject_hint", type=str, default="", help="Optional subject hint for retrieval.")
    args = ap.parse_args()

    # Environment summary (helps when debugging 401s/model routing)
    print(f"[demo] data source: {args.data or '(fallback)'}", flush=True)
    print(
        f"[demo] OPENAI_BASE_URL={os.getenv('OPENAI_BASE_URL','') or '(unset)'} | "
        f"MB_CHAT_MODEL={os.getenv('MB_CHAT_MODEL','') or '(unset)'} | "
        f"MB_EMBED_MODEL={os.getenv('MB_EMBED_MODEL','') or '(unset)'}",
        flush=True,
    )

    dataset = load_dataset(args.data)
    run_demo(
        dataset,
        top_k=args.top_k,
        log_path=(args.log or None),
        subject_hint_arg=(args.subject_hint or None),
    )


if __name__ == "__main__":
    main()
