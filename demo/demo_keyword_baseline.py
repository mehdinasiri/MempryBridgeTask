# repo/demo/demo_keyword_baseline.py
from __future__ import annotations

from typing import List, Dict, Any
from loguru import logger

from utils.config import load_config  # initializes loguru in your setup
from memory.keyword_baseline import KeywordBaseline


CONVERSATIONS: List[Dict[str, Any]] = [
    {
        "conversation_id": "001",
        "turns": [
            {"role": "user", "content": "My name is Alice and I work at OpenAI."},
            {"role": "assistant", "content": "Nice to meet you, Alice."},
            {"role": "user", "content": "Actually, I work at Anthropic now."},
            {"role": "assistant", "content": "Got it."}
        ],
        "eval": [
            {"question": "Where does Alice work now?", "expected_answer": "anthropic"},
        ],
    },
    {
        "conversation_id": "002",
        "turns": [
            {"role": "user", "content": "I live in San Francisco and love espresso."},
            {"role": "assistant", "content": "Espresso is great!"},
            {"role": "user", "content": "By the way I switched to oat milk."}
        ],
        "eval": [
            {"question": "What city do I live in?", "expected_answer": "san francisco"},
        ],
    },
]


def run_eval(mem: KeywordBaseline, conv: Dict[str, Any]) -> None:
    conv_id = conv["conversation_id"]
    print(f"\n=== Keyword Baseline — Conversation {conv_id} ===")

    # Ingest turns (only user turns are indexed inside KeywordBaseline)
    for i, turn in enumerate(conv["turns"], start=1):
        mem.add_turn(conv_id, i, turn["role"], turn["content"])

    # Evaluate simple questions
    for probe in conv.get("eval", []):
        q = probe["question"]
        expected = (probe.get("expected_answer") or "").strip().lower()

        res = mem.retrieve(conv_id, q, top_k=1)
        got = ""
        if res["results"]:
            top = res["results"][0]
            # Keyword baseline returns chunks (raw user texts); use them directly
            got = (top.get("text") or "").strip().lower()

        ok = "OK" if expected and expected in got else "FAIL"
        print(f"Q: {q}\n   expected={expected} got={(got[:80] + '...') if len(got) > 80 else got} [{ok}]")


def main():
    load_config()  # sets up loguru, seeds, etc.
    logger.info("Running KeywordBaseline demo")

    mem = KeywordBaseline(name="keyword_baseline_demo")

    for conv in CONVERSATIONS:
        run_eval(mem, conv)


if __name__ == "__main__":
    main()
