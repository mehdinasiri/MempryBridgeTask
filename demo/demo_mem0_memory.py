# demo_mem0_memory.py
"""
Demo: pass RAW user text to Mem0Memory.add_turn(...).
Mem0Memory delegates extraction + storage + retrieval to the mem0 library.

Requires your .env (or env vars) to define at least:
  - OPENAI_BASE_URL
  - OPENAI_API_KEY
  - MB_CHAT_MODEL
  - MB_EMBED_MODEL
Optional:
  - MB_COLLECTION (defaults to "mem0_default")
  - LOG_LEVEL=DEBUG (to see detailed logs)

Usage:
  python demo_mem0_memory.py
"""

from __future__ import annotations

import os
import re
from typing import Dict, Any, List

from memory.third_party.mem0_adapter import Mem0Memory

DATA: List[Dict[str, Any]] = [
{
  "conversation_id": "MULTIHOP_COMPLEX_42_000",
  "scenario": "multihop",
  "difficulty": "complex",
  "noise_tags": [
    "emoji",
    "multilingual",
    "coreference",
    "cross_session",
    "chatter"
  ],
  "turns": [
    {
      "role": "user",
      "content": "Hey there, I was just thinking about my old colleague, Liam Chen. He's quite a brilliant data scientist, you know? 🤓"
    },
    {
      "role": "assistant",
      "content": "Oh, that's interesting! Data science is a fascinating field. Is he still with the same company?"
    },
    {
      "role": "user",
      "content": "Nah, he moved. He's at Quantum Innovations now. They're doing some really cutting-edge stuff with AI and machine learning. Pretty cool! 👍"
    },
    {
      "role": "assistant",
      "content": "Quantum Innovations, got it. Sounds like an exciting place to work."
    },
    {
      "role": "user",
      "content": "Yeah, I guess so. He's always super busy. Do you know if they have a big presence in the US? I forgot."
    },
    {
      "role": "assistant",
      "content": "I don't have that detail on hand, but I can tell you more about Quantum Innovations if you'd like."
    },
    {
      "role": "user",
      "content": "Not now, maybe later. I was trying to remember where their main office is. I know it's in Europe somewhere, but the city escapes me. 😅"
    },
    {
      "role": "assistant",
      "content": "No worries, take your time. Europe is a big place!"
    },
    {
      "role": "user",
      "content": "Ah! It's Stockholm. Definitely Stockholm. Jag älskar Sverige! 🇸🇪 Such a beautiful place."
    },
    {
      "role": "assistant",
      "content": "Stockholm, excellent! I've heard wonderful things about it."
    },
    {
      "role": "user",
      "content": "It truly is! Anyway, completely unrelated, did you catch that new space documentary on Netflix? Blew my mind! 🚀"
    },
    {
      "role": "assistant",
      "content": "Not yet, but I'll add it to my watch list! Sounds intriguing."
    },
    {
      "role": "user",
      "content": "You totally should. Ok, I'm heading out for a quick coffee break. Back in a bit! 👋"
    },
    {
      "role": "assistant",
      "content": "Enjoy your coffee! See you soon."
    },
    {
      "role": "user",
      "content": "(end of session)\n(new session)\nHi again! 👋 So glad to be back. The coffee was excellent."
    },
    {
      "role": "assistant",
      "content": "Welcome back! Glad to hear you enjoyed it."
    },
    {
      "role": "user",
      "content": "Just circling back to our earlier chat. About Liam Chen, the data scientist."
    },
    {
      "role": "assistant",
      "content": "Yes, what about him?"
    },
    {
      "role": "user",
      "content": "Given what we discussed, what city is Liam Chen's employer based in? 🤔"
    },
    {
      "role": "assistant",
      "content": "Liam Chen's employer, Quantum Innovations, is based in Stockholm."
    }
  ],
  "eval": [
    {
      "question": "Given what we discussed, what city is Liam Chen's employer based in? 🤔",
      "expected_answer": "stockholm",
      "relevant_texts": [
        "he's at quantum innovations now.",
        "it's stockholm. definitely stockholm."
      ],
      "subject_hint": "liam chen"
    }
  ]
}
]


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _extract_answer_from_hit(hit: Dict[str, Any]) -> str:
    """
    Mem0 results may be structured triples or just text memories depending on extractor.
    Prefer structured ('object'), else fall back to raw 'text' and heuristic pick.
    """
    # Try structured fact
    obj = (hit.get("object") or "").strip()
    if obj:
        return obj

    # Otherwise, mine from text
    txt = (hit.get("text") or "").strip()
    return txt


def run_demo() -> None:
    # Build a Mem0Memory. Most config will be taken from env (see docstring).
    mem = Mem0Memory(
        name="demo_mem0",
        collection=os.getenv("MB_COLLECTION", "mem0_default"),
        chat_model=os.getenv("MB_CHAT_MODEL", "gemini-flash"),
        embed_model=os.getenv("MB_EMBED_MODEL", "gemini-embedding"),
        base_url=os.getenv("OPENAI_BASE_URL", ""),
        api_key=os.getenv("OPENAI_API_KEY", ""),
        restrict_to_conv=True,
    )

    # Stream conversations into memory (we only index user turns; add_turn handles that)
    for convo in DATA:
        conv_id = convo["conversation_id"]
        print(f"\n=== Conversation {conv_id} ===")
        for i, turn in enumerate(convo["turns"], start=1):
            mem.add_turn(conv_id, i, turn["role"], turn["content"])

        # Simple evaluation phase
        for q in convo.get("eval", []):
            query = q["question"]
            want = _normalize(q["expected_answer"])
            res = mem.retrieve(conv_id, query, top_k=3)

            got = ""
            if res.get("results"):
                # Take the best hit, attempt to extract the answer
                got = _extract_answer_from_hit(res["results"][0])
            got_norm = _normalize(got)

            status = "PASS" if got_norm == want else "FAIL"
            print(f"Q: {query}\n   expected={q['expected_answer']} | got={got}  [{status}]")

    # Optionally clear the collection between runs (disabled by default)
    # mem.reset()


if __name__ == "__main__":
    run_demo()
