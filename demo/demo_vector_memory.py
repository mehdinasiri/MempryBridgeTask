# demo_vector_memory_llm.py
"""
Demo: pass RAW user text to VectorMemory.add_turn(...).
VectorMemory extracts facts with an LLM (OpenAI SDK), embeds, and stores them.
Requires your .env to define:
  - OPENAI_BASE_URL
  - OPENAI_API_KEY
  - MB_CHAT_MODEL
  - MB_EMBED_MODEL
Optional:
  - LOG_LEVEL=DEBUG (to see detailed logs)
"""

from memory.vector_memory import VectorMemory

DATA = [
    {
        "conversation_id": "001",
        "turns": [
            {"role": "user", "content": "My name is Alice and I work at OpenAI."},
            {"role": "assistant", "content": "Nice to meet you, Alice."},
            {"role": "user", "content": "Actually, I work at Anthropic now."},
            {"role": "assistant", "content": "Got it."}
        ],
        "eval": [
            {"question": "Where does Alice work now?", "expected_answer": "anthropic"}
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
            {"question": "What city do I live in?", "expected_answer": "san francisco"}
        ]
    }
]

def main():
    # Use your real backend (Chroma by default). To use LanceDB, set index_backend="lancedb" and persist_path=".memdb/lancedb".
    mem = VectorMemory(
        name="demo_vm_llm",
        index_backend="chroma",
        collection_or_table="demo_facts_llm",
        persist_path=".memdb/chroma",
        restrict_to_conv=True,
    )

    # Stream turns into memory (we only index user turns; add_turn handles extraction)
    for convo in DATA:
        conv_id = convo["conversation_id"]
        print(f"\n=== Conversation {conv_id} ===")
        for i, turn in enumerate(convo["turns"], start=1):
            mem.add_turn(conv_id, i, turn["role"], turn["content"])

        # Evaluate after the conversation
        for q in convo.get("eval", []):
            res = mem.retrieve(conv_id, q["question"], top_k=1)
            got = res["results"][0]["object"].lower() if res["results"] else ""
            want = q["expected_answer"].lower()
            status = "PASS" if got == want else "FAIL"
            print(f"Q: {q['question']}\n   expected={want} got={got} [{status}]")

if __name__ == "__main__":
    main()
