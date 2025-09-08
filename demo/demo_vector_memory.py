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

DATA = [{
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

def main():
    # Use your real backend (Chroma by default). To use LanceDB, set index_backend="lancedb" and persist_path=".memdb/lancedb".
    mem = VectorMemory(
        name="demo_vm_llm",
        index_backend="chroma",
        collection_or_table="demo_facts_llm",
        persist_path=".memdb/chroma_vector_demo",
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
