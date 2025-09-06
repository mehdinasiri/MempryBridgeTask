# tests/test_vector_memory.py
import pytest

from utils.models import make_fact, make_chunk
from memory.vector_memory import VectorMemory
from utils.vector_adapters import VectorIndex


class FakeIndex(VectorIndex):
    """A simple in-memory vector index for testing VectorMemory."""
    def __init__(self, embed_fn):
        self.embed_fn = embed_fn
        self._rows = []  # list of (id, text, meta, vector)

    def upsert(self, chunks):
        for ch in chunks:
            vec = self.embed_fn([ch.text])[0]
            self._rows.append((ch.chunk_id, ch.text, ch.meta, vec))

    def search(self, query, k):
        qv = self.embed_fn([query])[0]
        # very simple similarity: closer length -> higher score
        results = []
        for cid, text, meta, vec in self._rows:
            sim = 1.0 / (1.0 + abs(len(text) - len(query)))
            ch = make_chunk(text, source=meta.get("conv_id", "test"), chunk_id=cid, meta=meta)
            results.append((ch, sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max(1, k)]

    def clear(self):
        self._rows.clear()


def fake_embed(texts):
    """Return a deterministic 1D embedding: [len(text)]."""
    return [[float(len(t))] for t in texts]


def test_vector_memory_add_and_retrieve():
    # Create memory with fake index + embedder
    index = FakeIndex(embed_fn=fake_embed)
    mem = VectorMemory(name="test_vm", embed_fn=fake_embed, index=index, restrict_to_conv=True)

    # Create a fake conversation and add facts (passed via add_turn 'text' payload)
    facts = [
        make_fact("user", "works_at", "OpenAI", source="conv1:turn1"),
        make_fact("user", "lives_in", "San Francisco", source="conv1:turn1"),
    ]
    mem.add_turn("conv1", 1, "user", facts)

    # Retrieve by query
    res = mem.retrieve("conv1", "Where do I work?", top_k=3)
    assert "results" in res
    results = res["results"]
    assert len(results) >= 1

    # Should return at least one fact with predicate "works_at"
    predicates = [r.get("predicate") for r in results]
    assert "works_at" in predicates


def test_restrict_to_conv():
    index = FakeIndex(embed_fn=fake_embed)
    mem = VectorMemory(name="test_vm2", embed_fn=fake_embed, index=index, restrict_to_conv=True)

    facts1 = [make_fact("user", "works_at", "OpenAI", source="conv1:turn1")]
    facts2 = [make_fact("user", "works_at", "Anthropic", source="conv2:turn1")]

    mem.add_turn("conv1", 1, "user", facts1)
    mem.add_turn("conv2", 1, "user", facts2)

    # Query restricted to conv1 → should only see "OpenAI"
    res = mem.retrieve("conv1", "Where do I work?", top_k=5)
    objs = [r.get("object") for r in res["results"]]
    assert "OpenAI" in objs
    assert "Anthropic" not in objs
