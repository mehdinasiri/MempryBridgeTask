# tests/test_memory_api.py
import importlib
from typing import Dict, List, Any
import pytest
from fastapi.testclient import TestClient


class FakeMemory:
    """
    Minimal fake memory for tests.
    Stores turns per conversation and echoes the last one in retrieve().
    """
    def __init__(self, name: str):
        self.name = name
        self.data: Dict[str, List[Dict[str, Any]]] = {}

    def add_turn(self, conv_id: str, turn_id: int, role: str, content: str):
        self.data.setdefault(conv_id, []).append(
            {"turn_id": turn_id, "role": role, "content": content}
        )

    def retrieve(self, conv_id: str, query: str, top_k: int = 3) -> Dict[str, Any]:
        items = self.data.get(conv_id, [])
        text = items[-1]["content"] if items else ""
        return {
            "results": [
                {"subject": "user", "predicate": "says", "object": text, "text": text, "score": 1.0}
            ][:top_k]
        }


@pytest.fixture(scope="session")
def client():
    """
    Import the FastAPI app from api/memory_api.py
    and replace the real memories with fakes.
    """
    app_module = importlib.import_module("api.memory_api")

    # swap in fakes
    app_module.graph_mem = FakeMemory("graph")
    app_module.keyword_mem = FakeMemory("keyword")
    app_module.llamaindex_mem = FakeMemory("llama")
    app_module.mem0_mem = FakeMemory("mem0")
    app_module.vector_mem = FakeMemory("vector")

    return TestClient(app_module.app)


def _exercise_pair(client: TestClient, base: str):
    """
    Helper: insert a turn, then retrieve it, and assert structure/content.
    """
    conv_id = f"conv_{base.strip('/')}"
    payload_insert = {
        "conv_id": conv_id,
        "turn_id": 1,
        "role": "user",
        "content": f"Hello from {base}!"
    }

    # Insert
    r_ins = client.post(f"{base}/insert", json=payload_insert)
    assert r_ins.status_code == 200
    body_ins = r_ins.json()
    assert body_ins.get("status") == "ok"
    assert body_ins.get("conv_id") == conv_id

    # Retrieve
    r_ret = client.post(f"{base}/retrieve", json={
        "conv_id": conv_id,
        "query": "echo last message",
        "top_k": 1
    })
    assert r_ret.status_code == 200
    body_ret = r_ret.json()
    assert "results" in body_ret
    top = body_ret["results"][0]
    assert top.get("object") == f"Hello from {base}!"
    assert top.get("subject") == "user"
    assert top.get("predicate") == "says"


def test_graph_endpoints(client): _exercise_pair(client, "/graph")
def test_keyword_endpoints(client): _exercise_pair(client, "/keyword")
def test_llama_endpoints(client): _exercise_pair(client, "/llama")
def test_mem0_endpoints(client): _exercise_pair(client, "/mem0")
def test_vector_endpoints(client): _exercise_pair(client, "/vector")
