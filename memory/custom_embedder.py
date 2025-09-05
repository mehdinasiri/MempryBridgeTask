from __future__ import annotations
from typing import List, Union, Optional
import os, json, requests
from types import SimpleNamespace
from .config import BASE_URL, MB_API_KEY, EMB_MODEL, EMB_DIMS

# ---- Simple embeddings helper used by Local backend -----------------------

def get_embedding(text: Union[str, List[str]]) -> List[float]:
    if isinstance(text, list):
        if len(text) != 1:
            raise ValueError("get_embedding expects a single string or single-item list.")
        text = text[0]
    url = f"{BASE_URL}/embeddings"
    headers = {"Authorization": f"Bearer {MB_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": EMB_MODEL, "input": text}
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
    r.raise_for_status()
    data = r.json()
    return data["data"][0]["embedding"]

# ---- mem0-compatible embedder used by MemZero backend --------------------

class CustomProxyEmbedder:
    """OpenAI-compatible embeddings client that matches mem0's expectations."""
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None, model: str = EMB_MODEL, embedding_dims: Optional[int] = EMB_DIMS, timeout: int = 60):
        self.base_url = (base_url or BASE_URL).rstrip("/")
        self.api_key = api_key or MB_API_KEY
        self.model = model
        self.timeout = timeout
        self.config = SimpleNamespace(model=self.model, embedding_dims=embedding_dims)
        if not self.base_url:
            raise ValueError("CustomProxyEmbedder: OPENAI_BASE_URL is not set")
        if not self.api_key:
            raise ValueError("CustomProxyEmbedder: OPENAI_API_KEY is not set")
    def get_model(self) -> str:
        return self.model
    def _post(self, inputs: List[str]) -> List[List[float]]:
        url = f"{self.base_url}/embeddings"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "input": inputs}
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        items = sorted(data.get("data", []), key=lambda d: d.get("index", 0))
        vecs = [it["embedding"] for it in items]
        if len(vecs) != len(inputs):
            raise RuntimeError("Embedding count mismatch")
        return vecs
    def embed(self, text: Union[str, List[str]], operation: Optional[str] = None):
        if isinstance(text, str):
            return self._post([text])[0]
        elif isinstance(text, list):
            return self._post(text)
        else:
            raise TypeError("embed(text) expects str or List[str]")