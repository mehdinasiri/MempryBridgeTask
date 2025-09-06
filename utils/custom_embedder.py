# repo/utils/custom_embedder.py
from __future__ import annotations

from typing import List, Union, Optional, Iterable
import time
import requests

from loguru import logger

from utils.config import load_config


def _as_list(text: Union[str, List[str]]) -> List[str]:
    if isinstance(text, str):
        return [text]
    if isinstance(text, list) and all(isinstance(t, str) for t in text):
        return text
    raise TypeError("embed(text) expects str or List[str]")


def _chunks(seq: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


class CustomProxyEmbedder:
    """
    OpenAI-compatible embeddings client (works with MemoryBridge/LiteLLM proxy too).

    - Reads defaults from src.config.load_config()
    - POSTs to: <base_url>/embeddings  with payload {"model": <model>, "input": [...]}
    - Returns embeddings preserving input order
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 60,
        batch_size: int = 96,
        max_retries: int = 3,
        retry_backoff_sec: float = 0.8,
    ):
        cfg = load_config()

        self.base_url = (base_url or cfg.get("OPENAI_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or cfg.get("OPENAI_API_KEY", "")
        self.model = model or cfg.get("MB_EMBED_MODEL", "text-embedding-3-small")

        if not self.base_url:
            raise ValueError("CustomProxyEmbedder: OPENAI_BASE_URL is not set")
        if not self.api_key:
            raise ValueError("CustomProxyEmbedder: OPENAI_API_KEY is not set")

        self.timeout = timeout
        self.batch_size = max(1, int(batch_size))
        self.max_retries = max(1, int(max_retries))
        self.retry_backoff_sec = float(retry_backoff_sec)

        self._url = f"{self.base_url}/embeddings"
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self._session = requests.Session()

        logger.debug(
            f"CustomProxyEmbedder init: url={self._url} model={self.model} "
            f"timeout={self.timeout}s batch_size={self.batch_size}"
        )

    # ------------- Public API -------------

    def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Embed a string or list of strings. Preserves order.

        Returns:
          - List[float] for a single string
          - List[List[float]] for a list input
        """
        inputs = _as_list(text)
        single = isinstance(text, str)

        if not inputs:
            return [] if single else []

        # Batch to avoid huge payloads
        all_embeddings: List[List[float]] = []
        for batch in _chunks(inputs, self.batch_size):
            embs = self._post_with_retry(batch)
            all_embeddings.extend(embs)

        # Sanity check: shapes must match
        if len(all_embeddings) != len(inputs):
            raise RuntimeError(
                f"Embedding count mismatch: expected {len(inputs)} got {len(all_embeddings)}"
            )

        return all_embeddings[0] if single else all_embeddings

    # ------------- Internals -------------

    def _post_with_retry(self, inputs: List[str]) -> List[List[float]]:
        last_err: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                return self._post_once(inputs)
            except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as e:
                last_err = e
                wait = self.retry_backoff_sec * (2 ** (attempt - 1))
                # Log a concise hint; include status code/text snippet if available
                status = getattr(getattr(e, "response", None), "status_code", None)
                text = getattr(getattr(e, "response", None), "text", "")
                snippet = (text[:200] + "…") if text and len(text) > 220 else text
                logger.warning(
                    f"[embed] attempt {attempt}/{self.max_retries} failed "
                    f"(status={status}): {type(e).__name__} {snippet!r} "
                    f"— retrying in {wait:.1f}s"
                )
                time.sleep(wait)
            except Exception as e:
                # Non-network errors: don't retry blindly
                raise RuntimeError(f"Embedding request failed: {e}") from e

        # Out of retries
        assert last_err is not None
        raise RuntimeError(f"Embedding request failed after {self.max_retries} attempts") from last_err

    def _post_once(self, inputs: List[str]) -> List[List[float]]:
        payload = {"model": self.model, "input": inputs}
        resp = self._session.post(
            self._url,
            headers=self._headers,
            json=payload,             # let requests serialize
            timeout=self.timeout,
        )
        resp.raise_for_status()

        data = resp.json()
        # OpenAI-compatible shape: {"data": [{"index": i, "embedding": [...]}, ...]}
        items = data.get("data")
        if not isinstance(items, list) or not items:
            raise ValueError(f"Bad embeddings response: missing/empty 'data' field: {data!r}")

        # sort by 'index' to preserve input order
        try:
            items_sorted = sorted(items, key=lambda d: int(d.get("index", 0)))
        except Exception:
            items_sorted = items  # best-effort

        embeddings: List[List[float]] = []
        for it in items_sorted:
            emb = it.get("embedding")
            if not isinstance(emb, list) or not all(isinstance(x, (int, float)) for x in emb):
                raise ValueError(f"Bad embedding item: {it!r}")
            embeddings.append([float(x) for x in emb])

        return embeddings
