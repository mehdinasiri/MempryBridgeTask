# repo/memory/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class MemorySystem(ABC):
    """
    Abstract interface for a memory system.
    Implementations: GraphMemory, VectorMemory, KeywordBaseline, etc.
    """

    def __init__(self, name: str, base_meta: Optional[Dict[str, Any]] = None):
        self._name = name
        self._meta = base_meta or {}

    # ---- identity ----
    @property
    def name(self) -> str:
        return self._name

    # ---- abstract API ----
    @abstractmethod
    def add_turn(self, conv_id: str, turn_id: int, role: str, text: str) -> None:
        """Ingest a single conversation turn."""
        raise NotImplementedError

    @abstractmethod
    def retrieve(
        self, conv_id: str, query_text: str, top_k: int = 5, **kwargs
    ) -> Dict[str, Any]:
        """Return retrieval results for a query."""
        raise NotImplementedError

    # ---- optional lifecycle ----
    def reset(self) -> None:
        """Clear in-memory indexes or truncate stores (optional)."""
        pass

    def close(self) -> None:
        """Release resources (optional)."""
        pass
