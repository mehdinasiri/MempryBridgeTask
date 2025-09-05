from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List

class AbstractMemory(ABC):
    @abstractmethod
    def connect(self, **kwargs) -> None: ...

    @abstractmethod
    def add_turn(self, *, text: str, conv_id: str, turn_id: str, user_id: str) -> Any: ...

    @abstractmethod
    def add_conversation(self, *, messages: List[Dict[str, str]], conv_id: str, user_id: str) -> Any: ...

    @abstractmethod
    def retrieve(self, *, query: str, user_id: str, k: int = 5) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def retrieve_reranked(self, *, query: str, user_id: str, k: int = 5, top_n: int = 3) -> List[Dict[str, Any]]: ...