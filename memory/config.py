from __future__ import annotations
import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    base_url: str = os.getenv("OPENAI_BASE_URL", "").rstrip("/")
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    chat_model: str = os.getenv("MB_CHAT_MODEL", "gemini-flash")
    emb_model: str = os.getenv("MB_EMBED_MODEL", "gemini-embedding")
    emb_dims: int = int(os.getenv("MB_EMBED_DIMS", "3072"))
    db_path: str = os.getenv("DB_PATH", ".memdb/memory.sqlite")
    collection: str = os.getenv("MB_COLLECTION", f"mem0_{os.getenv('MB_EMBED_MODEL','gemini-embedding')}")

    def require(self) -> "Settings":
        if not self.base_url:
            raise RuntimeError("OPENAI_BASE_URL missing")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY missing")
        return self

SETTINGS = Settings()
BASE_URL = SETTINGS.base_url
MB_API_KEY = SETTINGS.api_key
CHAT_MODEL = SETTINGS.chat_model
EMB_MODEL = SETTINGS.emb_model
EMB_DIMS = SETTINGS.emb_dims
DB_PATH = SETTINGS.db_path