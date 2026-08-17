"""Environment-backed application settings.

The module intentionally has no dependency on a dotenv or validation package.
Containers and command-line scripts can inject environment variables directly,
while tests can construct :class:`Settings` explicitly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


def _positive_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc

    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings used by database and vector scripts."""

    database_url: str = (
        "postgresql+psycopg://legal_rag:legal_rag_dev_password@localhost:5432/legal_rag"
    )
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "legal_paragraphs"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    embedding_batch_size: int = 64

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from environment variables."""

        return cls(
            database_url=os.getenv(
                "DATABASE_URL",
                "postgresql+psycopg://legal_rag:legal_rag_dev_password@localhost:5432/legal_rag",
            ),
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,
            qdrant_collection=os.getenv(
                "QDRANT_COLLECTION", "legal_paragraphs"
            ),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
            embedding_dimension=_positive_int("EMBEDDING_DIMENSION", 384),
            embedding_batch_size=_positive_int("EMBEDDING_BATCH_SIZE", 64),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide immutable settings object."""

    return Settings.from_env()
