"""Interfaces shared by embedding implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


Embedding = list[float]


class EmbeddingProvider(ABC):
    """Backend-independent interface for turning text into dense vectors."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the number of values in each produced embedding."""

    @abstractmethod
    def embed_documents(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
    ) -> list[Embedding]:
        """Embed documents while preserving their input order."""

    def embed_query(self, query: str) -> Embedding:
        """Embed a single search query."""

        if not query.strip():
            raise ValueError("The search query must not be empty")
        return self.embed_documents([query], batch_size=1)[0]
