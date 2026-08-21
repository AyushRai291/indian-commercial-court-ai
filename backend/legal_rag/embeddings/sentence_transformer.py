"""Sentence Transformers implementation of the embedding interface."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from legal_rag.embeddings.base import Embedding, EmbeddingProvider


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Generate embeddings with a model loaded only on first use.

    Keeping the model import and initialization lazy makes management scripts and
    unit tests which do not need embeddings start without importing PyTorch.
    """

    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        normalize_embeddings: bool = True,
        expected_dimension: int | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name must not be empty")
        if expected_dimension is not None and expected_dimension <= 0:
            raise ValueError("expected_dimension must be positive")

        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.expected_dimension = expected_dimension
        self._model: Any | None = None
        self._dimension: int | None = None

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - dependency error path
                raise RuntimeError(
                    "sentence-transformers is required to create embeddings"
                ) from exc

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            model = self._load_model()
            dimension_getter = getattr(model, "get_embedding_dimension", None)
            dimension = (
                dimension_getter()
                if callable(dimension_getter)
                else model.get_sentence_embedding_dimension()
            )
            if dimension is None:
                probe = self.embed_documents(["embedding dimension probe"], batch_size=1)
                dimension = len(probe[0])
            actual_dimension = int(dimension)
            self._validate_dimension(actual_dimension)
            self._dimension = actual_dimension
        return self._dimension

    def _validate_dimension(self, actual_dimension: int) -> None:
        if actual_dimension <= 0:
            raise RuntimeError("The embedding model returned an empty vector")
        if (
            self.expected_dimension is not None
            and actual_dimension != self.expected_dimension
        ):
            raise ValueError(
                f"Embedding model {self.model_name!r} produces {actual_dimension} "
                f"dimensions, but {self.expected_dimension} was configured"
            )

    def embed_documents(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
    ) -> list[Embedding]:
        if isinstance(texts, str):
            raise TypeError("texts must be a sequence of documents, not one string")
        values = list(texts)
        if not values:
            return []
        if any(not isinstance(text, str) for text in values):
            raise TypeError("All documents must be strings")
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive")

        encoded = self._load_model().encode(
            values,
            batch_size=batch_size or 32,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        raw_vectors = encoded.tolist() if hasattr(encoded, "tolist") else encoded
        vectors = [[float(value) for value in vector] for vector in raw_vectors]
        if len(vectors) != len(values):
            raise RuntimeError(
                "The embedding model returned a different number of vectors than inputs"
            )

        actual_dimension = len(vectors[0])
        if any(len(vector) != actual_dimension for vector in vectors):
            raise RuntimeError("The embedding model returned inconsistent dimensions")
        self._validate_dimension(actual_dimension)
        self._dimension = actual_dimension
        return vectors


# A concise alias for callers which prefer provider names without the suffix.
SentenceTransformerProvider = SentenceTransformerEmbeddingProvider
