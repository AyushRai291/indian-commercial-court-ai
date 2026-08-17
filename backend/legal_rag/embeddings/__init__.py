"""Embedding provider abstractions."""

from legal_rag.embeddings.base import Embedding, EmbeddingProvider
from legal_rag.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingProvider,
    SentenceTransformerProvider,
)

__all__ = [
    "Embedding",
    "EmbeddingProvider",
    "SentenceTransformerEmbeddingProvider",
    "SentenceTransformerProvider",
]
