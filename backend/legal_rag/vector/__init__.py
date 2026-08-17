"""Vector index adapters."""

from legal_rag.vector.qdrant_index import (
    ParagraphVectorRecord,
    QdrantParagraphIndex,
    SemanticSearchResult,
)

__all__ = [
    "ParagraphVectorRecord",
    "QdrantParagraphIndex",
    "SemanticSearchResult",
]
