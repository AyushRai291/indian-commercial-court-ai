"""Independent dense and BM25 paragraph retrieval interfaces."""

from legal_rag.retrieval.bm25 import (
    BM25ParagraphRetriever,
    ParagraphDocument,
    tokenize_legal_text,
)
from legal_rag.retrieval.dense import search_dense, semantic_hits_to_results
from legal_rag.retrieval.results import ParagraphSearchResult

__all__ = [
    "BM25ParagraphRetriever",
    "ParagraphDocument",
    "ParagraphSearchResult",
    "search_dense",
    "semantic_hits_to_results",
    "tokenize_legal_text",
]
