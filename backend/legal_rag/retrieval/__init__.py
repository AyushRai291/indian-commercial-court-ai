"""Dense, BM25, and rank-fused paragraph retrieval interfaces."""

from legal_rag.retrieval.bm25 import (
    BM25ParagraphRetriever,
    ParagraphDocument,
    tokenize_legal_text,
)
from legal_rag.retrieval.dense import (
    DenseParagraphRetriever,
    build_qdrant_filter,
    search_dense,
    semantic_hits_to_results,
)
from legal_rag.retrieval.filters import (
    RetrievalFilters,
    build_retrieval_filters,
    normalize_metadata_value,
)
from legal_rag.retrieval.hybrid import (
    DEFAULT_CANDIDATE_DEPTH,
    DEFAULT_RRF_K,
    HybridParagraphRetriever,
    HybridSearchDiagnostics,
    RankedParagraphRetriever,
    reciprocal_rank_fusion,
)
from legal_rag.retrieval.rerank import (
    CrossEncoderBatchResult,
    CrossEncoderReranker,
    CrossEncoderScorer,
    RerankSearchDiagnostics,
    SentenceTransformerCrossEncoderScorer,
)
from legal_rag.retrieval.results import (
    HybridSearchResult,
    ParagraphSearchResult,
    RerankedSearchResult,
)

__all__ = [
    "BM25ParagraphRetriever",
    "DEFAULT_CANDIDATE_DEPTH",
    "DEFAULT_RRF_K",
    "DenseParagraphRetriever",
    "CrossEncoderBatchResult",
    "CrossEncoderReranker",
    "CrossEncoderScorer",
    "HybridParagraphRetriever",
    "HybridSearchDiagnostics",
    "HybridSearchResult",
    "ParagraphDocument",
    "ParagraphSearchResult",
    "RankedParagraphRetriever",
    "RetrievalFilters",
    "RerankSearchDiagnostics",
    "RerankedSearchResult",
    "SentenceTransformerCrossEncoderScorer",
    "build_qdrant_filter",
    "build_retrieval_filters",
    "normalize_metadata_value",
    "reciprocal_rank_fusion",
    "search_dense",
    "semantic_hits_to_results",
    "tokenize_legal_text",
]
