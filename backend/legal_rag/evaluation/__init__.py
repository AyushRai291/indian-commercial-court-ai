"""Gold retrieval-evaluation dataset models and validation helpers."""

from legal_rag.evaluation.gold import (
    DIFFICULTIES,
    QUERY_TYPES,
    RELEVANCE_GRADES,
    GoldDatasetStatistics,
    GoldParagraphLabel,
    GoldQuery,
    GoldValidationError,
    load_gold_queries,
    parse_gold_query,
    render_review_markdown,
    validate_gold_queries,
    write_review_markdown,
)

__all__ = [
    "DIFFICULTIES",
    "QUERY_TYPES",
    "RELEVANCE_GRADES",
    "GoldDatasetStatistics",
    "GoldParagraphLabel",
    "GoldQuery",
    "GoldValidationError",
    "load_gold_queries",
    "parse_gold_query",
    "render_review_markdown",
    "validate_gold_queries",
    "write_review_markdown",
]
