"""Canonical corpus schema and normalization utilities."""

from legal_rag.corpus.normalization import NormalizationError, normalize_record
from legal_rag.corpus.paragraphs import (
    CanonicalParagraph,
    deduplicate_paragraphs,
    extract_paragraphs,
)
from legal_rag.corpus.schema import CanonicalCase
from legal_rag.corpus.text import document_hash, normalize_text, sha256_text

__all__ = [
    "CanonicalCase",
    "CanonicalParagraph",
    "NormalizationError",
    "deduplicate_paragraphs",
    "document_hash",
    "extract_paragraphs",
    "normalize_record",
    "normalize_text",
    "sha256_text",
]
