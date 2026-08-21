"""Corpus-quality and vector-coverage audit helpers."""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from legal_rag.models import Case, Paragraph


def _nearest_rank(values: Sequence[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _is_missing(value: object) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def build_corpus_audit(
    session: Session, *, very_short_characters: int = 20
) -> dict[str, Any]:
    """Calculate deterministic quality metrics for the relational corpus."""

    if very_short_characters <= 0:
        raise ValueError("very_short_characters must be positive")

    cases = list(session.scalars(select(Case).order_by(Case.id)))
    paragraphs = list(session.scalars(select(Paragraph).order_by(Paragraph.id)))
    paragraphs_by_case = Counter(paragraph.case_id for paragraph in paragraphs)
    per_case = [paragraphs_by_case[case.id] for case in cases]

    document_hash_counts = Counter(case.document_hash for case in cases)
    paragraph_uid_counts = Counter(paragraph.paragraph_uid for paragraph in paragraphs)
    empty_paragraphs = sum(not paragraph.text.strip() for paragraph in paragraphs)
    very_short_paragraphs = sum(
        bool(paragraph.text.strip())
        and len(paragraph.text.strip()) < very_short_characters
        for paragraph in paragraphs
    )
    paragraphs_with_pages = sum(
        paragraph.page_number is not None for paragraph in paragraphs
    )
    paragraph_total = len(paragraphs)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "counts": {"cases": len(cases), "paragraphs": paragraph_total},
        "paragraphs_per_case": {
            "min": min(per_case, default=0),
            "median": median(per_case) if per_case else 0,
            "p95": _nearest_rank(per_case, 0.95),
            "max": max(per_case, default=0),
        },
        "missing_metadata": {
            "court": sum(_is_missing(case.court) for case in cases),
            "judgment_date": sum(case.judgment_date is None for case in cases),
            "case_number": sum(_is_missing(case.case_number) for case in cases),
            "source_url": sum(_is_missing(case.source_url) for case in cases),
        },
        "paragraph_quality": {
            "empty": empty_paragraphs,
            "very_short": very_short_paragraphs,
            "very_short_threshold_characters": very_short_characters,
        },
        "duplicates": {
            "document_hashes": sum(
                count > 1 for count in document_hash_counts.values()
            ),
            "document_rows_beyond_first": sum(
                max(0, count - 1) for count in document_hash_counts.values()
            ),
            "paragraph_uids": sum(
                count > 1 for count in paragraph_uid_counts.values()
            ),
            "paragraph_rows_beyond_first": sum(
                max(0, count - 1) for count in paragraph_uid_counts.values()
            ),
        },
        "page_number_coverage": {
            "with_page_number": paragraphs_with_pages,
            "without_page_number": paragraph_total - paragraphs_with_pages,
            "percentage": (
                round(100.0 * paragraphs_with_pages / paragraph_total, 2)
                if paragraph_total
                else 0.0
            ),
        },
    }


def compare_vector_coverage(
    postgres_paragraph_uids: Sequence[str], qdrant_point_ids: Sequence[str]
) -> dict[str, Any]:
    """Compare durable paragraph identities across PostgreSQL and Qdrant."""

    postgres_ids = [str(value) for value in postgres_paragraph_uids]
    qdrant_ids = [str(value) for value in qdrant_point_ids]
    postgres_set = set(postgres_ids)
    qdrant_set = set(qdrant_ids)
    missing = sorted(postgres_set - qdrant_set)
    stale = sorted(qdrant_set - postgres_set)

    return {
        "postgres_paragraphs": len(postgres_ids),
        "postgres_unique_uids": len(postgres_set),
        "qdrant_points": len(qdrant_ids),
        "qdrant_unique_point_ids": len(qdrant_set),
        "missing_points": len(missing),
        "stale_orphan_points": len(stale),
        "duplicate_points": len(qdrant_ids) - len(qdrant_set),
        "missing_point_id_sample": missing[:20],
        "stale_point_id_sample": stale[:20],
    }


def paragraph_uids(session: Session) -> list[str]:
    """Return every PostgreSQL paragraph UID in stable numeric-ID order."""

    return list(
        session.scalars(select(Paragraph.paragraph_uid).order_by(Paragraph.id))
    )


def write_audit(path: Path, audit: dict[str, Any]) -> None:
    """Atomically write a human-readable tracked audit."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
