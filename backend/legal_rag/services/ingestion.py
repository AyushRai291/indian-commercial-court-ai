"""Transactional insertion of canonical cases and their paragraphs."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from legal_rag.corpus import (
    CanonicalCase,
    deduplicate_paragraphs,
    extract_paragraphs,
)
from legal_rag.models import Case, Paragraph


@dataclass(frozen=True, slots=True)
class InsertionResult:
    """Outcome of attempting to insert one canonical case."""

    inserted: bool
    case_id: int | None
    reason: str | None
    paragraphs_inserted: int


def _existing_case_id(session: Session, hash_value: str) -> int | None:
    return session.scalar(
        select(Case.id).where(Case.document_hash == hash_value).limit(1)
    )


def insert_case(session: Session, canonical_case: CanonicalCase) -> InsertionResult:
    """Insert a case and unique paragraphs, leaving commit to the caller.

    The insert uses a savepoint so a concurrent unique-hash conflict does not
    poison the caller's surrounding transaction. Other database errors are
    propagated for the ingestion script to record and continue from safely.
    """

    if not isinstance(canonical_case, CanonicalCase):
        raise TypeError("canonical_case must be a CanonicalCase")

    existing_id = _existing_case_id(session, canonical_case.document_hash)
    if existing_id is not None:
        return InsertionResult(
            inserted=False,
            case_id=existing_id,
            reason="duplicate_document_hash",
            paragraphs_inserted=0,
        )

    paragraphs = deduplicate_paragraphs(
        extract_paragraphs(canonical_case.raw_text)
    )
    db_case = Case(
        title=canonical_case.title,
        case_number=canonical_case.case_number,
        court=canonical_case.court,
        judgment_date=canonical_case.judgment_date,
        source=canonical_case.source,
        source_url=canonical_case.source_url,
        raw_text=canonical_case.raw_text,
        document_hash=canonical_case.document_hash,
    )

    try:
        with session.begin_nested():
            session.add(db_case)
            session.flush()
            session.add_all(
                Paragraph(
                    case_id=db_case.id,
                    paragraph_number=paragraph.paragraph_number,
                    page_number=paragraph.page_number,
                    text=paragraph.text,
                    text_hash=paragraph.text_hash,
                )
                for paragraph in paragraphs
            )
            session.flush()
    except IntegrityError:
        # The document-hash uniqueness constraint is the expected race. The
        # nested transaction has already rolled back, leaving the outer one valid.
        concurrent_id = _existing_case_id(session, canonical_case.document_hash)
        if concurrent_id is None:
            raise
        return InsertionResult(
            inserted=False,
            case_id=concurrent_id,
            reason="duplicate_document_hash",
            paragraphs_inserted=0,
        )

    return InsertionResult(
        inserted=True,
        case_id=db_case.id,
        reason=None,
        paragraphs_inserted=len(paragraphs),
    )
