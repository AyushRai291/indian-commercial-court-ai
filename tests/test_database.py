from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import sessionmaker

from legal_rag.corpus import normalize_record
from legal_rag.database import get_engine, get_session_factory, init_db
from legal_rag.models import Case, Paragraph, Statute
from legal_rag.services.ingestion import insert_case


@pytest.fixture
def sqlite_database() -> Iterator[tuple[Engine, sessionmaker]]:
    engine = get_engine("sqlite+pysqlite:///:memory:")
    init_db(engine)
    session_factory = get_session_factory(engine)
    try:
        yield engine, session_factory
    finally:
        engine.dispose()


def test_insert_case_persists_case_and_paragraphs(
    sqlite_database: tuple[Engine, sessionmaker],
) -> None:
    _, session_factory = sqlite_database
    canonical = normalize_record(
        {
            "title": "Acme Ltd. v. Zenith Ltd.",
            "case_number": "CS(COMM) 101/2025",
            "court": "Delhi High Court",
            "judgment_date": "2025-03-04",
            "source": "test-corpus",
            "source_url": "https://example.test/cases/101",
            "raw_text": (
                "1. The parties entered into a supply agreement.\n\n"
                "2. The defendant committed a material breach."
            ),
        }
    )

    with session_factory.begin() as session:
        result = insert_case(session, canonical)

    assert result.inserted is True
    assert result.case_id is not None
    assert result.reason is None
    assert result.paragraphs_inserted == 2

    with session_factory() as session:
        stored_case = session.scalars(select(Case)).one()
        stored_paragraphs = session.scalars(
            select(Paragraph).order_by(Paragraph.paragraph_number)
        ).all()

        assert stored_case.id == result.case_id
        assert stored_case.title == "Acme Ltd. v. Zenith Ltd."
        assert stored_case.document_hash == canonical.document_hash
        assert [paragraph.text for paragraph in stored_paragraphs] == [
            "The parties entered into a supply agreement.",
            "The defendant committed a material breach.",
        ]
        assert all(paragraph.case_id == stored_case.id for paragraph in stored_paragraphs)


def test_insert_case_skips_equivalent_document_hash(
    sqlite_database: tuple[Engine, sessionmaker],
) -> None:
    _, session_factory = sqlite_database
    first = normalize_record(
        {
            "title": "First source title",
            "raw_text": "1. The arbitral award is upheld.",
            "source": "source-a",
        }
    )
    duplicate = normalize_record(
        {
            "title": "Alternate source title",
            "raw_text": "  1.  The arbitral award is upheld.  ",
            "source": "source-b",
        }
    )

    with session_factory.begin() as session:
        first_result = insert_case(session, first)
    with session_factory.begin() as session:
        duplicate_result = insert_case(session, duplicate)

    assert first_result.inserted is True
    assert duplicate_result.inserted is False
    assert duplicate_result.reason == "duplicate_document_hash"
    assert duplicate_result.paragraphs_inserted == 0

    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(Case)) == 1
        assert session.scalar(select(func.count()).select_from(Paragraph)) == 1


def test_statute_model_can_be_inserted_in_sqlite(
    sqlite_database: tuple[Engine, sessionmaker],
) -> None:
    _, session_factory = sqlite_database

    with session_factory.begin() as session:
        session.add(
            Statute(
                act_name="Commercial Courts Act, 2015",
                section="12A",
                title="Pre-Institution Mediation and Settlement",
                text="A suit shall not be instituted unless the remedy is exhausted.",
            )
        )

    with session_factory() as session:
        statute = session.scalars(select(Statute)).one()
        assert statute.id is not None
        assert statute.act_name == "Commercial Courts Act, 2015"
        assert statute.section == "12A"

