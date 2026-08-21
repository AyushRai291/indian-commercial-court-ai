from __future__ import annotations

from pathlib import Path
from uuid import RFC_4122, UUID

import pytest
from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from legal_rag.corpus import generate_paragraph_uid, normalize_record
from legal_rag.database import get_engine, get_session_factory, init_db
from legal_rag.models import Case, Paragraph
from legal_rag.services.ingestion import insert_case


def _create_database(path: Path) -> tuple[Engine, sessionmaker]:
    engine = get_engine(f"sqlite+pysqlite:///{path.as_posix()}")
    init_db(engine)
    return engine, get_session_factory(engine)


def _stored_paragraph_identity(
    session_factory: sessionmaker,
    document_hash: str,
) -> tuple[int, list[tuple[int, str, int, str]]]:
    with session_factory() as session:
        stored_case = session.scalars(
            select(Case).where(Case.document_hash == document_hash)
        ).one()
        paragraphs = session.scalars(
            select(Paragraph)
            .where(Paragraph.case_id == stored_case.id)
            .order_by(Paragraph.paragraph_number)
        ).all()
        return stored_case.id, [
            (
                paragraph.id,
                paragraph.paragraph_uid,
                paragraph.paragraph_number,
                paragraph.text_hash,
            )
            for paragraph in paragraphs
        ]


def test_paragraph_uid_is_a_deterministic_uuid5() -> None:
    document_hash = "a" * 64
    text_hash = "b" * 64

    first = generate_paragraph_uid(document_hash, 12, text_hash)
    second = generate_paragraph_uid(document_hash, 12, text_hash)
    parsed = UUID(first)

    assert first == second
    assert first == "57eb9eb2-206c-572e-bc7a-f8b57f606d4a"
    assert str(parsed) == first
    assert parsed.version == 5
    assert parsed.variant == RFC_4122


@pytest.mark.parametrize(
    ("document_hash", "paragraph_number", "text_hash"),
    [
        ("", 1, "b" * 64),
        ("a" * 63, 1, "b" * 64),
        ("g" * 64, 1, "b" * 64),
        ("a" * 64, "1", "b" * 64),
        ("a" * 64, 1, "not-a-sha256"),
    ],
)
def test_paragraph_uid_rejects_invalid_identity_inputs(
    document_hash: object,
    paragraph_number: object,
    text_hash: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        generate_paragraph_uid(document_hash, paragraph_number, text_hash)  # type: ignore[arg-type]


def test_different_paragraph_identity_components_produce_different_uids() -> None:
    document_hash = "a" * 64
    text_hash = "b" * 64

    first = generate_paragraph_uid(document_hash, 1, text_hash)
    different_number = generate_paragraph_uid(document_hash, 2, text_hash)
    different_text = generate_paragraph_uid(document_hash, 1, "c" * 64)

    assert len({first, different_number, different_text}) == 3


def test_same_judgment_has_stable_uids_when_database_ids_differ(
    tmp_path: Path,
) -> None:
    target_record = {
        "title": "Acme Pvt. Ltd. v. Zenith Ltd.",
        "case_number": "CS(COMM) 42/2025",
        "court": "Delhi High Court",
        "raw_text": (
            "1. The supply agreement bound both parties.\n\n"
            "2. The plaintiff was entitled to its contractual remedy."
        ),
    }
    unrelated_record = {
        "title": "Unrelated Industries Ltd. v. Example Ltd.",
        "raw_text": "1. This unrelated judgment is inserted first.",
    }
    first_canonical = normalize_record(dict(target_record))
    second_canonical = normalize_record(dict(target_record))
    unrelated = normalize_record(unrelated_record)
    first_engine, first_factory = _create_database(tmp_path / "first.sqlite3")
    second_engine, second_factory = _create_database(tmp_path / "second.sqlite3")

    try:
        with first_factory.begin() as session:
            first_result = insert_case(session, first_canonical)
        with second_factory.begin() as session:
            insert_case(session, unrelated)
        with second_factory.begin() as session:
            second_result = insert_case(session, second_canonical)

        assert first_result.inserted is True
        assert second_result.inserted is True

        first_case_id, first_identities = _stored_paragraph_identity(
            first_factory, first_canonical.document_hash
        )
        second_case_id, second_identities = _stored_paragraph_identity(
            second_factory, second_canonical.document_hash
        )

        assert first_case_id != second_case_id
        assert [identity[0] for identity in first_identities] != [
            identity[0] for identity in second_identities
        ]
        assert [identity[1] for identity in first_identities] == [
            identity[1] for identity in second_identities
        ]
        assert len({identity[1] for identity in first_identities}) == 2

        for _, paragraph_uid, paragraph_number, text_hash in first_identities:
            assert paragraph_uid == generate_paragraph_uid(
                first_canonical.document_hash,
                paragraph_number,
                text_hash,
            )
    finally:
        first_engine.dispose()
        second_engine.dispose()
