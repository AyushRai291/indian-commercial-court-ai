from __future__ import annotations

from datetime import date
from pathlib import Path

from legal_rag.corpus_audit import build_corpus_audit, compare_vector_coverage
from legal_rag.database import get_engine, get_session_factory, init_db
from legal_rag.models import Case, Paragraph


def test_corpus_audit_metrics_with_temporary_database(tmp_path: Path) -> None:
    engine = get_engine(f"sqlite+pysqlite:///{(tmp_path / 'audit.db').as_posix()}")
    init_db(engine)
    session_factory = get_session_factory(engine)
    try:
        with session_factory.begin() as session:
            complete = Case(
                title="Complete Ltd. v. Example Ltd.",
                case_number="CA 1/2024",
                court="Supreme Court of India",
                judgment_date=date(2024, 1, 1),
                source="test",
                source_url="https://example.test/complete.pdf",
                raw_text="Complete",
                document_hash="a" * 64,
            )
            one_paragraph = Case(
                title="One paragraph",
                case_number="CA 2/2024",
                court="Supreme Court of India",
                judgment_date=date(2024, 1, 2),
                source="test",
                source_url="https://example.test/one.pdf",
                raw_text="Short",
                document_hash="b" * 64,
            )
            missing = Case(
                title="Missing metadata",
                case_number=None,
                court=" ",
                judgment_date=None,
                source="test",
                source_url=None,
                raw_text="No paragraphs",
                document_hash="c" * 64,
            )
            session.add_all([complete, one_paragraph, missing])
            session.flush()
            session.add_all(
                [
                    Paragraph(
                        paragraph_uid="00000000-0000-5000-8000-000000000001",
                        case_id=complete.id,
                        paragraph_number=1,
                        page_number=1,
                        text="",
                        text_hash="d" * 64,
                    ),
                    Paragraph(
                        paragraph_uid="00000000-0000-5000-8000-000000000002",
                        case_id=complete.id,
                        paragraph_number=2,
                        page_number=None,
                        text="A sufficiently long legal paragraph.",
                        text_hash="e" * 64,
                    ),
                    Paragraph(
                        paragraph_uid="00000000-0000-5000-8000-000000000003",
                        case_id=one_paragraph.id,
                        paragraph_number=1,
                        page_number=None,
                        text="Short",
                        text_hash="f" * 64,
                    ),
                ]
            )

        with session_factory() as session:
            audit = build_corpus_audit(session, very_short_characters=10)
    finally:
        engine.dispose()

    assert audit["counts"] == {"cases": 3, "paragraphs": 3}
    assert audit["paragraphs_per_case"] == {
        "min": 0,
        "median": 1,
        "p95": 2,
        "max": 2,
    }
    assert audit["missing_metadata"] == {
        "court": 1,
        "judgment_date": 1,
        "case_number": 1,
        "source_url": 1,
    }
    assert audit["paragraph_quality"]["empty"] == 1
    assert audit["paragraph_quality"]["very_short"] == 1
    assert audit["duplicates"]["document_hashes"] == 0
    assert audit["duplicates"]["paragraph_uids"] == 0
    assert audit["page_number_coverage"] == {
        "with_page_number": 1,
        "without_page_number": 2,
        "percentage": 33.33,
    }


def test_vector_coverage_reports_missing_stale_and_duplicate_points() -> None:
    coverage = compare_vector_coverage(
        ["uid-a", "uid-b"],
        ["uid-b", "uid-c", "uid-c"],
    )

    assert coverage["postgres_paragraphs"] == 2
    assert coverage["qdrant_points"] == 3
    assert coverage["missing_points"] == 1
    assert coverage["stale_orphan_points"] == 1
    assert coverage["duplicate_points"] == 1
    assert coverage["indexing_coverage_percentage"] == 50.0
    assert coverage["missing_point_id_sample"] == ["uid-a"]
    assert coverage["stale_point_id_sample"] == ["uid-c"]
