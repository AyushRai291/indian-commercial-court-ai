from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from legal_rag.database import get_engine, get_session_factory, init_db
from legal_rag.evaluation import (
    GoldParagraphLabel,
    GoldQuery,
    GoldValidationError,
    parse_gold_query,
    render_review_markdown,
    validate_gold_queries,
)
from legal_rag.models import Case, Paragraph


UID_ONE = "00000000-0000-5000-8000-000000000001"
UID_TWO = "00000000-0000-5000-8000-000000000002"
UID_THREE = "00000000-0000-5000-8000-000000000003"


@pytest.fixture
def gold_database(tmp_path: Path) -> tuple[Engine, sessionmaker[Session]]:
    engine = get_engine(f"sqlite+pysqlite:///{(tmp_path / 'gold.db').as_posix()}")
    init_db(engine)
    factory = get_session_factory(engine)
    with factory.begin() as session:
        arbitration = Case(
            title="Alpha Ltd. v. Beta Ltd.",
            case_number="Arbitration Appeal No. 1 of 2024",
            court="Supreme Court of India",
            judgment_date=None,
            source="test",
            source_url="https://example.test/alpha.pdf",
            raw_text="Arbitration judgment.",
            document_hash="a" * 64,
        )
        insolvency = Case(
            title="Gamma Bank v. Delta Ltd.",
            case_number=None,
            court="Supreme Court of India",
            judgment_date=None,
            source="test",
            source_url="https://example.test/gamma.pdf",
            raw_text="Insolvency judgment.",
            document_hash="b" * 64,
        )
        session.add_all([arbitration, insolvency])
        session.flush()
        session.add_all(
            [
                Paragraph(
                    paragraph_uid=UID_ONE,
                    case_id=arbitration.id,
                    paragraph_number=10,
                    page_number=5,
                    text=(
                        "A party interested in the outcome cannot unilaterally "
                        "appoint the sole arbitrator."
                    ),
                    text_hash="c" * 64,
                ),
                Paragraph(
                    paragraph_uid=UID_TWO,
                    case_id=arbitration.id,
                    paragraph_number=11,
                    page_number=None,
                    text="The appointment procedure must preserve equal treatment.",
                    text_hash="d" * 64,
                ),
                Paragraph(
                    paragraph_uid=UID_THREE,
                    case_id=insolvency.id,
                    paragraph_number=20,
                    page_number=12,
                    text="The adjudicating authority must first establish default.",
                    text_hash="e" * 64,
                ),
            ]
        )
    try:
        yield engine, factory
    finally:
        engine.dispose()


def _label(
    uid: str = UID_ONE,
    *,
    relevance: int = 3,
    case_name: str = "Alpha Ltd. v. Beta Ltd.",
    case_number: str | None = "Arbitration Appeal No. 1 of 2024",
    paragraph_number: int = 10,
    page_number: int | None = 5,
) -> GoldParagraphLabel:
    return GoldParagraphLabel(
        paragraph_uid=uid,
        relevance=relevance,
        case_name=case_name,
        case_number=case_number,
        paragraph_number=paragraph_number,
        page_number=page_number,
        reason="Directly states the governing rule.",
    )


def _query(
    query_id: str = "Q001",
    *,
    query: str = "Can one party appoint the sole arbitrator?",
    labels: tuple[GoldParagraphLabel, ...] | None = None,
) -> GoldQuery:
    return GoldQuery(
        query_id=query_id,
        query=query,
        query_type="legal_principle",
        difficulty="medium",
        notes="Tests a paraphrased arbitration principle.",
        relevant_paragraphs=labels or (_label(),),
    )


@pytest.mark.parametrize(
    ("queries", "message"),
    [
        (
            (_query(), _query(query="A distinct query")),
            "duplicate query_id values: Q001",
        ),
        (
            (_query(), _query("Q002")),
            "duplicate query text in: Q001, Q002",
        ),
    ],
)
def test_validator_rejects_duplicate_query_identity(
    gold_database: tuple[Engine, sessionmaker[Session]],
    queries: tuple[GoldQuery, GoldQuery],
    message: str,
) -> None:
    _, factory = gold_database
    with factory() as session, pytest.raises(GoldValidationError, match=message):
        validate_gold_queries(queries, session, expected_count=2)


def test_validator_rejects_nonexistent_paragraph_uid(
    gold_database: tuple[Engine, sessionmaker[Session]],
) -> None:
    _, factory = gold_database
    missing = replace(_label(), paragraph_uid="00000000-0000-5000-8000-999999999999")
    with factory() as session, pytest.raises(
        GoldValidationError, match="references nonexistent paragraph_uid"
    ):
        validate_gold_queries((_query(labels=(missing,)),), session, expected_count=1)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"case_name": "Wrong title"}, "case_name mismatch"),
        ({"case_number": "Wrong number"}, "case_number mismatch"),
        ({"paragraph_number": 99}, "paragraph_number mismatch"),
        ({"page_number": 99}, "page_number mismatch"),
    ],
)
def test_validator_rejects_metadata_mismatch(
    gold_database: tuple[Engine, sessionmaker[Session]],
    change: dict[str, object],
    message: str,
) -> None:
    _, factory = gold_database
    wrong_label = replace(_label(), **change)
    with factory() as session, pytest.raises(GoldValidationError, match=message):
        validate_gold_queries(
            (_query(labels=(wrong_label,)),), session, expected_count=1
        )


def test_validator_requires_a_relevance_three_label(
    gold_database: tuple[Engine, sessionmaker[Session]],
) -> None:
    _, factory = gold_database
    with factory() as session, pytest.raises(
        GoldValidationError, match="Q001 has no relevance=3 paragraph"
    ):
        validate_gold_queries(
            (_query(labels=(_label(relevance=2),)),), session, expected_count=1
        )


def test_parser_rejects_invalid_relevance_grade() -> None:
    record = _query_record()
    record["relevant_paragraphs"][0]["relevance"] = 4

    with pytest.raises(GoldValidationError, match="relevance must be one of"):
        parse_gold_query(record)


def test_validator_rejects_duplicate_label(
    gold_database: tuple[Engine, sessionmaker[Session]],
) -> None:
    _, factory = gold_database
    with factory() as session, pytest.raises(
        GoldValidationError, match="duplicate paragraph_uid labels"
    ):
        validate_gold_queries(
            (_query(labels=(_label(), _label())),), session, expected_count=1
        )


def test_parser_rejects_empty_query() -> None:
    record = _query_record()
    record["query"] = "  "

    with pytest.raises(GoldValidationError, match="query must be a non-empty string"):
        parse_gold_query(record)


def test_validator_statistics_and_review_use_authoritative_text(
    gold_database: tuple[Engine, sessionmaker[Session]],
) -> None:
    _, factory = gold_database
    cross_case_label = _label(
        UID_THREE,
        relevance=2,
        case_name="Gamma Bank v. Delta Ltd.",
        case_number=None,
        paragraph_number=20,
        page_number=12,
    )
    second_query_label = _label(
        UID_TWO,
        paragraph_number=11,
        page_number=None,
    )
    queries = (
        _query(labels=(_label(), cross_case_label)),
        GoldQuery(
            query_id="Q002",
            query="What safeguards apply to appointment procedure?",
            query_type="procedural",
            difficulty="easy",
            notes="Uses distinctive appointment terminology.",
            relevant_paragraphs=(second_query_label,),
        ),
    )

    with factory() as session:
        statistics = validate_gold_queries(queries, session, expected_count=2)
        review = render_review_markdown(queries, session, expected_count=2)

    assert statistics.total_queries == 2
    assert statistics.queries_by_type["legal_principle"] == 1
    assert statistics.queries_by_type["procedural"] == 1
    assert statistics.queries_by_difficulty == {
        "easy": 1,
        "medium": 1,
        "hard": 0,
    }
    assert statistics.total_relevance_labels == 3
    assert statistics.relevance_by_grade == {"1": 0, "2": 1, "3": 2}
    assert statistics.average_relevant_paragraphs_per_query == 1.5
    assert statistics.minimum_relevant_paragraphs_per_query == 1
    assert statistics.maximum_relevant_paragraphs_per_query == 2
    assert statistics.distinct_judgments == 2
    assert statistics.queries_with_multiple_relevant_paragraphs == 1
    assert statistics.queries_with_multiple_relevant_judgments == 1
    assert "## Q001 — Can one party appoint the sole arbitrator?" in review
    assert "paragraph 10, page 5" in review
    assert "A party interested in the outcome" in review
    assert "Paragraph UID: `00000000-0000-5000-8000-000000000001`" in review


def _query_record() -> dict[str, object]:
    return {
        "query_id": "Q001",
        "query": "Can one party appoint the sole arbitrator?",
        "query_type": "legal_principle",
        "difficulty": "medium",
        "notes": "Tests a paraphrased arbitration principle.",
        "relevant_paragraphs": [
            {
                "paragraph_uid": UID_ONE,
                "relevance": 3,
                "case_name": "Alpha Ltd. v. Beta Ltd.",
                "case_number": "Arbitration Appeal No. 1 of 2024",
                "paragraph_number": 10,
                "page_number": 5,
                "reason": "Directly states the governing rule.",
            }
        ],
    }
