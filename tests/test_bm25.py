from __future__ import annotations

from dataclasses import fields
from datetime import date
from pathlib import Path

import pytest

from legal_rag.retrieval import (
    BM25ParagraphRetriever,
    ParagraphDocument,
    ParagraphSearchResult,
    semantic_hits_to_results,
    tokenize_legal_text,
)
from legal_rag.database import get_engine, get_session_factory, init_db
from legal_rag.models import Case, Paragraph
from legal_rag.vector import SemanticSearchResult


def _document(
    suffix: int,
    text: str,
    *,
    paragraph_uid: str | None = None,
) -> ParagraphDocument:
    return ParagraphDocument(
        paragraph_uid=paragraph_uid or f"00000000-0000-5000-8000-{suffix:012d}",
        text=text,
        case_id=suffix,
        title=f"Case {suffix}",
        case_number=f"CA {suffix}/2024",
        court="Supreme Court of India",
        judgment_date=date(2024, 1, min(suffix, 28)),
        source_url=f"https://example.test/{suffix}.pdf",
        paragraph_number=suffix,
        page_number=suffix,
    )


def test_legal_tokenization_is_unicode_normalized_and_deterministic() -> None:
    assert tokenize_legal_text("SECTION ７ IBC—CoC's discretion") == [
        "section",
        "7",
        "ibc",
        "coc's",
        "discretion",
    ]


def test_exact_lexical_terminology_outranks_unrelated_text() -> None:
    retriever = BM25ParagraphRetriever(
        [
            _document(1, "The unilateral appointment of an arbitrator was invalid."),
            _document(2, "The limitation appeal was dismissed."),
        ]
    )

    results = retriever.search("unilateral appointment arbitrator", top_k=10)

    assert [result.case_id for result in results] == [1]
    assert results[0].score > 0


def test_bm25_uses_document_frequency_and_saturating_term_frequency() -> None:
    document_frequency = BM25ParagraphRetriever(
        [
            _document(1, "rareterm filler filler"),
            _document(2, "commonterm filler filler"),
            _document(3, "commonterm filler filler"),
            _document(4, "commonterm filler filler"),
        ]
    )
    rare_score = document_frequency.search("rareterm", top_k=1)[0].score
    common_score = document_frequency.search("commonterm", top_k=1)[0].score
    assert rare_score > common_score

    term_frequency = BM25ParagraphRetriever(
        [
            _document(5, "arbitrator filler filler filler"),
            _document(6, "arbitrator arbitrator filler filler"),
        ]
    )
    scores = {
        result.case_id: result.score
        for result in term_frequency.search("arbitrator", top_k=2)
    }
    assert scores[6] > scores[5]
    assert scores[6] < 2 * scores[5]


def test_bm25_and_dense_results_share_one_public_contract() -> None:
    bm25_result = BM25ParagraphRetriever(
        [_document(1, "Section 7 IBC admission discretion")]
    ).search("Section 7 IBC", top_k=1)[0]
    document = _document(2, "Committee of creditors")
    dense_result = semantic_hits_to_results(
        [
            SemanticSearchResult(
                point_id=document.paragraph_uid,
                score=0.75,
                payload={
                    "paragraph_uid": document.paragraph_uid,
                    "text": document.text,
                    "case_id": document.case_id,
                    "title": document.title,
                    "case_number": document.case_number,
                    "court": document.court,
                    "judgment_date": document.judgment_date.isoformat(),
                    "source_url": document.source_url,
                    "paragraph_number": document.paragraph_number,
                    "page_number": document.page_number,
                },
            )
        ]
    )[0]

    assert isinstance(bm25_result, ParagraphSearchResult)
    assert isinstance(dense_result, ParagraphSearchResult)
    assert fields(bm25_result) == fields(dense_result)
    assert dense_result.rank == 1
    assert dense_result.score == 0.75


def test_bm25_validates_queries_and_top_k() -> None:
    retriever = BM25ParagraphRetriever([_document(1, "arbitration")])

    with pytest.raises(ValueError, match="must not be empty"):
        retriever.search("  ", top_k=1)
    with pytest.raises(ValueError, match="lexical token"):
        retriever.search("---", top_k=1)
    with pytest.raises(ValueError, match="top_k"):
        retriever.search("arbitration", top_k=0)


def test_bm25_large_top_k_deduplicates_and_ties_break_by_uid() -> None:
    duplicate_uid = "00000000-0000-5000-8000-000000000003"
    retriever = BM25ParagraphRetriever(
        [
            _document(3, "arbitration", paragraph_uid=duplicate_uid),
            _document(2, "arbitration"),
            _document(1, "arbitration"),
            _document(4, "arbitration", paragraph_uid=duplicate_uid),
        ],
        b=0,
    )

    first = retriever.search("arbitration", top_k=50)
    second = retriever.search("arbitration", top_k=50)

    assert [result.paragraph_uid for result in first] == sorted(
        {result.paragraph_uid for result in first}
    )
    assert first == second
    assert [result.rank for result in first] == [1, 2, 3]


def test_bm25_builds_from_canonical_database_rows(tmp_path: Path) -> None:
    engine = get_engine(f"sqlite+pysqlite:///{(tmp_path / 'bm25.db').as_posix()}")
    init_db(engine)
    session_factory = get_session_factory(engine)
    paragraph_uid = "00000000-0000-5000-8000-000000000009"
    try:
        with session_factory.begin() as session:
            case = Case(
                title="Canonical Commercial Case",
                case_number="CA 9/2024",
                court="Supreme Court of India",
                judgment_date=date(2024, 1, 9),
                source="test",
                source_url="https://example.test/9.pdf",
                raw_text="The insolvency application was admitted.",
                document_hash="a" * 64,
            )
            session.add(case)
            session.flush()
            session.add(
                Paragraph(
                    paragraph_uid=paragraph_uid,
                    case_id=case.id,
                    paragraph_number=9,
                    page_number=3,
                    text="The insolvency application was admitted.",
                    text_hash="b" * 64,
                )
            )

        with session_factory() as session:
            retriever = BM25ParagraphRetriever.from_session(session)
        result = retriever.search("insolvency application", top_k=5)[0]
    finally:
        engine.dispose()

    assert retriever.indexed_paragraphs == 1
    assert result.paragraph_uid == paragraph_uid
    assert result.title == "Canonical Commercial Case"
    assert result.case_number == "CA 9/2024"
    assert result.page_number == 3
