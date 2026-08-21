from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

from legal_rag.retrieval import (
    BM25ParagraphRetriever,
    DenseParagraphRetriever,
    HybridParagraphRetriever,
    ParagraphDocument,
    ParagraphSearchResult,
    RetrievalFilters,
    build_qdrant_filter,
)
from legal_rag.vector import QdrantParagraphIndex


def _uid(suffix: int) -> str:
    return f"00000000-0000-5000-8000-{suffix:012d}"


def _document(
    suffix: int,
    *,
    text: str = "arbitration",
    court: str | None,
    judgment_date: date | None,
    case_number: str | None,
) -> ParagraphDocument:
    return ParagraphDocument(
        paragraph_uid=_uid(suffix),
        text=text,
        case_id=suffix,
        title=f"Case {suffix}",
        case_number=case_number,
        court=court,
        judgment_date=judgment_date,
        source_url=f"https://example.test/{suffix}.pdf",
        paragraph_number=suffix,
        page_number=suffix,
    )


@pytest.fixture
def metadata_documents() -> list[ParagraphDocument]:
    return [
        _document(
            1,
            court="Supreme Court of India",
            judgment_date=date(2024, 1, 2),
            case_number="CA 1/2024",
        ),
        _document(
            2,
            court="Delhi High Court",
            judgment_date=date(2024, 2, 3),
            case_number="CS(COMM) 2/2024",
        ),
        _document(
            3,
            court="Supreme Court of India",
            judgment_date=date(2023, 3, 4),
            case_number="CA 3/2023",
        ),
        _document(
            4,
            court="Supreme Court of India",
            judgment_date=date(2024, 4, 5),
            case_number="CA 4/2024",
        ),
    ]


def test_filter_contract_normalizes_nfkc_case_and_whitespace() -> None:
    filters = RetrievalFilters(
        court="  ＳＵＰＲＥＭＥ\tＣＯＵＲＴ\nＯＦ  ＩＮＤＩＡ ",
        year=2024,
        case_number="  ＣＡ  １２／２０２４ ",
    )

    assert filters.court == "supreme court of india"
    assert filters.year == 2024
    assert filters.case_number == "ca 12/2024"
    assert filters.is_active is True
    assert RetrievalFilters().is_active is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"court": ""},
        {"court": " \t\n "},
        {"case_number": "\u3000"},
    ],
)
def test_filter_contract_rejects_empty_string_constraints(
    kwargs: dict[str, str],
) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        RetrievalFilters(**kwargs)


@pytest.mark.parametrize("year", [0, -1, 10_000])
def test_filter_contract_rejects_years_outside_calendar_range(year: int) -> None:
    with pytest.raises(ValueError, match="between 1 and 9999"):
        RetrievalFilters(year=year)


@pytest.mark.parametrize("year", [True, "2024", 2024.0])
def test_filter_contract_rejects_non_integer_years(year: object) -> None:
    with pytest.raises(TypeError, match="year must be an integer"):
        RetrievalFilters(year=year)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("filters", "expected_uids"),
    [
        (
            RetrievalFilters(court="  SUPREME\tCOURT OF INDIA "),
            [_uid(1), _uid(3), _uid(4)],
        ),
        (RetrievalFilters(year=2024), [_uid(1), _uid(2), _uid(4)]),
        (RetrievalFilters(case_number="  cs(comm) 2/2024 "), [_uid(2)]),
        (
            RetrievalFilters(
                court="supreme court of india",
                year=2024,
                case_number=" ca 4/2024 ",
            ),
            [_uid(4)],
        ),
    ],
    ids=["court", "year", "case-number", "combined-and"],
)
def test_bm25_applies_each_filter_and_combines_filters_with_and(
    metadata_documents: list[ParagraphDocument],
    filters: RetrievalFilters,
    expected_uids: list[str],
) -> None:
    retriever = BM25ParagraphRetriever(metadata_documents, b=0)

    results = retriever.search("arbitration", top_k=20, filters=filters)

    assert [result.paragraph_uid for result in results] == expected_uids
    assert all(
        filters.matches(
            court=result.court,
            judgment_date=result.judgment_date,
            case_number=result.case_number,
        )
        for result in results
    )
    assert [result.rank for result in results] == list(range(1, len(results) + 1))


def test_bm25_returns_zero_or_fewer_than_top_k_for_small_eligible_sets(
    metadata_documents: list[ParagraphDocument],
) -> None:
    retriever = BM25ParagraphRetriever(metadata_documents)

    assert retriever.search(
        "arbitration",
        top_k=50,
        filters=RetrievalFilters(case_number="missing case"),
    ) == []
    assert retriever.search(
        "arbitration",
        top_k=50,
        filters=RetrievalFilters(case_number="cs comm 2/2024"),
    ) == []

    one_result = retriever.search(
        "arbitration",
        top_k=50,
        filters=RetrievalFilters(case_number="ca 4/2024"),
    )
    assert [result.paragraph_uid for result in one_result] == [_uid(4)]
    assert one_result[0].rank == 1


def test_bm25_filter_ranks_full_eligible_corpus_before_top_k() -> None:
    retriever = BM25ParagraphRetriever(
        [
            _document(
                10,
                text="arbitration arbitration arbitration arbitration",
                court="Court A",
                judgment_date=date(2024, 1, 1),
                case_number="A/2024",
            ),
            _document(
                11,
                text="arbitration",
                court="Court B",
                judgment_date=date(2024, 1, 1),
                case_number="B/2024",
            ),
        ],
        b=0,
    )

    global_results = retriever.search("arbitration", top_k=2)
    filtered_results = retriever.search(
        "arbitration",
        top_k=1,
        filters=RetrievalFilters(court="court b"),
    )

    assert [result.paragraph_uid for result in global_results] == [_uid(10), _uid(11)]
    assert [result.paragraph_uid for result in filtered_results] == [_uid(11)]
    assert filtered_results[0].score == pytest.approx(global_results[1].score)


def test_bm25_unfiltered_ranking_and_scores_are_unchanged(
    metadata_documents: list[ParagraphDocument],
) -> None:
    retriever = BM25ParagraphRetriever(metadata_documents)

    baseline = retriever.search("arbitration", top_k=10)

    assert retriever.search("arbitration", top_k=10, filters=None) == baseline
    assert retriever.search(
        "arbitration",
        top_k=10,
        filters=RetrievalFilters(),
    ) == baseline


def _payload(
    suffix: int,
    *,
    court: str = "Supreme Court of India",
    judgment_date: str = "2024-01-02",
    case_number: str = "CA 1/2024",
) -> dict[str, object]:
    return {
        "paragraph_uid": _uid(suffix),
        "text": f"Paragraph {suffix}",
        "case_id": suffix,
        "title": f"Case {suffix}",
        "case_number": case_number,
        "court": court,
        "judgment_date": judgment_date,
        "source_url": f"https://example.test/{suffix}.pdf",
        "paragraph_number": suffix,
        "page_number": suffix,
    }


class _CapturingQdrantClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def query_points(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    id=_uid(1),
                    score=0.875,
                    payload=_payload(1),
                )
            ]
        )


class _FakeEmbeddingProvider:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_query(self, query: str) -> list[float]:
        self.queries.append(query)
        return [0.25, 0.75]


def test_dense_builds_and_forwards_native_qdrant_filter() -> None:
    client = _CapturingQdrantClient()
    paragraph_index = QdrantParagraphIndex(
        url="http://unused.test",
        collection_name="paragraphs",
        client=client,
    )
    provider = _FakeEmbeddingProvider()
    retriever = DenseParagraphRetriever(paragraph_index, provider)  # type: ignore[arg-type]
    filters = RetrievalFilters(
        court=" SUPREME COURT OF INDIA ",
        year=2024,
        case_number=" CA 1/2024 ",
    )

    results = retriever.search("arbitration", top_k=7, filters=filters)

    assert provider.queries == ["arbitration"]
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["query"] == [0.25, 0.75]
    assert call["limit"] == 7
    assert call["with_payload"] is True
    native_filter = call["query_filter"]
    assert native_filter is not None
    assert native_filter.model_dump(exclude_none=True) == {  # type: ignore[union-attr]
        "must": [
            {
                "key": "court_filter",
                "match": {"value": "supreme court of india"},
            },
            {"key": "year", "match": {"value": 2024}},
            {
                "key": "case_number_filter",
                "match": {"value": "ca 1/2024"},
            },
        ]
    }
    assert len(results) == 1
    assert results[0].score == 0.875
    assert results[0].court == "Supreme Court of India"
    assert results[0].case_number == "CA 1/2024"


def test_dense_unfiltered_search_has_no_qdrant_filter_and_same_results() -> None:
    client = _CapturingQdrantClient()
    retriever = DenseParagraphRetriever(
        QdrantParagraphIndex(
            url="http://unused.test",
            collection_name="paragraphs",
            client=client,
        ),
        _FakeEmbeddingProvider(),  # type: ignore[arg-type]
    )

    baseline = retriever.search("arbitration", top_k=3)
    explicit_none = retriever.search("arbitration", top_k=3, filters=None)
    empty_contract = retriever.search(
        "arbitration",
        top_k=3,
        filters=RetrievalFilters(),
    )

    assert explicit_none == baseline
    assert empty_contract == baseline
    assert [call["query_filter"] for call in client.calls] == [None, None, None]


def _result(
    suffix: int,
    *,
    score: float,
    court: str = "Supreme Court of India",
    judgment_date: date = date(2024, 1, 2),
    case_number: str = "CA 1/2024",
) -> ParagraphSearchResult:
    return ParagraphSearchResult(
        paragraph_uid=_uid(suffix),
        text=f"Paragraph {suffix}",
        case_id=suffix,
        title=f"Case {suffix}",
        case_number=case_number,
        court=court,
        judgment_date=judgment_date,
        source_url=f"https://example.test/{suffix}.pdf",
        paragraph_number=suffix,
        page_number=suffix,
        score=score,
        rank=1,
    )


class _CapturingRetriever:
    def __init__(self, results: list[ParagraphSearchResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, int, RetrievalFilters | None]] = []

    def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: RetrievalFilters | None = None,
    ) -> list[ParagraphSearchResult]:
        self.calls.append((query, top_k, filters))
        return self.results[:top_k]


def test_hybrid_passes_same_filters_to_both_candidate_retrievers_before_rrf() -> None:
    bm25_shared = _result(20, score=9.0)
    dense_shared = _result(20, score=0.9)
    dense_only = _result(21, score=0.7)
    bm25 = _CapturingRetriever([bm25_shared])
    dense = _CapturingRetriever([dense_shared, dense_only])
    retriever = HybridParagraphRetriever(
        bm25,
        dense,
        bm25_candidate_depth=4,
        dense_candidate_depth=5,
    )
    filters = RetrievalFilters(
        court="supreme court of india",
        year=2024,
        case_number="ca 1/2024",
    )

    results, diagnostics = retriever.search_with_diagnostics(
        "arbitration",
        top_k=10,
        filters=filters,
    )

    assert bm25.calls[0][:2] == ("arbitration", 4)
    assert dense.calls[0][:2] == ("arbitration", 5)
    assert bm25.calls[0][2] is filters
    assert dense.calls[0][2] is filters
    assert [result.paragraph_uid for result in results] == [_uid(20), _uid(21)]
    assert all(
        filters.matches(
            court=result.court,
            judgment_date=result.judgment_date,
            case_number=result.case_number,
        )
        for result in results
    )
    assert results[0].rrf_score == pytest.approx(2 / 61)
    assert results[0].bm25_score == 9.0
    assert results[0].dense_score == 0.9
    assert diagnostics.bm25_candidates == 1
    assert diagnostics.dense_candidates == 2
    assert diagnostics.unique_candidates == 2


def test_hybrid_filtered_zero_matches_returns_empty_diagnostics() -> None:
    bm25 = _CapturingRetriever([])
    dense = _CapturingRetriever([])
    retriever = HybridParagraphRetriever(bm25, dense)
    filters = RetrievalFilters(court="nonexistent court")

    results, diagnostics = retriever.search_with_diagnostics(
        "arbitration",
        top_k=10,
        filters=filters,
    )

    assert results == []
    assert bm25.calls[0][2] is filters
    assert dense.calls[0][2] is filters
    assert diagnostics.bm25_candidates == 0
    assert diagnostics.dense_candidates == 0
    assert diagnostics.unique_candidates == 0


def test_hybrid_unfiltered_ranking_is_unchanged() -> None:
    bm25 = _CapturingRetriever(
        [_result(30, score=8.0), _result(31, score=7.0)]
    )
    dense = _CapturingRetriever(
        [_result(31, score=0.9), _result(32, score=0.8)]
    )
    retriever = HybridParagraphRetriever(bm25, dense)

    baseline = retriever.search("arbitration", top_k=10)
    explicit_none = retriever.search("arbitration", top_k=10, filters=None)
    empty_contract = retriever.search(
        "arbitration",
        top_k=10,
        filters=RetrievalFilters(),
    )

    assert explicit_none == baseline
    assert empty_contract == baseline
    assert [result.paragraph_uid for result in baseline] == [
        _uid(31),
        _uid(30),
        _uid(32),
    ]


def test_empty_filter_contract_builds_no_native_qdrant_filter() -> None:
    assert build_qdrant_filter(None) is None
    assert build_qdrant_filter(RetrievalFilters()) is None
