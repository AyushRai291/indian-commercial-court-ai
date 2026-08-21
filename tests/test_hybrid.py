from __future__ import annotations

from datetime import date

import pytest

from legal_rag.retrieval import (
    HybridParagraphRetriever,
    HybridSearchResult,
    ParagraphSearchResult,
    reciprocal_rank_fusion,
)


def _result(
    uid: str,
    *,
    score: float,
    rank: int,
    text: str | None = None,
) -> ParagraphSearchResult:
    suffix = int(uid.rsplit("-", 1)[-1])
    return ParagraphSearchResult(
        paragraph_uid=uid,
        text=text or f"Paragraph {suffix}",
        case_id=suffix,
        title=f"Case {suffix}",
        case_number=f"CA {suffix}/2024",
        court="Supreme Court of India",
        judgment_date=date(2024, 1, min(suffix, 28)),
        source_url=f"https://example.test/{suffix}.pdf",
        paragraph_number=suffix,
        page_number=suffix,
        score=score,
        rank=rank,
    )


def _uid(suffix: int) -> str:
    return f"00000000-0000-5000-8000-{suffix:012d}"


class _FakeRetriever:
    def __init__(self, results: list[ParagraphSearchResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, *, top_k: int) -> list[ParagraphSearchResult]:
        self.calls.append((query, top_k))
        return self.results[:top_k]


def test_rrf_exact_math_order_and_cross_list_boost() -> None:
    a, b, c = _uid(1), _uid(2), _uid(3)
    results = reciprocal_rank_fusion(
        [_result(a, score=100.0, rank=1), _result(b, score=2.0, rank=2)],
        [_result(b, score=0.7, rank=1), _result(c, score=0.9, rank=2)],
        top_k=10,
        rrf_k=60,
    )

    assert [result.paragraph_uid for result in results] == [b, a, c]
    assert results[0].rrf_score == pytest.approx(1 / 62 + 1 / 61)
    assert results[1].rrf_score == pytest.approx(1 / 61)
    assert results[2].rrf_score == pytest.approx(1 / 62)
    assert [result.rank for result in results] == [1, 2, 3]


def test_one_list_candidates_are_retained_and_cross_list_uid_is_deduplicated() -> None:
    a, b, c = _uid(1), _uid(2), _uid(3)
    results = reciprocal_rank_fusion(
        [_result(a, score=4.0, rank=1), _result(b, score=3.0, rank=2)],
        [_result(b, score=0.8, rank=1), _result(c, score=0.7, rank=2)],
        top_k=50,
    )

    assert {result.paragraph_uid for result in results} == {a, b, c}
    assert len(results) == 3
    assert next(result for result in results if result.paragraph_uid == a).dense_rank is None
    assert next(result for result in results if result.paragraph_uid == c).bm25_rank is None


def test_duplicate_within_a_list_contributes_only_once_at_first_position() -> None:
    a = _uid(1)
    results = reciprocal_rank_fusion(
        [
            _result(a, score=9.0, rank=1),
            _result(a, score=8.0, rank=2, text="duplicate"),
        ],
        [],
        top_k=5,
    )

    assert len(results) == 1
    assert results[0].rrf_score == pytest.approx(1 / 61)
    assert results[0].bm25_rank == 1
    assert results[0].bm25_score == 9.0
    assert results[0].text == "Paragraph 1"


def test_ties_are_deterministic_by_paragraph_uid() -> None:
    a, b = _uid(1), _uid(2)
    first = reciprocal_rank_fusion(
        [_result(b, score=10.0, rank=1)],
        [_result(a, score=0.9, rank=1)],
        top_k=5,
    )
    second = reciprocal_rank_fusion(
        [_result(b, score=10.0, rank=99)],
        [_result(a, score=0.9, rank=42)],
        top_k=5,
    )

    assert [result.paragraph_uid for result in first] == [a, b]
    assert [result.paragraph_uid for result in second] == [a, b]


def test_hybrid_contract_preserves_metadata_and_native_provenance() -> None:
    uid = _uid(7)
    bm25_result = _result(uid, score=12.5, rank=1, text="Canonical text")
    dense_result = _result(uid, score=0.8123, rank=1)
    result = reciprocal_rank_fusion(
        [bm25_result],
        [dense_result],
        top_k=1,
    )[0]

    assert isinstance(result, HybridSearchResult)
    assert result.text == "Canonical text"
    assert result.case_id == 7
    assert result.title == "Case 7"
    assert result.case_number == "CA 7/2024"
    assert result.court == "Supreme Court of India"
    assert result.source_url == "https://example.test/7.pdf"
    assert result.paragraph_number == 7
    assert result.page_number == 7
    assert result.bm25_rank == 1
    assert result.dense_rank == 1
    assert result.bm25_score == 12.5
    assert result.dense_score == 0.8123
    assert result.score == result.rrf_score


def test_hybrid_retriever_uses_configured_depths_and_large_top_k() -> None:
    bm25 = _FakeRetriever([_result(_uid(1), score=2.0, rank=1)])
    dense = _FakeRetriever([_result(_uid(2), score=0.8, rank=1)])
    retriever = HybridParagraphRetriever(
        bm25,
        dense,
        bm25_candidate_depth=50,
        dense_candidate_depth=60,
    )

    results, diagnostics = retriever.search_with_diagnostics("arbitration", top_k=100)

    assert len(results) == 2
    assert bm25.calls == [("arbitration", 50)]
    assert dense.calls == [("arbitration", 60)]
    assert diagnostics.bm25_candidates == 1
    assert diagnostics.dense_candidates == 1
    assert diagnostics.unique_candidates == 2
    assert diagnostics.fusion_seconds >= 0
    assert diagnostics.total_seconds >= diagnostics.fusion_seconds


@pytest.mark.parametrize("bm25_count,dense_count", [(0, 0), (1, 0), (0, 1)])
def test_hybrid_handles_empty_candidate_lists(
    bm25_count: int,
    dense_count: int,
) -> None:
    result = _result(_uid(1), score=1.0, rank=1)
    retriever = HybridParagraphRetriever(
        _FakeRetriever([result] * bm25_count),
        _FakeRetriever([result] * dense_count),
    )

    results = retriever.search("insolvency", top_k=10)

    assert len(results) == max(bm25_count, dense_count)


def test_hybrid_validation_rejects_empty_query_and_nonpositive_parameters() -> None:
    retriever = HybridParagraphRetriever(_FakeRetriever([]), _FakeRetriever([]))

    with pytest.raises(ValueError, match="query"):
        retriever.search("  ", top_k=1)
    with pytest.raises(ValueError, match="top_k"):
        retriever.search("arbitration", top_k=0)
    with pytest.raises(ValueError, match="bm25_candidate_depth"):
        retriever.search("arbitration", top_k=1, bm25_candidate_depth=0)
    with pytest.raises(ValueError, match="dense_candidate_depth"):
        retriever.search("arbitration", top_k=1, dense_candidate_depth=-1)
    with pytest.raises(ValueError, match="rrf_k"):
        reciprocal_rank_fusion([], [], top_k=1, rrf_k=0)
