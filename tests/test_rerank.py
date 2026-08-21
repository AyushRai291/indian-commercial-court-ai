from __future__ import annotations

import sys
from collections.abc import Sequence
from datetime import date
from types import ModuleType

import pytest

from legal_rag.retrieval import (
    CrossEncoderBatchResult,
    CrossEncoderReranker,
    HybridSearchDiagnostics,
    HybridSearchResult,
    RerankedSearchResult,
    RetrievalFilters,
    SentenceTransformerCrossEncoderScorer,
)


def _uid(suffix: int) -> str:
    return f"00000000-0000-5000-8000-{suffix:012d}"


def _candidate(
    suffix: int,
    *,
    hybrid_rank: int,
    rrf_score: float | None = None,
    text: str | None = None,
    bm25_rank: int | None = None,
    dense_rank: int | None = None,
    bm25_score: float | None = None,
    dense_score: float | None = None,
) -> HybridSearchResult:
    return HybridSearchResult(
        paragraph_uid=_uid(suffix),
        text=text if text is not None else f"Paragraph {suffix}",
        case_id=1000 + suffix,
        title=f"Case {suffix}",
        case_number=f"CA {suffix}/2024",
        court="Supreme Court of India",
        judgment_date=date(2024, 1, min(suffix, 28)),
        source_url=f"https://example.test/{suffix}.pdf",
        paragraph_number=200 + suffix,
        page_number=300 + suffix,
        score=rrf_score if rrf_score is not None else 1.0 / (60 + hybrid_rank),
        rank=hybrid_rank,
        bm25_rank=bm25_rank,
        dense_rank=dense_rank,
        bm25_score=bm25_score,
        dense_score=dense_score,
    )


class _FakeHybridRetriever:
    def __init__(self, results: Sequence[HybridSearchResult]) -> None:
        self.results = list(results)
        self.calls: list[dict[str, object]] = []

    def search_with_diagnostics(
        self,
        query: str,
        *,
        top_k: int,
        bm25_candidate_depth: int | None = None,
        dense_candidate_depth: int | None = None,
        rrf_k: int | None = None,
        filters: RetrievalFilters | None = None,
    ) -> tuple[list[HybridSearchResult], HybridSearchDiagnostics]:
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "bm25_candidate_depth": bm25_candidate_depth,
                "dense_candidate_depth": dense_candidate_depth,
                "rrf_k": rrf_k,
                "filters": filters,
            }
        )
        results = self.results[:top_k]
        return results, HybridSearchDiagnostics(
            bm25_candidates=len(results),
            dense_candidates=len(results),
            unique_candidates=len({result.paragraph_uid for result in results}),
            fusion_seconds=0.0,
            total_seconds=0.0,
        )


class _FakeScorer:
    def __init__(self, scores: Sequence[float]) -> None:
        self.scores = tuple(scores)
        self.calls: list[tuple[list[tuple[str, str]], int]] = []

    def score_pairs(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        batch_size: int,
    ) -> CrossEncoderBatchResult:
        values = list(pairs)
        self.calls.append((values, batch_size))
        return CrossEncoderBatchResult(
            scores=self.scores[: len(values)],
            model_load_seconds=0.0,
            inference_seconds=0.0,
        )


def _reranker(
    hybrid: _FakeHybridRetriever,
    scorer: object,
    *,
    candidate_k: int = 50,
    batch_size: int = 16,
) -> CrossEncoderReranker:
    return CrossEncoderReranker(
        hybrid,  # type: ignore[arg-type]
        scorer,  # type: ignore[arg-type]
        candidate_k=candidate_k,
        batch_size=batch_size,
    )


def test_reranker_scores_only_hybrid_candidates_in_one_batch_and_orders_by_score(
) -> None:
    candidates = [
        _candidate(1, hybrid_rank=1, text="  Exact first\nparagraph  "),
        _candidate(2, hybrid_rank=2, text="Second\tparagraph"),
        _candidate(3, hybrid_rank=3, text="Third paragraph"),
        _candidate(4, hybrid_rank=4, text="Must never be scored"),
    ]
    hybrid = _FakeHybridRetriever(candidates)
    scorer = _FakeScorer([-2.0, 3.5, 0.25])
    reranker = _reranker(hybrid, scorer, batch_size=7)

    results = reranker.search("appointment query", top_k=3, candidate_k=3)

    assert hybrid.calls[0]["top_k"] == 3
    assert scorer.calls == [
        (
            [
                ("appointment query", "  Exact first\nparagraph  "),
                ("appointment query", "Second\tparagraph"),
                ("appointment query", "Third paragraph"),
            ],
            7,
        )
    ]
    assert [result.paragraph_uid for result in results] == [
        _uid(2),
        _uid(3),
        _uid(1),
    ]
    assert [result.cross_encoder_score for result in results] == [3.5, 0.25, -2.0]
    assert [result.reranked_rank for result in results] == [1, 2, 3]


def test_reranker_forwards_depths_rrf_and_same_filter_object() -> None:
    candidate = _candidate(1, hybrid_rank=1)
    hybrid = _FakeHybridRetriever([candidate])
    scorer = _FakeScorer([0.8])
    filters = RetrievalFilters(
        court="Supreme Court of India",
        year=2024,
        case_number="CA 1/2024",
    )
    reranker = _reranker(hybrid, scorer)

    reranker.search(
        "filtered query",
        top_k=1,
        candidate_k=7,
        bm25_candidate_depth=11,
        dense_candidate_depth=13,
        rrf_k=42,
        filters=filters,
    )

    assert hybrid.calls == [
        {
            "query": "filtered query",
            "top_k": 7,
            "bm25_candidate_depth": 11,
            "dense_candidate_depth": 13,
            "rrf_k": 42,
            "filters": filters,
        }
    ]
    assert hybrid.calls[0]["filters"] is filters
    assert scorer.calls[0][0] == [("filtered query", candidate.text)]


def test_reranked_result_preserves_all_metadata_and_hybrid_provenance() -> None:
    candidate = _candidate(
        7,
        hybrid_rank=4,
        rrf_score=0.0312345,
        text="Canonical paragraph text",
        bm25_rank=2,
        dense_rank=9,
        bm25_score=12.75,
        dense_score=0.8125,
    )
    result = _reranker(
        _FakeHybridRetriever([candidate]),
        _FakeScorer([-1.375]),
    ).search("query", top_k=1)[0]

    assert isinstance(result, RerankedSearchResult)
    for field_name in (
        "paragraph_uid",
        "text",
        "case_id",
        "title",
        "case_number",
        "court",
        "judgment_date",
        "source_url",
        "paragraph_number",
        "page_number",
        "score",
        "rank",
        "bm25_rank",
        "dense_rank",
        "bm25_score",
        "dense_score",
    ):
        assert getattr(result, field_name) == getattr(candidate, field_name)
    assert result.rrf_score == candidate.rrf_score
    assert result.hybrid_rank == candidate.rank
    assert result.cross_encoder_score == -1.375
    assert result.reranked_rank == 1


def test_equal_scores_use_hybrid_rank_then_paragraph_uid() -> None:
    a, b, c = (
        _candidate(1, hybrid_rank=2),
        _candidate(2, hybrid_rank=2),
        _candidate(3, hybrid_rank=1),
    )
    hybrid = _FakeHybridRetriever([b, c, a])
    results = _reranker(hybrid, _FakeScorer([0.5, 0.5, 0.5])).search(
        "query",
        top_k=3,
    )

    assert [result.paragraph_uid for result in results] == [
        c.paragraph_uid,
        a.paragraph_uid,
        b.paragraph_uid,
    ]
    assert [result.hybrid_rank for result in results] == [1, 2, 2]
    assert [result.reranked_rank for result in results] == [1, 2, 3]


@pytest.mark.parametrize(
    ("available", "candidate_k", "top_k", "expected_scored", "expected_returned"),
    [(5, 3, 2, 3, 2), (2, 50, 10, 2, 2)],
)
def test_candidate_and_output_limits(
    available: int,
    candidate_k: int,
    top_k: int,
    expected_scored: int,
    expected_returned: int,
) -> None:
    hybrid = _FakeHybridRetriever(
        [_candidate(index, hybrid_rank=index) for index in range(1, available + 1)]
    )
    scorer = _FakeScorer([float(index) for index in range(expected_scored)])
    results = _reranker(hybrid, scorer).search(
        "query",
        candidate_k=candidate_k,
        top_k=top_k,
    )

    assert hybrid.calls[0]["top_k"] == candidate_k
    assert len(scorer.calls[0][0]) == expected_scored
    assert len(results) == expected_returned
    assert [result.reranked_rank for result in results] == list(
        range(1, expected_returned + 1)
    )


def test_reranker_uses_fifty_to_ten_defaults() -> None:
    candidates = [_candidate(index, hybrid_rank=index) for index in range(1, 61)]
    hybrid = _FakeHybridRetriever(candidates)
    scorer = _FakeScorer([float(index) for index in range(50)])

    results = _reranker(hybrid, scorer).search("query")

    assert hybrid.calls[0]["top_k"] == 50
    assert len(scorer.calls[0][0]) == 50
    assert len(results) == 10
    assert [result.reranked_rank for result in results] == list(range(1, 11))


def test_empty_hybrid_results_skip_scoring_and_model_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructor_calls: list[tuple[object, ...]] = []

    class UnexpectedCrossEncoder:
        def __init__(self, *args: object, **kwargs: object) -> None:
            constructor_calls.append((*args, kwargs))

    fake_module = ModuleType("sentence_transformers")
    fake_module.CrossEncoder = UnexpectedCrossEncoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    scorer = SentenceTransformerCrossEncoderScorer("fake/model")
    hybrid = _FakeHybridRetriever([])

    results = _reranker(hybrid, scorer).search("query")

    assert results == []
    assert constructor_calls == []
    assert scorer.is_loaded is False


def test_duplicate_uids_are_scored_once_and_first_occurrence_wins() -> None:
    first = _candidate(1, hybrid_rank=1, text="First canonical occurrence")
    duplicate = _candidate(1, hybrid_rank=9, text="Duplicate must be ignored")
    other = _candidate(2, hybrid_rank=2)
    scorer = _FakeScorer([0.1, 0.9])

    results = _reranker(
        _FakeHybridRetriever([first, duplicate, other]),
        scorer,
    ).search("query", top_k=10)

    assert scorer.calls[0][0] == [
        ("query", first.text),
        ("query", other.text),
    ]
    assert [result.paragraph_uid for result in results] == [
        other.paragraph_uid,
        first.paragraph_uid,
    ]
    assert len({result.paragraph_uid for result in results}) == len(results)
    assert next(
        result for result in results if result.paragraph_uid == first.paragraph_uid
    ).text == first.text


def test_cross_encoder_model_is_lazy_batched_and_reused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructor_calls: list[tuple[str, str | None]] = []
    prediction_calls: list[tuple[list[tuple[str, str]], dict[str, object]]] = []

    class FakeCrossEncoder:
        def __init__(self, model_name: str, *, device: str | None = None) -> None:
            constructor_calls.append((model_name, device))

        def predict(
            self,
            pairs: Sequence[tuple[str, str]],
            **kwargs: object,
        ) -> list[float]:
            values = list(pairs)
            prediction_calls.append((values, kwargs))
            return [float(index) for index in range(len(values))]

    fake_module = ModuleType("sentence_transformers")
    fake_module.CrossEncoder = FakeCrossEncoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    scorer = SentenceTransformerCrossEncoderScorer("fake/model", device="cpu")
    hybrid = _FakeHybridRetriever(
        [_candidate(1, hybrid_rank=1), _candidate(2, hybrid_rank=2)]
    )
    reranker = _reranker(hybrid, scorer, batch_size=4)

    assert constructor_calls == []
    assert scorer.is_loaded is False

    reranker.search("first query", top_k=2)
    reranker.search("second query", top_k=2)

    assert constructor_calls == [("fake/model", "cpu")]
    assert scorer.is_loaded is True
    assert [call[0] for call in prediction_calls] == [
        [
            ("first query", "Paragraph 1"),
            ("first query", "Paragraph 2"),
        ],
        [
            ("second query", "Paragraph 1"),
            ("second query", "Paragraph 2"),
        ],
    ]
    assert [call[1] for call in prediction_calls] == [
        {
            "batch_size": 4,
            "show_progress_bar": False,
            "convert_to_numpy": True,
            "apply_softmax": False,
        },
        {
            "batch_size": 4,
            "show_progress_bar": False,
            "convert_to_numpy": True,
            "apply_softmax": False,
        },
    ]


def test_cross_encoder_load_failure_names_model_and_chains_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_error = OSError("offline")

    def fail_to_load(*args: object, **kwargs: object) -> None:
        raise load_error

    fake_module = ModuleType("sentence_transformers")
    fake_module.CrossEncoder = fail_to_load  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    scorer = SentenceTransformerCrossEncoderScorer("fake/unavailable-model")

    with pytest.raises(
        RuntimeError,
        match="Unable to load cross-encoder model 'fake/unavailable-model'",
    ) as exc_info:
        scorer.score_pairs([("query", "paragraph")], batch_size=2)

    assert exc_info.value.__cause__ is load_error
