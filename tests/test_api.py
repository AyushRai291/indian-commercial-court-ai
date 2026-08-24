from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pytest
from fastapi.testclient import TestClient

from legal_rag.api.app import SERVICE_UNAVAILABLE_DETAIL, create_app
from legal_rag.api.schemas import SearchResponse
from legal_rag.api.service import (
    BM25_CANDIDATE_DEPTH,
    DENSE_CANDIDATE_DEPTH,
    MODEL_WARMUP_QUERY,
    RERANKER_CANDIDATE_DEPTH,
    RRF_K,
    ModelWarmupResult,
    SearchService,
)
from legal_rag.retrieval import (
    HybridSearchResult,
    ParagraphSearchResult,
    RerankedSearchResult,
    RetrievalFilters,
)


def _uid(suffix: int) -> str:
    return f"00000000-0000-5000-8000-{suffix:012d}"


def _basic_result(
    suffix: int = 1,
    *,
    score: float = 12.5,
    rank: int = 1,
) -> ParagraphSearchResult:
    return ParagraphSearchResult(
        paragraph_uid=_uid(suffix),
        text="The ineligible arbitrator could not nominate another arbitrator.",
        case_id=1000 + suffix,
        title="TRF Limited v. Energo Engineering Projects Limited",
        case_number="Civil Appeal No. 5306 of 2017",
        court="Supreme Court of India",
        judgment_date=date(2017, 7, 3),
        source_url=(
            "https://main.sci.gov.in/supremecourt/2016/12345/"
            "12345_2016_Judgement_03-Jul-2017.pdf?download=1"
        ),
        paragraph_number=50 + suffix,
        page_number=20 + suffix,
        score=score,
        rank=rank,
    )


def _hybrid_result(suffix: int = 1) -> HybridSearchResult:
    base = _basic_result(suffix, score=0.03125, rank=1)
    return HybridSearchResult(
        paragraph_uid=base.paragraph_uid,
        text=base.text,
        case_id=base.case_id,
        title=base.title,
        case_number=base.case_number,
        court=base.court,
        judgment_date=base.judgment_date,
        source_url=base.source_url,
        paragraph_number=base.paragraph_number,
        page_number=base.page_number,
        score=base.score,
        rank=base.rank,
        bm25_rank=2,
        dense_rank=9,
        bm25_score=12.75,
        dense_score=0.8125,
    )


def _reranked_result(suffix: int = 1) -> RerankedSearchResult:
    hybrid = _hybrid_result(suffix)
    return RerankedSearchResult(
        paragraph_uid=hybrid.paragraph_uid,
        text=hybrid.text,
        case_id=hybrid.case_id,
        title=hybrid.title,
        case_number=hybrid.case_number,
        court=hybrid.court,
        judgment_date=hybrid.judgment_date,
        source_url=hybrid.source_url,
        paragraph_number=hybrid.paragraph_number,
        page_number=hybrid.page_number,
        score=hybrid.score,
        rank=4,
        bm25_rank=hybrid.bm25_rank,
        dense_rank=hybrid.dense_rank,
        bm25_score=hybrid.bm25_score,
        dense_score=hybrid.dense_score,
        cross_encoder_score=4.875,
        reranked_rank=1,
    )


@dataclass
class _CapturingBasicRetriever:
    results: list[ParagraphSearchResult] = field(default_factory=list)
    error: Exception | None = None
    calls: list[dict[str, object]] = field(default_factory=list)

    def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: RetrievalFilters | None = None,
    ) -> list[ParagraphSearchResult]:
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "filters": filters,
            }
        )
        if self.error is not None:
            raise self.error
        return self.results[:top_k]


@dataclass
class _CapturingHybridRetriever:
    results: list[HybridSearchResult] = field(default_factory=list)
    error: Exception | None = None
    calls: list[dict[str, object]] = field(default_factory=list)

    def search(
        self,
        query: str,
        *,
        top_k: int,
        bm25_candidate_depth: int | None = None,
        dense_candidate_depth: int | None = None,
        rrf_k: int | None = None,
        filters: RetrievalFilters | None = None,
    ) -> list[HybridSearchResult]:
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
        if self.error is not None:
            raise self.error
        return self.results[:top_k]


@dataclass
class _CapturingReranker:
    results: list[RerankedSearchResult] = field(default_factory=list)
    error: Exception | None = None
    calls: list[dict[str, object]] = field(default_factory=list)

    def search(
        self,
        query: str,
        *,
        top_k: int,
        candidate_k: int | None = None,
        bm25_candidate_depth: int | None = None,
        dense_candidate_depth: int | None = None,
        rrf_k: int | None = None,
        filters: RetrievalFilters | None = None,
    ) -> list[RerankedSearchResult]:
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "candidate_k": candidate_k,
                "bm25_candidate_depth": bm25_candidate_depth,
                "dense_candidate_depth": dense_candidate_depth,
                "rrf_k": rrf_k,
                "filters": filters,
            }
        )
        if self.error is not None:
            raise self.error
        return self.results[:top_k]


@dataclass
class _ApiHarness:
    client: TestClient
    bm25: _CapturingBasicRetriever
    dense: _CapturingBasicRetriever
    hybrid: _CapturingHybridRetriever
    reranker: _CapturingReranker

    @property
    def all_calls(self) -> list[dict[str, object]]:
        return [
            *self.bm25.calls,
            *self.dense.calls,
            *self.hybrid.calls,
            *self.reranker.calls,
        ]


@pytest.fixture
def api_harness() -> Any:
    bm25 = _CapturingBasicRetriever()
    dense = _CapturingBasicRetriever()
    hybrid = _CapturingHybridRetriever()
    reranker = _CapturingReranker()
    service = SearchService(
        bm25_retriever=bm25,
        dense_retriever=dense,
        hybrid_retriever=hybrid,
        reranker=reranker,
    )
    app = create_app(search_service=service)

    with TestClient(app) as client:
        yield _ApiHarness(
            client=client,
            bm25=bm25,
            dense=dense,
            hybrid=hybrid,
            reranker=reranker,
        )


def test_health_returns_process_liveness_without_running_search(
    api_harness: _ApiHarness,
) -> None:
    response = api_harness.client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert api_harness.all_calls == []


def test_model_warmup_loads_local_models_without_search_or_storage_access() -> None:
    bm25 = _CapturingBasicRetriever()
    dense = _CapturingBasicRetriever()
    hybrid = _CapturingHybridRetriever()
    reranker = _CapturingReranker()
    embedding_calls: list[str] = []
    reranker_calls: list[tuple[list[tuple[str, str]], int]] = []

    class _EmbeddingProvider:
        def embed_query(self, query: str) -> list[float]:
            embedding_calls.append(query)
            return [0.0, 1.0]

    class _Scorer:
        def score_pairs(
            self,
            pairs: list[tuple[str, str]],
            *,
            batch_size: int,
        ) -> object:
            reranker_calls.append((pairs, batch_size))
            return object()

    dense.embedding_provider = _EmbeddingProvider()
    reranker.scorer = _Scorer()
    service = SearchService(bm25, dense, hybrid, reranker)

    result = service.warmup()

    assert isinstance(result, ModelWarmupResult)
    assert embedding_calls == [MODEL_WARMUP_QUERY]
    assert reranker_calls == [
        ([(MODEL_WARMUP_QUERY, MODEL_WARMUP_QUERY)], 1)
    ]
    assert result.dense_ms >= 0
    assert result.reranker_ms >= 0
    assert result.total_ms >= result.dense_ms
    assert bm25.calls == []
    assert dense.calls == []
    assert hybrid.calls == []
    assert reranker.calls == []


@pytest.mark.parametrize(
    "payload",
    [
        {"query": ""},
        {"query": " \t\n "},
        {"query": "---"},
        {"query": 123},
        {"query": "arbitration", "retrieval_mode": "semantic"},
        {"query": "arbitration", "top_k": 0},
        {"query": "arbitration", "top_k": -1},
        {"query": "arbitration", "top_k": 51},
        {"query": "arbitration", "top_k": "10"},
        {"query": "arbitration", "top_k": True},
    ],
    ids=[
        "empty-query",
        "whitespace-query",
        "query-without-lexical-token",
        "non-string-query",
        "invalid-mode",
        "zero-top-k",
        "negative-top-k",
        "top-k-over-limit",
        "string-top-k",
        "boolean-top-k",
    ],
)
def test_search_rejects_invalid_core_request_values(
    api_harness: _ApiHarness,
    payload: dict[str, object],
) -> None:
    response = api_harness.client.post("/search", json=payload)

    assert response.status_code == 422
    assert api_harness.all_calls == []


@pytest.mark.parametrize(
    "filters",
    [
        {"year": 0},
        {"year": 10_000},
        {"year": "2024"},
        {"year": 2024.0},
        {"year": True},
        {"court": " \t\n "},
        {"court": 123},
        {"case_number": "\u3000"},
        {"case_number": False},
        {"unknown_filter": "value"},
    ],
    ids=[
        "year-too-low",
        "year-too-high",
        "string-year",
        "float-year",
        "boolean-year",
        "blank-court",
        "non-string-court",
        "blank-case-number",
        "non-string-case-number",
        "unknown-filter",
    ],
)
def test_search_rejects_malformed_metadata_filters(
    api_harness: _ApiHarness,
    filters: dict[str, object],
) -> None:
    response = api_harness.client.post(
        "/search",
        json={"query": "arbitration", "filters": filters},
    )

    assert response.status_code == 422
    assert api_harness.all_calls == []


def test_search_rejects_unknown_request_fields(api_harness: _ApiHarness) -> None:
    response = api_harness.client.post(
        "/search",
        json={"query": "arbitration", "generate_answer": True},
    )

    assert response.status_code == 422
    assert api_harness.all_calls == []


def test_bm25_routes_and_maps_native_score_and_rank(
    api_harness: _ApiHarness,
) -> None:
    api_harness.bm25.results = [_basic_result(score=12.5, rank=1)]

    response = api_harness.client.post(
        "/search",
        json={
            "query": "  unilateral appointment of arbitrator  ",
            "top_k": 7,
            "retrieval_mode": "bm25",
        },
    )

    assert response.status_code == 200
    assert api_harness.bm25.calls == [
        {"query": "unilateral appointment of arbitrator", "top_k": 7, "filters": None}
    ]
    assert api_harness.dense.calls == []
    assert api_harness.hybrid.calls == []
    assert api_harness.reranker.calls == []

    body = response.json()
    assert body["query"] == "unilateral appointment of arbitrator"
    assert body["retrieval_mode"] == "bm25"
    assert body["top_k"] == 7
    assert body["filters"] == {
        "court": None,
        "year": None,
        "case_number": None,
    }
    assert body["result_count"] == 1
    assert body["latency_ms"] >= 0
    assert body["results"][0]["bm25_rank"] == 1
    assert body["results"][0]["bm25_score"] == 12.5
    assert body["results"][0]["dense_rank"] is None
    assert body["results"][0]["dense_score"] is None
    assert body["results"][0]["rrf_score"] is None
    assert body["results"][0]["hybrid_rank"] is None
    assert body["results"][0]["cross_encoder_score"] is None
    assert body["results"][0]["final_rank"] == 1


def test_dense_routes_and_maps_native_score_and_rank(
    api_harness: _ApiHarness,
) -> None:
    api_harness.dense.results = [_basic_result(score=0.8125, rank=1)]

    response = api_harness.client.post(
        "/search",
        json={
            "query": "commercial wisdom",
            "top_k": 5,
            "retrieval_mode": "dense",
        },
    )

    assert response.status_code == 200
    assert api_harness.dense.calls == [
        {"query": "commercial wisdom", "top_k": 5, "filters": None}
    ]
    assert api_harness.bm25.calls == []
    assert api_harness.hybrid.calls == []
    assert api_harness.reranker.calls == []

    result = response.json()["results"][0]
    assert result["bm25_rank"] is None
    assert result["bm25_score"] is None
    assert result["dense_rank"] == 1
    assert result["dense_score"] == 0.8125
    assert result["rrf_score"] is None
    assert result["hybrid_rank"] is None
    assert result["cross_encoder_score"] is None
    assert result["final_rank"] == 1


def test_hybrid_routes_with_tuned_depths_and_preserves_fusion_provenance(
    api_harness: _ApiHarness,
) -> None:
    api_harness.hybrid.results = [_hybrid_result()]

    response = api_harness.client.post(
        "/search",
        json={
            "query": "commercial wisdom of committee of creditors",
            "top_k": 9,
            "retrieval_mode": "hybrid",
        },
    )

    assert response.status_code == 200
    assert api_harness.hybrid.calls == [
        {
            "query": "commercial wisdom of committee of creditors",
            "top_k": 9,
            "bm25_candidate_depth": BM25_CANDIDATE_DEPTH,
            "dense_candidate_depth": DENSE_CANDIDATE_DEPTH,
            "rrf_k": RRF_K,
            "filters": None,
        }
    ]
    assert BM25_CANDIDATE_DEPTH == 50
    assert DENSE_CANDIDATE_DEPTH == 50
    assert RRF_K == 10
    assert api_harness.bm25.calls == []
    assert api_harness.dense.calls == []
    assert api_harness.reranker.calls == []

    result = response.json()["results"][0]
    assert result["bm25_rank"] == 2
    assert result["bm25_score"] == 12.75
    assert result["dense_rank"] == 9
    assert result["dense_score"] == 0.8125
    assert result["rrf_score"] == 0.03125
    assert result["hybrid_rank"] == 1
    assert result["cross_encoder_score"] is None
    assert result["final_rank"] == 1


def test_default_reranked_mode_routes_with_all_tuned_defaults(
    api_harness: _ApiHarness,
) -> None:
    api_harness.reranker.results = [_reranked_result()]

    response = api_harness.client.post(
        "/search",
        json={"query": "commercial wisdom of committee of creditors"},
    )

    assert response.status_code == 200
    assert api_harness.reranker.calls == [
        {
            "query": "commercial wisdom of committee of creditors",
            "top_k": 10,
            "candidate_k": RERANKER_CANDIDATE_DEPTH,
            "bm25_candidate_depth": BM25_CANDIDATE_DEPTH,
            "dense_candidate_depth": DENSE_CANDIDATE_DEPTH,
            "rrf_k": RRF_K,
            "filters": None,
        }
    ]
    assert RERANKER_CANDIDATE_DEPTH == 30
    assert api_harness.bm25.calls == []
    assert api_harness.dense.calls == []
    assert api_harness.hybrid.calls == []

    body = response.json()
    assert body["retrieval_mode"] == "reranked"
    assert body["top_k"] == 10
    result = body["results"][0]
    assert result["bm25_rank"] == 2
    assert result["bm25_score"] == 12.75
    assert result["dense_rank"] == 9
    assert result["dense_score"] == 0.8125
    assert result["rrf_score"] == 0.03125
    assert result["hybrid_rank"] == 4
    assert result["cross_encoder_score"] == 4.875
    assert result["final_rank"] == 1


def test_metadata_filters_are_forwarded_together_through_shared_contract(
    api_harness: _ApiHarness,
) -> None:
    api_harness.hybrid.results = []
    request_filters = {
        "court": "  SUPREME\tCOURT OF INDIA ",
        "year": 2019,
        "case_number": " Arbitration Application No. 32 of 2019 ",
    }

    response = api_harness.client.post(
        "/search",
        json={
            "query": "arbitration",
            "retrieval_mode": "hybrid",
            "filters": request_filters,
        },
    )

    assert response.status_code == 200
    captured = api_harness.hybrid.calls[0]["filters"]
    assert captured == RetrievalFilters(
        court="supreme court of india",
        year=2019,
        case_number="arbitration application no. 32 of 2019",
    )
    assert isinstance(captured, RetrievalFilters)
    assert response.json()["filters"] == request_filters


def test_empty_results_return_successful_zero_count(
    api_harness: _ApiHarness,
) -> None:
    api_harness.bm25.results = []

    response = api_harness.client.post(
        "/search",
        json={
            "query": "no matching terminology",
            "top_k": 10,
            "retrieval_mode": "bm25",
            "filters": {"court": "Court That Does Not Exist"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_count"] == 0
    assert body["results"] == []
    assert body["latency_ms"] >= 0


def test_retrieval_failure_returns_generic_service_error_without_internal_detail(
    api_harness: _ApiHarness,
) -> None:
    private_message = "database password leaked in raw traceback"
    api_harness.dense.error = RuntimeError(private_message)

    response = api_harness.client.post(
        "/search",
        json={"query": "arbitration", "retrieval_mode": "dense"},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": SERVICE_UNAVAILABLE_DETAIL}
    assert private_message not in response.text
    assert "Traceback" not in response.text


def test_failed_service_initialization_keeps_health_live_and_search_unavailable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fail_to_initialize() -> Any:
        raise ConnectionError("postgresql unavailable with private connection data")

    app = create_app(service_factory=fail_to_initialize)
    with caplog.at_level("ERROR"), TestClient(app) as client:
        health_response = client.get("/health")
        search_response = client.post("/search", json={"query": "arbitration"})

    assert health_response.status_code == 200
    assert health_response.json() == {"status": "ok"}
    assert search_response.status_code == 503
    assert search_response.json() == {"detail": SERVICE_UNAVAILABLE_DETAIL}
    assert "private connection data" not in search_response.text


def test_owned_service_is_warmed_before_requests_and_closed_after_shutdown() -> None:
    class _OwnedService:
        warmed = False
        closed = False

        def warmup(self) -> ModelWarmupResult:
            self.warmed = True
            return ModelWarmupResult(dense_ms=2.0, reranker_ms=3.0, total_ms=5.0)

        def search(self, request: Any) -> Any:
            assert self.warmed is True
            return SearchResponse(
                query=request.query,
                retrieval_mode=request.retrieval_mode,
                top_k=request.top_k,
                filters=request.filters,
                result_count=0,
                latency_ms=0.0,
                results=[],
            )

        def close(self) -> None:
            self.closed = True

    service = _OwnedService()
    application = create_app(service_factory=lambda: service)  # type: ignore[arg-type]

    with TestClient(application) as client:
        response = client.post("/search", json={"query": "arbitration"})
        assert application.state.search_service_ready is True
        assert application.state.model_warmup.total_ms == 5.0
        assert application.state.search_startup_ms >= 0

    assert response.status_code == 200
    assert service.closed is True


def test_warmup_failure_keeps_dependencies_unavailable_and_closes_service() -> None:
    private_detail = "private local model cache path"

    class _FailingWarmupService:
        closed = False

        def warmup(self) -> ModelWarmupResult:
            raise RuntimeError(private_detail)

        def search(self, *_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("failed warmup service must not be request-visible")

        def close(self) -> None:
            self.closed = True

    service = _FailingWarmupService()
    application = create_app(service_factory=lambda: service)  # type: ignore[arg-type]

    with TestClient(application) as client:
        health_response = client.get("/health")
        search_response = client.post("/search", json={"query": "arbitration"})

    assert health_response.status_code == 200
    assert search_response.status_code == 503
    assert search_response.json() == {"detail": SERVICE_UNAVAILABLE_DETAIL}
    assert private_detail not in search_response.text
    assert application.state.search_service_ready is False
    assert service.closed is True


def test_response_preserves_exact_canonical_metadata_and_source_provenance(
    api_harness: _ApiHarness,
) -> None:
    expected = _reranked_result(7)
    api_harness.reranker.results = [expected]

    response = api_harness.client.post(
        "/search",
        json={"query": "ineligible arbitrator", "retrieval_mode": "reranked"},
    )

    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["paragraph_uid"] == expected.paragraph_uid
    assert result["text"] == expected.text
    assert result["case_id"] == expected.case_id
    assert result["title"] == expected.title
    assert result["case_number"] == expected.case_number
    assert result["court"] == expected.court
    assert result["judgment_date"] == expected.judgment_date.isoformat()
    assert result["source_url"] == expected.source_url
    assert result["paragraph_number"] == expected.paragraph_number
    assert result["page_number"] == expected.page_number
