from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from legal_rag.api.app import (
    GENERATION_UNAVAILABLE_DETAIL,
    LOCAL_FRONTEND_ORIGINS,
    RESEARCH_UNAVAILABLE_DETAIL,
    create_app,
)
from legal_rag.api.schemas import (
    AnswerEvidence,
    AnswerRequest,
    AnswerResponse,
    ResearchLatency,
    ResearchRequest,
    ResearchResponse,
    ResearchVerificationState,
    RetrievalMode,
    SearchFilters,
    SearchResponse,
    VerificationClaim,
    VerificationSummary,
    VerifyRequest,
    VerifyResponse,
)
from legal_rag.generation.errors import ProviderUnavailableError
from legal_rag.generation.service import AnswerService, NO_EVIDENCE_ANSWER
from legal_rag.research.service import (
    RESEARCH_VERIFICATION_UNAVAILABLE_ERROR,
    ResearchService,
)
from legal_rag.verification.errors import (
    VerificationProviderUnavailableError,
)
from legal_rag.verification.models import VerificationStatus


def _uid(suffix: int) -> str:
    return f"00000000-0000-5000-8000-{suffix:012d}"


def _evidence(suffix: int = 1) -> AnswerEvidence:
    return AnswerEvidence(
        evidence_id=f"E{suffix}",
        paragraph_uid=_uid(suffix),
        text="An ineligible arbitrator cannot nominate another arbitrator.",
        case_id=1000 + suffix,
        case_name="TRF Limited v. Energo Engineering Projects Limited",
        case_number="Civil Appeal No. 5306 of 2017",
        court="Supreme Court of India",
        judgment_date=date(2017, 7, 3),
        source_url=f"https://example.test/judgment-{suffix}.pdf",
        paragraph_number=50 + suffix,
        page_number=20 + suffix,
        bm25_rank=2,
        bm25_score=12.75,
        dense_rank=9,
        dense_score=0.8125,
        rrf_score=0.03125,
        hybrid_rank=4,
        cross_encoder_score=4.875,
        reranked_rank=suffix,
    )


def _answer_response(*, evidence: list[AnswerEvidence] | None = None) -> AnswerResponse:
    supplied_evidence = [_evidence()] if evidence is None else evidence
    return AnswerResponse(
        query="ineligible arbitrator",
        answer=(
            "An ineligible arbitrator cannot nominate another arbitrator. [E1]"
            if supplied_evidence
            else NO_EVIDENCE_ANSWER
        ),
        used_evidence_ids=["E1"] if supplied_evidence else [],
        evidence=supplied_evidence,
        retrieval_latency_ms=12.5,
        generation_latency_ms=23.75 if supplied_evidence else 0.0,
        total_latency_ms=36.25,
    )


def _verify_response() -> VerifyResponse:
    return VerifyResponse(
        claims=[
            VerificationClaim(
                claim_id="C1",
                claim=(
                    "An ineligible arbitrator cannot nominate another "
                    "arbitrator."
                ),
                citation_ids=["E1"],
                status=VerificationStatus.SUPPORTED,
                reason="E1 directly states the proposition.",
                evidence_uids=[_uid(1)],
            )
        ],
        summary=VerificationSummary(
            supported=1,
            partial=0,
            unsupported=0,
        ),
        claim_extraction_latency_ms=0.25,
        verification_latency_ms=31.5,
        total_latency_ms=31.75,
    )


@dataclass
class _AnswerStub:
    response: AnswerResponse = field(default_factory=_answer_response)
    error: Exception | None = None
    requests: list[AnswerRequest] = field(default_factory=list)

    def answer(self, request: AnswerRequest) -> AnswerResponse:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.response


@dataclass
class _VerificationStub:
    response: VerifyResponse = field(default_factory=_verify_response)
    error: Exception | None = None
    requests: list[VerifyRequest] = field(default_factory=list)

    def verify(self, request: VerifyRequest) -> VerifyResponse:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.response


@dataclass
class _ResearchStub:
    response: ResearchResponse | None = None
    error: Exception | None = None
    requests: list[ResearchRequest] = field(default_factory=list)

    def research(self, request: ResearchRequest) -> ResearchResponse:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


class _UnusedSearch:
    def search(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("search endpoint must not run in this test")


def _app_client(research_service: Any) -> TestClient:
    return TestClient(
        create_app(
            search_service=_UnusedSearch(),
            answer_service=_AnswerStub(),
            verification_service=_VerificationStub(),
            research_service=research_service,
        )
    )


def test_research_composes_existing_services_and_preserves_contract() -> None:
    answer = _AnswerStub()
    verifier = _VerificationStub()
    service = ResearchService(answer, verifier)
    filters = SearchFilters(
        court="Supreme Court of India",
        year=2017,
        case_number="Civil Appeal No. 5306 of 2017",
    )

    response = service.research(
        ResearchRequest(
            query="  ineligible arbitrator  ",
            top_k=7,
            filters=filters,
        )
    )

    assert answer.requests == [
        AnswerRequest(
            query="ineligible arbitrator",
            top_k=7,
            filters=filters,
        )
    ]
    assert verifier.requests == [
        VerifyRequest(
            answer=answer.response.answer,
            used_evidence_ids=["E1"],
            evidence=answer.response.evidence,
        )
    ]
    assert response.query == answer.response.query
    assert response.answer == answer.response.answer
    assert response.used_evidence_ids == ["E1"]
    assert response.evidence == answer.response.evidence
    assert response.claims == verifier.response.claims
    assert response.verification_summary == verifier.response.summary
    assert response.verification_state is ResearchVerificationState.COMPLETE
    assert response.verification_error is None
    assert response.latency.retrieval_ms == 12.5
    assert response.latency.generation_ms == 23.75
    assert response.latency.verification_ms >= 0
    assert response.latency.total_ms >= response.latency.verification_ms


@dataclass
class _EmptySearch:
    requests: list[Any] = field(default_factory=list)

    def search(self, request: Any) -> SearchResponse:
        self.requests.append(request)
        return SearchResponse(
            query=request.query,
            retrieval_mode=RetrievalMode.RERANKED,
            top_k=request.top_k,
            filters=request.filters,
            result_count=0,
            latency_ms=4.25,
            results=[],
        )


class _GenerationMustNotRun:
    def generate(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("Gemini generation must not run without evidence")


def test_no_evidence_skips_generation_and_verification() -> None:
    search = _EmptySearch()
    verifier = _VerificationStub(
        error=AssertionError("verification must not run without evidence")
    )
    service = ResearchService(
        AnswerService(search, _GenerationMustNotRun()),
        verifier,
    )

    response = service.research(ResearchRequest(query="out of corpus issue"))

    assert len(search.requests) == 1
    assert verifier.requests == []
    assert response.answer == NO_EVIDENCE_ANSWER
    assert response.evidence == []
    assert response.used_evidence_ids == []
    assert response.claims == []
    assert response.verification_summary == VerificationSummary(
        supported=0,
        partial=0,
        unsupported=0,
    )
    assert response.verification_state is ResearchVerificationState.NOT_RUN
    assert response.verification_error is None
    assert response.latency.retrieval_ms == 4.25
    assert response.latency.generation_ms == 0.0
    assert response.latency.verification_ms == 0.0


def test_generation_failure_prevents_verification() -> None:
    private_detail = "private generation provider detail"
    answer = _AnswerStub(error=ProviderUnavailableError(private_detail))
    verifier = _VerificationStub(
        error=AssertionError("verification must not follow generation failure")
    )

    with pytest.raises(ProviderUnavailableError, match=private_detail):
        ResearchService(answer, verifier).research(
            ResearchRequest(query="ineligible arbitrator")
        )

    assert len(answer.requests) == 1
    assert verifier.requests == []


@pytest.mark.parametrize(
    "error",
    [
        VerificationProviderUnavailableError("private quota and key detail"),
        RuntimeError("private unexpected verifier detail"),
    ],
    ids=["provider-error", "unexpected-error"],
)
def test_verification_failure_preserves_generated_answer_without_fabrication(
    error: Exception,
) -> None:
    answer = _AnswerStub()
    verifier = _VerificationStub(error=error)

    response = ResearchService(answer, verifier).research(
        ResearchRequest(query="ineligible arbitrator")
    )

    assert response.answer == answer.response.answer
    assert response.used_evidence_ids == ["E1"]
    assert response.evidence == answer.response.evidence
    assert response.claims == []
    assert response.verification_summary is None
    assert response.verification_state is ResearchVerificationState.UNAVAILABLE
    assert (
        response.verification_error
        == RESEARCH_VERIFICATION_UNAVAILABLE_ERROR
    )
    assert str(error) not in response.verification_error
    assert response.latency.verification_ms >= 0


def test_research_endpoint_returns_exact_success_shape_and_filters() -> None:
    answer = _AnswerStub()
    verifier = _VerificationStub()
    service = ResearchService(answer, verifier)

    with _app_client(service) as client:
        response = client.post(
            "/research",
            json={
                "query": "ineligible arbitrator",
                "top_k": 6,
                "filters": {
                    "court": "Supreme Court of India",
                    "year": 2017,
                    "case_number": "Civil Appeal No. 5306 of 2017",
                },
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "query",
        "answer",
        "used_evidence_ids",
        "evidence",
        "claims",
        "verification_summary",
        "verification_state",
        "verification_error",
        "latency",
    }
    assert body["verification_state"] == "complete"
    assert body["verification_summary"] == {
        "supported": 1,
        "partial": 0,
        "unsupported": 0,
    }
    assert body["claims"][0]["claim"] == verifier.response.claims[0].claim
    assert body["evidence"][0]["paragraph_uid"] == _uid(1)
    assert set(body["latency"]) == {
        "retrieval_ms",
        "generation_ms",
        "verification_ms",
        "total_ms",
    }
    assert answer.requests[0].top_k == 6
    assert answer.requests[0].filters.year == 2017


@pytest.mark.parametrize(
    "payload",
    [
        {"query": ""},
        {"query": "---"},
        {"query": "arbitration", "top_k": 0},
        {"query": "arbitration", "top_k": 51},
        {"query": "arbitration", "filters": {"year": "2017"}},
        {"query": "arbitration", "retrieval_mode": "bm25"},
        {"query": "arbitration", "evidence": []},
    ],
    ids=[
        "blank-query",
        "no-lexical-token",
        "low-top-k",
        "high-top-k",
        "invalid-year",
        "retrieval-mode-forbidden",
        "evidence-forbidden",
    ],
)
def test_research_endpoint_rejects_invalid_or_extra_input(
    payload: dict[str, Any],
) -> None:
    service = _ResearchStub()

    with _app_client(service) as client:
        response = client.post("/research", json=payload)

    assert response.status_code == 422
    assert service.requests == []


def test_research_generation_failure_returns_generic_503() -> None:
    private_detail = "private Gemini key and provider path"
    service = ResearchService(
        _AnswerStub(error=ProviderUnavailableError(private_detail)),
        _VerificationStub(),
    )

    with _app_client(service) as client:
        response = client.post(
            "/research",
            json={"query": "ineligible arbitrator"},
        )

    assert response.status_code == 503
    assert response.json() == {"detail": GENERATION_UNAVAILABLE_DETAIL}
    assert private_detail not in response.text
    assert "Traceback" not in response.text


def test_lifespan_composes_research_from_exact_existing_services() -> None:
    answer = _AnswerStub()
    verifier = _VerificationStub()
    factory_calls: list[tuple[Any, Any]] = []

    def factory(answer_service: Any, verification_service: Any) -> ResearchService:
        factory_calls.append((answer_service, verification_service))
        return ResearchService(answer_service, verification_service)

    application = create_app(
        search_service=_UnusedSearch(),
        answer_service=answer,
        verification_service=verifier,
        research_service_factory=factory,
    )
    with TestClient(application) as client:
        response = client.post(
            "/research",
            json={"query": "ineligible arbitrator"},
        )

    assert response.status_code == 200
    assert factory_calls == [(answer, verifier)]


def test_existing_endpoint_routes_remain_registered() -> None:
    paths = {
        route.path: route.methods
        for route in create_app().routes
        if hasattr(route, "methods")
    }

    assert paths["/health"] == {"GET"}
    assert paths["/search"] == {"POST"}
    assert paths["/answer"] == {"POST"}
    assert paths["/verify"] == {"POST"}
    assert paths["/research"] == {"POST"}


@pytest.mark.parametrize("origin", LOCAL_FRONTEND_ORIGINS)
def test_local_frontend_origins_pass_cors_preflight(origin: str) -> None:
    with _app_client(_ResearchStub()) as client:
        response = client.options(
            "/research",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin
    assert "POST" in response.headers["access-control-allow-methods"]


def test_untrusted_origin_is_not_granted_cors_access() -> None:
    with _app_client(_ResearchStub()) as client:
        response = client.options(
            "/research",
            headers={
                "Origin": "https://evil.example",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )

    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers


@pytest.mark.parametrize(
    "values",
    [
        {
            "verification_state": ResearchVerificationState.UNAVAILABLE,
            "verification_summary": VerificationSummary(
                supported=0,
                partial=0,
                unsupported=0,
            ),
            "verification_error": "Unavailable.",
        },
        {
            "verification_state": ResearchVerificationState.COMPLETE,
            "verification_summary": None,
            "verification_error": None,
        },
        {
            "verification_state": ResearchVerificationState.NOT_RUN,
            "verification_summary": VerificationSummary(
                supported=1,
                partial=0,
                unsupported=0,
            ),
            "verification_error": None,
        },
    ],
    ids=["unavailable-with-summary", "complete-without-summary", "not-run-results"],
)
def test_response_schema_rejects_inconsistent_verification_state(
    values: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        ResearchResponse(
            query="query",
            answer="answer",
            used_evidence_ids=[],
            evidence=[],
            claims=[],
            latency=ResearchLatency(
                retrieval_ms=0,
                generation_ms=0,
                verification_ms=0,
                total_ms=0,
            ),
            **values,
        )
