"""Compose grounded answering and claim verification without duplicating them."""

from __future__ import annotations

from time import perf_counter
from typing import Protocol

from legal_rag.api.schemas import (
    AnswerRequest,
    AnswerResponse,
    ResearchLatency,
    ResearchRequest,
    ResearchResponse,
    ResearchVerificationState,
    VerificationClaim,
    VerificationSummary,
    VerifyRequest,
    VerifyResponse,
)


RESEARCH_VERIFICATION_UNAVAILABLE_ERROR = (
    "Citation verification is temporarily unavailable."
)


class AnswerExecutor(Protocol):
    """Existing grounded answer service boundary."""

    def answer(self, request: AnswerRequest) -> AnswerResponse: ...


class VerificationExecutor(Protocol):
    """Existing claim verification service boundary."""

    def verify(self, request: VerifyRequest) -> VerifyResponse: ...


class ResearchService:
    """Run answer generation, then verify that exact answer and evidence."""

    def __init__(
        self,
        answer_service: AnswerExecutor,
        verification_service: VerificationExecutor,
    ) -> None:
        self.answer_service = answer_service
        self.verification_service = verification_service

    def research(self, request: ResearchRequest) -> ResearchResponse:
        total_started = perf_counter()
        answer_response = self.answer_service.answer(
            AnswerRequest(
                query=request.query,
                top_k=request.top_k,
                filters=request.filters,
            )
        )

        if not answer_response.evidence:
            return self._response(
                answer_response,
                claims=[],
                verification_summary=VerificationSummary(
                    supported=0,
                    partial=0,
                    unsupported=0,
                ),
                verification_state=ResearchVerificationState.NOT_RUN,
                verification_error=None,
                verification_ms=0.0,
                total_started=total_started,
            )

        verification_started = perf_counter()
        try:
            verification_response = self.verification_service.verify(
                VerifyRequest(
                    answer=answer_response.answer,
                    used_evidence_ids=answer_response.used_evidence_ids,
                    evidence=answer_response.evidence,
                )
            )
        except Exception:
            verification_ms = (
                perf_counter() - verification_started
            ) * 1000.0
            return self._response(
                answer_response,
                claims=[],
                verification_summary=None,
                verification_state=ResearchVerificationState.UNAVAILABLE,
                verification_error=RESEARCH_VERIFICATION_UNAVAILABLE_ERROR,
                verification_ms=verification_ms,
                total_started=total_started,
            )

        verification_ms = (
            perf_counter() - verification_started
        ) * 1000.0
        return self._response(
            answer_response,
            claims=verification_response.claims,
            verification_summary=verification_response.summary,
            verification_state=ResearchVerificationState.COMPLETE,
            verification_error=None,
            verification_ms=verification_ms,
            total_started=total_started,
        )

    @staticmethod
    def _response(
        answer_response: AnswerResponse,
        *,
        claims: list[VerificationClaim],
        verification_summary: VerificationSummary | None,
        verification_state: ResearchVerificationState,
        verification_error: str | None,
        verification_ms: float,
        total_started: float,
    ) -> ResearchResponse:
        return ResearchResponse(
            query=answer_response.query,
            answer=answer_response.answer,
            used_evidence_ids=answer_response.used_evidence_ids,
            evidence=answer_response.evidence,
            claims=claims,
            verification_summary=verification_summary,
            verification_state=verification_state,
            verification_error=verification_error,
            latency=ResearchLatency(
                retrieval_ms=answer_response.retrieval_latency_ms,
                generation_ms=answer_response.generation_latency_ms,
                verification_ms=verification_ms,
                total_ms=(perf_counter() - total_started) * 1000.0,
            ),
        )


def build_research_service(
    answer_service: AnswerExecutor,
    verification_service: VerificationExecutor,
) -> ResearchService:
    """Compose the existing answer and verification services."""

    return ResearchService(answer_service, verification_service)
