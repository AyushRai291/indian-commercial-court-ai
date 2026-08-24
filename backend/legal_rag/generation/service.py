"""Orchestrate reranked retrieval, grounded generation, and timing."""

from __future__ import annotations

from time import perf_counter
from typing import Protocol

from pydantic import ValidationError

from legal_rag.api.schemas import (
    AnswerRequest,
    AnswerResponse,
    RetrievalMode,
    SearchRequest,
    SearchResponse,
)
from legal_rag.config import Settings, get_settings
from legal_rag.generation.errors import (
    GenerationError,
    MalformedModelResponseError,
    ProviderUnavailableError,
)
from legal_rag.generation.evidence import (
    assign_evidence_ids,
    validate_citation_integrity,
)
from legal_rag.generation.models import GroundedModelOutput
from legal_rag.generation.prompt import build_grounded_prompt
from legal_rag.generation.provider import (
    GroundedAnswerProvider,
    OpenAIResponsesProvider,
)


NO_EVIDENCE_ANSWER = (
    "The retrieved evidence is insufficient to answer this question."
)


class SearchExecutor(Protocol):
    """The existing search service boundary reused by answer generation."""

    def search(self, request: SearchRequest) -> SearchResponse: ...


class AnswerService:
    """Generate answers exclusively over server-retrieved reranked evidence."""

    def __init__(
        self,
        search_service: SearchExecutor,
        provider: GroundedAnswerProvider,
    ) -> None:
        self.search_service = search_service
        self.provider = provider

    def answer(self, request: AnswerRequest) -> AnswerResponse:
        total_started = perf_counter()
        search_response = self.search_service.search(
            SearchRequest(
                query=request.query,
                top_k=request.top_k,
                filters=request.filters,
                retrieval_mode=RetrievalMode.RERANKED,
            )
        )
        evidence = assign_evidence_ids(search_response.results)

        if not evidence:
            return AnswerResponse(
                query=request.query,
                answer=NO_EVIDENCE_ANSWER,
                used_evidence_ids=[],
                evidence=[],
                retrieval_latency_ms=search_response.latency_ms,
                generation_latency_ms=0.0,
                total_latency_ms=(perf_counter() - total_started) * 1000.0,
            )

        generation_started = perf_counter()
        prompt = build_grounded_prompt(request.query, evidence)
        try:
            raw_output = self.provider.generate(prompt)
            output = GroundedModelOutput.model_validate(raw_output)
            output = validate_citation_integrity(
                output,
                (item.evidence_id for item in evidence),
            )
        except GenerationError:
            raise
        except ValidationError as exc:
            raise MalformedModelResponseError(
                "provider response did not match the structured contract"
            ) from exc
        except Exception as exc:
            raise ProviderUnavailableError("answer provider failed") from exc
        generation_latency_ms = (perf_counter() - generation_started) * 1000.0

        return AnswerResponse(
            query=request.query,
            answer=output.answer,
            used_evidence_ids=output.used_evidence_ids,
            evidence=evidence,
            retrieval_latency_ms=search_response.latency_ms,
            generation_latency_ms=generation_latency_ms,
            total_latency_ms=(perf_counter() - total_started) * 1000.0,
        )

    def close(self) -> None:
        close = getattr(self.provider, "close", None)
        if callable(close):
            close()


def build_answer_service(
    search_service: SearchExecutor,
    settings: Settings | None = None,
) -> AnswerService:
    """Compose the single configured provider without requiring a key at boot."""

    runtime_settings = settings or get_settings()
    provider = OpenAIResponsesProvider(
        api_key=runtime_settings.openai_api_key,
        model=runtime_settings.openai_model,
        timeout_seconds=runtime_settings.openai_timeout_seconds,
    )
    return AnswerService(search_service, provider)
