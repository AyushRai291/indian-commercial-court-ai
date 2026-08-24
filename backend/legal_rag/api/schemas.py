"""Explicit API contracts for search, answers, and citation verification."""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from legal_rag.retrieval import (
    RetrievalFilters,
    build_retrieval_filters,
    tokenize_legal_text,
)
from legal_rag.verification.models import VerificationStatus


class RetrievalMode(str, Enum):
    """Retrieval pipelines exposed by the API."""

    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"
    RERANKED = "reranked"


StrictTopK = Annotated[int, Field(strict=True, ge=1, le=50)]
StrictYear = Annotated[int, Field(strict=True, ge=1, le=9999)]


class SearchFilters(BaseModel):
    """Optional exact-match metadata constraints."""

    model_config = ConfigDict(extra="forbid")

    court: str | None = None
    year: StrictYear | None = None
    case_number: str | None = None

    @model_validator(mode="after")
    def validate_retrieval_contract(self) -> "SearchFilters":
        """Delegate normalization rules and edge cases to the shared contract."""

        RetrievalFilters(
            court=self.court,
            year=self.year,
            case_number=self.case_number,
        )
        return self

    def to_retrieval_filters(self) -> RetrievalFilters | None:
        """Build the existing retrieval-layer filter object."""

        return build_retrieval_filters(
            court=self.court,
            year=self.year,
            case_number=self.case_number,
        )


class RetrievalRequest(BaseModel):
    """Shared validated query, result count, and metadata filters."""

    model_config = ConfigDict(extra="forbid")

    query: str
    top_k: StrictTopK = 10
    filters: SearchFilters = Field(default_factory=SearchFilters)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("query must not be empty")
        if not tokenize_legal_text(stripped):
            raise ValueError("query must contain at least one lexical token")
        return stripped


class SearchRequest(RetrievalRequest):
    """Validated legal paragraph search request."""

    retrieval_mode: RetrievalMode = RetrievalMode.RERANKED


class AnswerRequest(RetrievalRequest):
    """Validated grounded-answer request over server-retrieved evidence."""


class SearchResult(BaseModel):
    """One ranked paragraph with canonical metadata and score provenance."""

    model_config = ConfigDict(extra="forbid")

    paragraph_uid: str
    text: str
    case_id: int
    title: str
    case_number: str | None
    court: str | None
    judgment_date: date | None
    source_url: str | None
    paragraph_number: int
    page_number: int | None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    dense_rank: int | None = None
    dense_score: float | None = None
    rrf_score: float | None = None
    hybrid_rank: int | None = None
    cross_encoder_score: float | None = None
    final_rank: int


class SearchResponse(BaseModel):
    """Complete search response, including the effective request and timing."""

    model_config = ConfigDict(extra="forbid")

    query: str
    retrieval_mode: RetrievalMode
    top_k: int
    filters: SearchFilters
    result_count: int
    latency_ms: float
    results: list[SearchResult]


class AnswerEvidence(BaseModel):
    """One request-local evidence item with complete retrieval provenance."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    paragraph_uid: str
    text: str
    case_id: int
    case_name: str
    case_number: str | None
    court: str | None
    judgment_date: date | None
    source_url: str | None
    paragraph_number: int
    page_number: int | None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    dense_rank: int | None = None
    dense_score: float | None = None
    rrf_score: float | None = None
    hybrid_rank: int | None = None
    cross_encoder_score: float | None = None
    reranked_rank: int


class AnswerResponse(BaseModel):
    """Grounded answer, cited evidence, and stage-level latency."""

    model_config = ConfigDict(extra="forbid")

    query: str
    answer: str
    used_evidence_ids: list[str]
    evidence: list[AnswerEvidence]
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float


class VerifyRequest(BaseModel):
    """Existing grounded answer and full evidence supplied for verification."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1, max_length=50_000)
    used_evidence_ids: list[str] = Field(max_length=50)
    evidence: list[AnswerEvidence] = Field(max_length=50)

    @field_validator("answer")
    @classmethod
    def answer_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("answer must not be blank")
        return stripped


class VerificationClaim(BaseModel):
    """One material claim with semantic support and durable provenance."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    claim: str
    citation_ids: list[str]
    status: VerificationStatus
    reason: str
    evidence_uids: list[str]


class VerificationSummary(BaseModel):
    """Transparent status counts without a synthetic confidence score."""

    model_config = ConfigDict(extra="forbid")

    supported: int
    partial: int
    unsupported: int


class VerifyResponse(BaseModel):
    """Ordered claim verification plus extraction/provider timing."""

    model_config = ConfigDict(extra="forbid")

    claims: list[VerificationClaim]
    summary: VerificationSummary
    claim_extraction_latency_ms: float
    verification_latency_ms: float
    total_latency_ms: float


class HealthResponse(BaseModel):
    """Lightweight process-liveness response."""

    status: str
