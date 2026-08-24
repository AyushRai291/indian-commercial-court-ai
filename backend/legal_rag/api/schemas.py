"""Explicit request and response contracts for paragraph search."""

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


class SearchRequest(BaseModel):
    """Validated legal paragraph search request."""

    model_config = ConfigDict(extra="forbid")

    query: str
    top_k: StrictTopK = 10
    retrieval_mode: RetrievalMode = RetrievalMode.RERANKED
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


class HealthResponse(BaseModel):
    """Lightweight process-liveness response."""

    status: str
