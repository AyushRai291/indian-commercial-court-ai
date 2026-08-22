"""Rank-only fusion of independent BM25 and dense paragraph retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

from legal_rag.retrieval.filters import RetrievalFilters
from legal_rag.retrieval.results import HybridSearchResult, ParagraphSearchResult


DEFAULT_CANDIDATE_DEPTH = 50
DEFAULT_RRF_K = 10


class RankedParagraphRetriever(Protocol):
    """Structural interface shared by paragraph retrievers used for fusion."""

    def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: RetrievalFilters | None = None,
    ) -> list[ParagraphSearchResult]: ...


@dataclass(frozen=True, slots=True)
class HybridSearchDiagnostics:
    """Candidate counts and wall-clock timings for one hybrid search."""

    bm25_candidates: int
    dense_candidates: int
    unique_candidates: int
    fusion_seconds: float
    total_seconds: float


@dataclass(slots=True)
class _FusedCandidate:
    result: ParagraphSearchResult
    score: float = 0.0
    bm25_rank: int | None = None
    dense_rank: int | None = None
    bm25_score: float | None = None
    dense_score: float | None = None


def _validate_positive(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def reciprocal_rank_fusion(
    bm25_results: Sequence[ParagraphSearchResult],
    dense_results: Sequence[ParagraphSearchResult],
    *,
    top_k: int,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[HybridSearchResult]:
    """Fuse two ranked lists using ``sum(1 / (rrf_k + rank))``.

    Input order supplies the one-based rank position. Duplicate UIDs within one
    input list contribute only at their first position. Native score magnitudes
    are retained as provenance but never participate in fusion.
    """

    _validate_positive("top_k", top_k)
    _validate_positive("rrf_k", rrf_k)

    candidates: dict[str, _FusedCandidate] = {}

    def add_results(
        results: Sequence[ParagraphSearchResult],
        source: str,
    ) -> None:
        seen_uids: set[str] = set()
        for rank, result in enumerate(results, start=1):
            paragraph_uid = result.paragraph_uid.strip()
            if not paragraph_uid:
                raise ValueError("paragraph_uid must not be empty")
            if paragraph_uid in seen_uids:
                continue
            seen_uids.add(paragraph_uid)

            candidate = candidates.setdefault(
                paragraph_uid,
                _FusedCandidate(result=result),
            )
            candidate.score += 1.0 / (rrf_k + rank)
            if source == "bm25":
                candidate.bm25_rank = rank
                candidate.bm25_score = float(result.score)
            else:
                candidate.dense_rank = rank
                candidate.dense_score = float(result.score)

    add_results(bm25_results, "bm25")
    add_results(dense_results, "dense")

    ordered = sorted(
        candidates.items(),
        key=lambda item: (-item[1].score, item[0]),
    )[:top_k]
    return [
        HybridSearchResult(
            paragraph_uid=paragraph_uid,
            text=candidate.result.text,
            case_id=candidate.result.case_id,
            title=candidate.result.title,
            case_number=candidate.result.case_number,
            court=candidate.result.court,
            judgment_date=candidate.result.judgment_date,
            source_url=candidate.result.source_url,
            paragraph_number=candidate.result.paragraph_number,
            page_number=candidate.result.page_number,
            score=candidate.score,
            rank=final_rank,
            bm25_rank=candidate.bm25_rank,
            dense_rank=candidate.dense_rank,
            bm25_score=candidate.bm25_score,
            dense_score=candidate.dense_score,
        )
        for final_rank, (paragraph_uid, candidate) in enumerate(ordered, start=1)
    ]


class HybridParagraphRetriever:
    """Retrieve independent candidate lists and combine them using RRF."""

    def __init__(
        self,
        bm25_retriever: RankedParagraphRetriever,
        dense_retriever: RankedParagraphRetriever,
        *,
        bm25_candidate_depth: int = DEFAULT_CANDIDATE_DEPTH,
        dense_candidate_depth: int = DEFAULT_CANDIDATE_DEPTH,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        _validate_positive("bm25_candidate_depth", bm25_candidate_depth)
        _validate_positive("dense_candidate_depth", dense_candidate_depth)
        _validate_positive("rrf_k", rrf_k)
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.bm25_candidate_depth = bm25_candidate_depth
        self.dense_candidate_depth = dense_candidate_depth
        self.rrf_k = rrf_k

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
        """Return the fused results without diagnostics."""

        results, _ = self.search_with_diagnostics(
            query,
            top_k=top_k,
            bm25_candidate_depth=bm25_candidate_depth,
            dense_candidate_depth=dense_candidate_depth,
            rrf_k=rrf_k,
            filters=filters,
        )
        return results

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
        """Return fused results plus candidate counts and latency."""

        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            raise ValueError("query must not be empty")
        _validate_positive("top_k", top_k)
        bm25_depth = (
            self.bm25_candidate_depth
            if bm25_candidate_depth is None
            else bm25_candidate_depth
        )
        dense_depth = (
            self.dense_candidate_depth
            if dense_candidate_depth is None
            else dense_candidate_depth
        )
        fusion_k = self.rrf_k if rrf_k is None else rrf_k
        _validate_positive("bm25_candidate_depth", bm25_depth)
        _validate_positive("dense_candidate_depth", dense_depth)
        _validate_positive("rrf_k", fusion_k)

        total_started = perf_counter()
        bm25_results = self.bm25_retriever.search(
            query,
            top_k=bm25_depth,
            filters=filters,
        )
        dense_results = self.dense_retriever.search(
            query,
            top_k=dense_depth,
            filters=filters,
        )
        fusion_started = perf_counter()
        results = reciprocal_rank_fusion(
            bm25_results,
            dense_results,
            top_k=top_k,
            rrf_k=fusion_k,
        )
        fusion_seconds = perf_counter() - fusion_started
        total_seconds = perf_counter() - total_started
        unique_candidates = len(
            {result.paragraph_uid for result in bm25_results}
            | {result.paragraph_uid for result in dense_results}
        )
        return results, HybridSearchDiagnostics(
            bm25_candidates=len(bm25_results),
            dense_candidates=len(dense_results),
            unique_candidates=unique_candidates,
            fusion_seconds=fusion_seconds,
            total_seconds=total_seconds,
        )
