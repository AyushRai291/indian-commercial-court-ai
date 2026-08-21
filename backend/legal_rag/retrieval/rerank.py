"""Lazy batched cross-encoder reranking over hybrid paragraph candidates."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

from legal_rag.config import (
    DEFAULT_RERANKER_BATCH_SIZE,
    DEFAULT_RERANKER_CANDIDATE_K,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RERANKER_TOP_K,
)
from legal_rag.retrieval.filters import RetrievalFilters
from legal_rag.retrieval.hybrid import (
    HybridParagraphRetriever,
    HybridSearchDiagnostics,
)
from legal_rag.retrieval.results import HybridSearchResult, RerankedSearchResult


@dataclass(frozen=True, slots=True)
class CrossEncoderBatchResult:
    """Native scores and timings for one batched cross-encoder prediction."""

    scores: tuple[float, ...]
    model_load_seconds: float
    inference_seconds: float


class CrossEncoderScorer(Protocol):
    """Structural interface for independently testable pair scorers."""

    def score_pairs(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        batch_size: int,
    ) -> CrossEncoderBatchResult: ...


class SentenceTransformerCrossEncoderScorer:
    """Lazily load and reuse a Sentence Transformers CrossEncoder."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        *,
        device: str | None = None,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must not be empty")
        self.model_name = model_name
        self.device = device
        self._model: Any | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _load_model(self) -> tuple[Any, float]:
        if self._model is not None:
            return self._model, 0.0

        load_started = perf_counter()
        try:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(self.model_name, device=self.device)
        except Exception as error:
            raise RuntimeError(
                f"Unable to load cross-encoder model {self.model_name!r}"
            ) from error
        load_seconds = perf_counter() - load_started
        self._model = model
        return model, load_seconds

    def score_pairs(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        batch_size: int,
    ) -> CrossEncoderBatchResult:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        values = list(pairs)
        if not values:
            return CrossEncoderBatchResult((), 0.0, 0.0)
        if any(
            not isinstance(pair, tuple)
            or len(pair) != 2
            or not all(isinstance(value, str) for value in pair)
            for pair in values
        ):
            raise TypeError("pairs must contain two-string tuples")

        model, load_seconds = self._load_model()
        inference_started = perf_counter()
        try:
            raw_scores = model.predict(
                values,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                apply_softmax=False,
            )
        except Exception as error:
            raise RuntimeError(
                f"Cross-encoder inference failed for model {self.model_name!r} "
                f"while scoring {len(values)} candidates"
            ) from error
        inference_seconds = perf_counter() - inference_started
        scores = _coerce_scores(raw_scores, expected_count=len(values))
        return CrossEncoderBatchResult(
            scores=tuple(scores),
            model_load_seconds=load_seconds,
            inference_seconds=inference_seconds,
        )


def _coerce_scores(raw_scores: object, *, expected_count: int) -> list[float]:
    values: object = (
        raw_scores.tolist() if hasattr(raw_scores, "tolist") else raw_scores
    )
    if isinstance(values, (int, float)):
        items = [values]
    else:
        try:
            items = list(values)  # type: ignore[arg-type]
        except TypeError as error:
            raise RuntimeError("Cross-encoder returned non-sequential scores") from error

    scores: list[float] = []
    for item in items:
        if isinstance(item, (list, tuple)):
            if len(item) != 1:
                raise RuntimeError("Cross-encoder returned multi-label scores")
            item = item[0]
        try:
            score = float(item)
        except (TypeError, ValueError) as error:
            raise RuntimeError("Cross-encoder returned a non-numeric score") from error
        if not math.isfinite(score):
            raise RuntimeError("Cross-encoder returned a non-finite score")
        scores.append(score)

    if len(scores) != expected_count:
        raise RuntimeError(
            "Cross-encoder returned a different number of scores than candidates"
        )
    return scores


@dataclass(frozen=True, slots=True)
class RerankSearchDiagnostics:
    """Candidate counts and stage timings for one reranked search."""

    hybrid_candidates: int
    unique_candidates: int
    hybrid_seconds: float
    model_load_seconds: float
    inference_seconds: float
    total_seconds: float
    hybrid: HybridSearchDiagnostics


class CrossEncoderReranker:
    """Rerank only the leading hybrid candidates with native model scores."""

    def __init__(
        self,
        hybrid_retriever: HybridParagraphRetriever,
        scorer: CrossEncoderScorer,
        *,
        candidate_k: int = DEFAULT_RERANKER_CANDIDATE_K,
        batch_size: int = DEFAULT_RERANKER_BATCH_SIZE,
    ) -> None:
        _validate_positive("candidate_k", candidate_k)
        _validate_positive("batch_size", batch_size)
        self.hybrid_retriever = hybrid_retriever
        self.scorer = scorer
        self.candidate_k = candidate_k
        self.batch_size = batch_size

    def search(
        self,
        query: str,
        *,
        top_k: int = DEFAULT_RERANKER_TOP_K,
        candidate_k: int | None = None,
        bm25_candidate_depth: int | None = None,
        dense_candidate_depth: int | None = None,
        rrf_k: int | None = None,
        batch_size: int | None = None,
        filters: RetrievalFilters | None = None,
    ) -> list[RerankedSearchResult]:
        results, _ = self.search_with_diagnostics(
            query,
            top_k=top_k,
            candidate_k=candidate_k,
            bm25_candidate_depth=bm25_candidate_depth,
            dense_candidate_depth=dense_candidate_depth,
            rrf_k=rrf_k,
            batch_size=batch_size,
            filters=filters,
        )
        return results

    def search_with_diagnostics(
        self,
        query: str,
        *,
        top_k: int = DEFAULT_RERANKER_TOP_K,
        candidate_k: int | None = None,
        bm25_candidate_depth: int | None = None,
        dense_candidate_depth: int | None = None,
        rrf_k: int | None = None,
        batch_size: int | None = None,
        filters: RetrievalFilters | None = None,
    ) -> tuple[list[RerankedSearchResult], RerankSearchDiagnostics]:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            raise ValueError("query must not be empty")
        _validate_positive("top_k", top_k)
        requested_candidates = self.candidate_k if candidate_k is None else candidate_k
        inference_batch_size = self.batch_size if batch_size is None else batch_size
        _validate_positive("candidate_k", requested_candidates)
        _validate_positive("batch_size", inference_batch_size)

        total_started = perf_counter()
        hybrid_started = perf_counter()
        candidates, hybrid_diagnostics = (
            self.hybrid_retriever.search_with_diagnostics(
                query,
                top_k=requested_candidates,
                bm25_candidate_depth=bm25_candidate_depth,
                dense_candidate_depth=dense_candidate_depth,
                rrf_k=rrf_k,
                filters=filters,
            )
        )
        hybrid_seconds = perf_counter() - hybrid_started

        unique_candidates: list[HybridSearchResult] = []
        seen_uids: set[str] = set()
        for candidate in candidates:
            if candidate.paragraph_uid in seen_uids:
                continue
            seen_uids.add(candidate.paragraph_uid)
            unique_candidates.append(candidate)

        if not unique_candidates:
            return [], RerankSearchDiagnostics(
                hybrid_candidates=len(candidates),
                unique_candidates=0,
                hybrid_seconds=hybrid_seconds,
                model_load_seconds=0.0,
                inference_seconds=0.0,
                total_seconds=perf_counter() - total_started,
                hybrid=hybrid_diagnostics,
            )

        prediction = self.scorer.score_pairs(
            [(query, candidate.text) for candidate in unique_candidates],
            batch_size=inference_batch_size,
        )
        if len(prediction.scores) != len(unique_candidates):
            raise RuntimeError(
                "Cross-encoder returned a different number of scores than candidates"
            )
        ordered = sorted(
            zip(prediction.scores, unique_candidates, strict=True),
            key=lambda item: (
                -item[0],
                item[1].rank,
                item[1].paragraph_uid,
            ),
        )[:top_k]
        results = [
            _to_reranked(candidate, cross_encoder_score, final_rank)
            for final_rank, (cross_encoder_score, candidate) in enumerate(
                ordered,
                start=1,
            )
        ]
        return results, RerankSearchDiagnostics(
            hybrid_candidates=len(candidates),
            unique_candidates=len(unique_candidates),
            hybrid_seconds=hybrid_seconds,
            model_load_seconds=prediction.model_load_seconds,
            inference_seconds=prediction.inference_seconds,
            total_seconds=perf_counter() - total_started,
            hybrid=hybrid_diagnostics,
        )


def _validate_positive(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _to_reranked(
    candidate: HybridSearchResult,
    cross_encoder_score: float,
    reranked_rank: int,
) -> RerankedSearchResult:
    return RerankedSearchResult(
        paragraph_uid=candidate.paragraph_uid,
        text=candidate.text,
        case_id=candidate.case_id,
        title=candidate.title,
        case_number=candidate.case_number,
        court=candidate.court,
        judgment_date=candidate.judgment_date,
        source_url=candidate.source_url,
        paragraph_number=candidate.paragraph_number,
        page_number=candidate.page_number,
        score=candidate.score,
        rank=candidate.rank,
        bm25_rank=candidate.bm25_rank,
        dense_rank=candidate.dense_rank,
        bm25_score=candidate.bm25_score,
        dense_score=candidate.dense_score,
        cross_encoder_score=float(cross_encoder_score),
        reranked_rank=reranked_rank,
    )
