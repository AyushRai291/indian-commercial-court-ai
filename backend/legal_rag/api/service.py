"""Long-lived composition and orchestration for the retrieval API."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

from legal_rag.config import DEFAULT_RERANKER_CANDIDATE_K, Settings, get_settings
from legal_rag.database import get_engine, get_session_factory
from legal_rag.embeddings import SentenceTransformerEmbeddingProvider
from legal_rag.retrieval import (
    DEFAULT_CANDIDATE_DEPTH,
    DEFAULT_RRF_K,
    BM25ParagraphRetriever,
    CrossEncoderReranker,
    DenseParagraphRetriever,
    HybridParagraphRetriever,
    HybridSearchResult,
    ParagraphSearchResult,
    RerankedSearchResult,
    RetrievalFilters,
    SentenceTransformerCrossEncoderScorer,
)
from legal_rag.vector import QdrantParagraphIndex

from legal_rag.api.schemas import (
    RetrievalMode,
    SearchRequest,
    SearchResponse,
    SearchResult,
)


BM25_CANDIDATE_DEPTH = DEFAULT_CANDIDATE_DEPTH
DENSE_CANDIDATE_DEPTH = DEFAULT_CANDIDATE_DEPTH
RRF_K = DEFAULT_RRF_K
RERANKER_CANDIDATE_DEPTH = DEFAULT_RERANKER_CANDIDATE_K
MODEL_WARMUP_QUERY = "Indian commercial court legal research"


@dataclass(frozen=True, slots=True)
class ModelWarmupResult:
    """Mutation-free local-model startup timings in milliseconds."""

    dense_ms: float
    reranker_ms: float
    total_ms: float


class BasicRetriever(Protocol):
    """Interface used by native BM25 and dense searches."""

    def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: RetrievalFilters | None = None,
    ) -> list[ParagraphSearchResult]: ...


class HybridRetriever(Protocol):
    """Interface used by tuned hybrid search."""

    def search(
        self,
        query: str,
        *,
        top_k: int,
        bm25_candidate_depth: int | None = None,
        dense_candidate_depth: int | None = None,
        rrf_k: int | None = None,
        filters: RetrievalFilters | None = None,
    ) -> list[HybridSearchResult]: ...


class RerankedRetriever(Protocol):
    """Interface used by tuned hybrid plus cross-encoder search."""

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
    ) -> list[RerankedSearchResult]: ...


@dataclass(slots=True)
class SearchService:
    """Reuse one initialized set of retrievers across every HTTP request."""

    bm25_retriever: BasicRetriever
    dense_retriever: BasicRetriever
    hybrid_retriever: HybridRetriever
    reranker: RerankedRetriever

    def warmup(self) -> ModelWarmupResult:
        """Load both local ML models without querying or mutating storage."""

        total_started = perf_counter()
        embedding_provider = getattr(
            self.dense_retriever,
            "embedding_provider",
            None,
        )
        embed_query = getattr(embedding_provider, "embed_query", None)
        if not callable(embed_query):
            raise RuntimeError("dense embedding provider cannot be warmed")
        dense_started = perf_counter()
        embed_query(MODEL_WARMUP_QUERY)
        dense_ms = (perf_counter() - dense_started) * 1000.0

        scorer = getattr(self.reranker, "scorer", None)
        score_pairs = getattr(scorer, "score_pairs", None)
        if not callable(score_pairs):
            raise RuntimeError("cross-encoder scorer cannot be warmed")
        reranker_started = perf_counter()
        score_pairs(
            [(MODEL_WARMUP_QUERY, MODEL_WARMUP_QUERY)],
            batch_size=1,
        )
        reranker_ms = (perf_counter() - reranker_started) * 1000.0
        return ModelWarmupResult(
            dense_ms=dense_ms,
            reranker_ms=reranker_ms,
            total_ms=(perf_counter() - total_started) * 1000.0,
        )

    def search(self, request: SearchRequest) -> SearchResponse:
        """Execute the selected existing retriever and serialize its evidence."""

        filters = request.filters.to_retrieval_filters()
        started = perf_counter()
        results = self._retrieve(request, filters)
        latency_ms = (perf_counter() - started) * 1000.0
        serialized = [
            _serialize_result(result, request.retrieval_mode) for result in results
        ]
        return SearchResponse(
            query=request.query,
            retrieval_mode=request.retrieval_mode,
            top_k=request.top_k,
            filters=request.filters,
            result_count=len(serialized),
            latency_ms=latency_ms,
            results=serialized,
        )

    def _retrieve(
        self,
        request: SearchRequest,
        filters: RetrievalFilters | None,
    ) -> list[ParagraphSearchResult]:
        if request.retrieval_mode is RetrievalMode.BM25:
            return self.bm25_retriever.search(
                request.query,
                top_k=request.top_k,
                filters=filters,
            )
        if request.retrieval_mode is RetrievalMode.DENSE:
            return self.dense_retriever.search(
                request.query,
                top_k=request.top_k,
                filters=filters,
            )
        if request.retrieval_mode is RetrievalMode.HYBRID:
            return self.hybrid_retriever.search(
                request.query,
                top_k=request.top_k,
                bm25_candidate_depth=BM25_CANDIDATE_DEPTH,
                dense_candidate_depth=DENSE_CANDIDATE_DEPTH,
                rrf_k=RRF_K,
                filters=filters,
            )
        return self.reranker.search(
            request.query,
            top_k=request.top_k,
            candidate_k=RERANKER_CANDIDATE_DEPTH,
            bm25_candidate_depth=BM25_CANDIDATE_DEPTH,
            dense_candidate_depth=DENSE_CANDIDATE_DEPTH,
            rrf_k=RRF_K,
            filters=filters,
        )

    def close(self) -> None:
        """Release the reusable Qdrant client's transport at app shutdown."""

        paragraph_index = getattr(self.dense_retriever, "paragraph_index", None)
        client = getattr(paragraph_index, "client", None)
        close = getattr(client, "close", None)
        if callable(close):
            close()


def build_search_service(settings: Settings | None = None) -> SearchService:
    """Compose the existing retrievers; the API lifespan performs warmup."""

    runtime_settings = settings or get_settings()
    session_factory = get_session_factory(get_engine(runtime_settings.database_url))
    with session_factory() as session:
        bm25_retriever = BM25ParagraphRetriever.from_session(session)

    embedding_provider = SentenceTransformerEmbeddingProvider(
        runtime_settings.embedding_model,
        expected_dimension=runtime_settings.embedding_dimension,
    )
    paragraph_index = QdrantParagraphIndex(
        url=runtime_settings.qdrant_url,
        api_key=_qdrant_api_key(runtime_settings.qdrant_api_key),
        collection_name=runtime_settings.qdrant_collection,
    )
    dense_retriever = DenseParagraphRetriever(paragraph_index, embedding_provider)
    hybrid_retriever = HybridParagraphRetriever(
        bm25_retriever,
        dense_retriever,
        bm25_candidate_depth=BM25_CANDIDATE_DEPTH,
        dense_candidate_depth=DENSE_CANDIDATE_DEPTH,
        rrf_k=RRF_K,
    )
    reranker = CrossEncoderReranker(
        hybrid_retriever,
        SentenceTransformerCrossEncoderScorer(runtime_settings.reranker_model),
        candidate_k=RERANKER_CANDIDATE_DEPTH,
        batch_size=runtime_settings.reranker_batch_size,
    )
    return SearchService(
        bm25_retriever=bm25_retriever,
        dense_retriever=dense_retriever,
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
    )


def _qdrant_api_key(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "get_secret_value"):
        return str(value.get_secret_value())
    return str(value)


def _serialize_result(
    result: ParagraphSearchResult,
    mode: RetrievalMode,
) -> SearchResult:
    bm25_rank: int | None = None
    bm25_score: float | None = None
    dense_rank: int | None = None
    dense_score: float | None = None
    rrf_score: float | None = None
    hybrid_rank: int | None = None
    cross_encoder_score: float | None = None
    final_rank = result.rank

    if mode is RetrievalMode.BM25:
        bm25_rank = result.rank
        bm25_score = result.score
    elif mode is RetrievalMode.DENSE:
        dense_rank = result.rank
        dense_score = result.score
    else:
        hybrid_result = result
        if not isinstance(hybrid_result, HybridSearchResult):
            raise TypeError("hybrid retrieval returned an incompatible result")
        bm25_rank = hybrid_result.bm25_rank
        bm25_score = hybrid_result.bm25_score
        dense_rank = hybrid_result.dense_rank
        dense_score = hybrid_result.dense_score
        rrf_score = hybrid_result.rrf_score
        hybrid_rank = hybrid_result.hybrid_rank if isinstance(
            hybrid_result, RerankedSearchResult
        ) else hybrid_result.rank
        if mode is RetrievalMode.RERANKED:
            if not isinstance(hybrid_result, RerankedSearchResult):
                raise TypeError("reranked retrieval returned an incompatible result")
            cross_encoder_score = hybrid_result.cross_encoder_score
            final_rank = hybrid_result.reranked_rank

    return SearchResult(
        paragraph_uid=result.paragraph_uid,
        text=result.text,
        case_id=result.case_id,
        title=result.title,
        case_number=result.case_number,
        court=result.court,
        judgment_date=result.judgment_date,
        source_url=result.source_url,
        paragraph_number=result.paragraph_number,
        page_number=result.page_number,
        bm25_rank=bm25_rank,
        bm25_score=bm25_score,
        dense_rank=dense_rank,
        dense_score=dense_score,
        rrf_score=rrf_score,
        hybrid_rank=hybrid_rank,
        cross_encoder_score=cross_encoder_score,
        final_rank=final_rank,
    )
