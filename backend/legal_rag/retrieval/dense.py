"""Convert storage-level Qdrant hits to the shared retrieval contract."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime

from qdrant_client.models import FieldCondition, Filter, MatchValue

from legal_rag.embeddings import EmbeddingProvider
from legal_rag.retrieval.filters import (
    CASE_NUMBER_FILTER_FIELD,
    COURT_FILTER_FIELD,
    JUDGMENT_YEAR_FILTER_FIELD,
    RetrievalFilters,
)
from legal_rag.retrieval.results import ParagraphSearchResult
from legal_rag.vector import QdrantParagraphIndex, SemanticSearchResult


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_date(value: object) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as error:
        raise ValueError(f"invalid judgment_date in vector payload: {value!r}") from error


def semantic_hits_to_results(
    hits: Sequence[SemanticSearchResult],
) -> list[ParagraphSearchResult]:
    """Map already-ranked Qdrant hits without changing their order or scores."""

    results: list[ParagraphSearchResult] = []
    seen_uids: set[str] = set()
    for hit in hits:
        payload = hit.payload
        paragraph_uid = str(payload.get("paragraph_uid") or hit.point_id)
        if paragraph_uid in seen_uids:
            continue
        if str(hit.point_id) != paragraph_uid:
            raise ValueError(
                "Qdrant point ID does not match payload paragraph_uid: "
                f"{hit.point_id!r} != {paragraph_uid!r}"
            )
        seen_uids.add(paragraph_uid)

        try:
            case_id = int(payload["case_id"])
            paragraph_number = int(payload["paragraph_number"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "vector payload must contain integer case_id and paragraph_number"
            ) from error
        page_value = payload.get("page_number")
        try:
            page_number = int(page_value) if page_value is not None else None
        except (TypeError, ValueError) as error:
            raise ValueError("vector payload page_number must be an integer") from error

        results.append(
            ParagraphSearchResult(
                paragraph_uid=paragraph_uid,
                text=str(payload.get("text") or ""),
                case_id=case_id,
                title=str(payload.get("title") or ""),
                case_number=_optional_string(payload.get("case_number")),
                court=_optional_string(payload.get("court")),
                judgment_date=_optional_date(payload.get("judgment_date")),
                source_url=_optional_string(payload.get("source_url")),
                paragraph_number=paragraph_number,
                page_number=page_number,
                score=float(hit.score),
                rank=len(results) + 1,
            )
        )
    return results


def search_dense(
    paragraph_index: QdrantParagraphIndex,
    query_vector: Sequence[float],
    *,
    top_k: int,
    filters: RetrievalFilters | None = None,
) -> list[ParagraphSearchResult]:
    """Run cosine search and expose hits through the shared result contract."""

    hits = paragraph_index.search(
        query_vector,
        limit=top_k,
        query_filter=build_qdrant_filter(filters),
    )
    return semantic_hits_to_results(hits)


def build_qdrant_filter(filters: RetrievalFilters | None) -> Filter | None:
    """Translate shared metadata constraints to native Qdrant conditions."""

    if filters is None or not filters.is_active:
        return None
    conditions: list[FieldCondition] = []
    if filters.court is not None:
        conditions.append(
            FieldCondition(
                key=COURT_FILTER_FIELD,
                match=MatchValue(value=filters.court),
            )
        )
    if filters.year is not None:
        conditions.append(
            FieldCondition(
                key=JUDGMENT_YEAR_FILTER_FIELD,
                match=MatchValue(value=filters.year),
            )
        )
    if filters.case_number is not None:
        conditions.append(
            FieldCondition(
                key=CASE_NUMBER_FILTER_FIELD,
                match=MatchValue(value=filters.case_number),
            )
        )
    return Filter(must=conditions)


class DenseParagraphRetriever:
    """Adapt an embedding provider and Qdrant index to textual paragraph search."""

    def __init__(
        self,
        paragraph_index: QdrantParagraphIndex,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self.paragraph_index = paragraph_index
        self.embedding_provider = embedding_provider

    def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: RetrievalFilters | None = None,
    ) -> list[ParagraphSearchResult]:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        return search_dense(
            self.paragraph_index,
            self.embedding_provider.embed_query(query),
            top_k=top_k,
            filters=filters,
        )
