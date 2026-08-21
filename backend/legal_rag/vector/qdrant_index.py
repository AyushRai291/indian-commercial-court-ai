"""Small Qdrant adapter for paragraph vectors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Filter, PointStruct, VectorParams


@dataclass(frozen=True, slots=True)
class ParagraphVectorRecord:
    """A paragraph vector ready to be written to Qdrant."""

    point_id: int | str
    vector: Sequence[float]
    payload: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class SemanticSearchResult:
    """Storage-neutral representation of one Qdrant search hit."""

    point_id: int | str
    score: float
    payload: dict[str, Any]


class QdrantParagraphIndex:
    """Manage one cosine-similarity Qdrant collection."""

    def __init__(
        self,
        *,
        url: str,
        collection_name: str,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        if not collection_name.strip():
            raise ValueError("collection_name must not be empty")
        self.collection_name = collection_name
        self.client = (
            client if client is not None else QdrantClient(url=url, api_key=api_key)
        )

    def ensure_collection(self, vector_size: int, *, recreate: bool = False) -> None:
        """Create the collection or validate its dense cosine configuration.

        ``recreate`` is intentionally explicit because it deletes every existing
        point before rebuilding the PostgreSQL-backed paragraph index.
        """

        if vector_size <= 0:
            raise ValueError("vector_size must be positive")

        if recreate and self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )
            return

        self._validate_existing_collection(vector_size)

    def validate_collection(self, vector_size: int) -> None:
        """Require an existing compatible collection without mutating Qdrant."""

        if vector_size <= 0:
            raise ValueError("vector_size must be positive")
        if not self.client.collection_exists(self.collection_name):
            raise ValueError(
                f"Qdrant collection {self.collection_name!r} does not exist; "
                "run scripts/index_vectors.py first"
            )
        self._validate_existing_collection(vector_size)

    def _validate_existing_collection(self, vector_size: int) -> None:
        collection = self.client.get_collection(self.collection_name)
        vectors_config = collection.config.params.vectors
        existing_size = getattr(vectors_config, "size", None)
        if existing_size is None:
            raise ValueError(
                f"Qdrant collection {self.collection_name!r} does not use the "
                "required unnamed dense-vector configuration"
            )
        if int(existing_size) != vector_size:
            raise ValueError(
                f"Qdrant collection {self.collection_name!r} uses vectors of "
                f"size {existing_size}, but the embedding model produces {vector_size}"
            )
        existing_distance = getattr(vectors_config, "distance", None)
        if existing_distance is not None and existing_distance != Distance.COSINE:
            raise ValueError(
                f"Qdrant collection {self.collection_name!r} uses "
                f"{existing_distance} distance, but paragraph search requires cosine"
            )

    def upsert(self, records: Sequence[ParagraphVectorRecord]) -> int:
        """Idempotently insert or replace paragraph points by stable point ID."""

        if not records:
            return 0

        points = [
            PointStruct(
                id=record.point_id,
                vector=[float(value) for value in record.vector],
                payload=dict(record.payload),
            )
            for record in records
        ]
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
        return len(points)

    def list_point_ids(self, *, batch_size: int = 256) -> list[int | str]:
        """Return all point IDs using Qdrant's paginated scroll API."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        point_ids: list[int | str] = []
        offset: int | str | None = None
        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            point_ids.extend(point.id for point in points)
            if next_offset is None:
                return point_ids
            offset = next_offset

    def search(
        self,
        query_vector: Sequence[float],
        *,
        limit: int,
        query_filter: Filter | None = None,
    ) -> list[SemanticSearchResult]:
        """Return the closest paragraphs in descending similarity order."""

        if len(query_vector) == 0:
            raise ValueError("query_vector must not be empty")
        if limit <= 0:
            raise ValueError("limit must be positive")

        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=[float(value) for value in query_vector],
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
            hits = response.points
        else:  # pragma: no cover - compatibility with older/fake clients
            hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=[float(value) for value in query_vector],
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )

        return [
            SemanticSearchResult(
                point_id=hit.id,
                score=float(hit.score),
                payload=dict(hit.payload or {}),
            )
            for hit in hits
        ]
