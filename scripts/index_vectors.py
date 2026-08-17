#!/usr/bin/env python3
"""Embed PostgreSQL paragraphs and upsert them into Qdrant."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import select


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from legal_rag.config import get_settings  # noqa: E402
from legal_rag.database import get_engine, get_session_factory  # noqa: E402
from legal_rag.embeddings import SentenceTransformerEmbeddingProvider  # noqa: E402
from legal_rag.models import Case, Paragraph  # noqa: E402
from legal_rag.vector import ParagraphVectorRecord, QdrantParagraphIndex  # noqa: E402


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Embed stored legal paragraphs and index them in Qdrant.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(getattr(settings, "embedding_batch_size", 64)),
        help="Paragraphs embedded and upserted per batch (default: %(default)s)",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Qdrant collection (defaults to QDRANT_COLLECTION)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Sentence Transformers model (defaults to EMBEDDING_MODEL)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and rebuild the collection, removing stale paragraph points",
    )
    return parser.parse_args()


def _configured_dimension(settings: Any) -> int | None:
    value = getattr(settings, "embedding_dimension", None)
    return int(value) if value is not None else None


def _qdrant_api_key(settings: Any) -> str | None:
    value = getattr(settings, "qdrant_api_key", None)
    if value is None:
        return None
    if hasattr(value, "get_secret_value"):
        return value.get_secret_value()
    return str(value)


def paragraph_payload(paragraph: Paragraph, case: Case) -> dict[str, Any]:
    """Build the canonical payload stored for every paragraph point."""

    judgment_date = case.judgment_date
    return {
        "case_id": case.id,
        "title": case.title,
        "court": case.court,
        "year": judgment_date.year if judgment_date is not None else None,
        "paragraph_number": paragraph.paragraph_number,
        "text": paragraph.text,
    }


def index_vectors(
    *,
    batch_size: int,
    collection: str | None,
    model: str | None,
    recreate: bool = False,
) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    settings = get_settings()
    provider = SentenceTransformerEmbeddingProvider(
        model or settings.embedding_model,
        expected_dimension=_configured_dimension(settings),
    )
    paragraph_index = QdrantParagraphIndex(
        url=str(settings.qdrant_url),
        api_key=_qdrant_api_key(settings),
        collection_name=collection or settings.qdrant_collection,
    )
    paragraph_index.ensure_collection(provider.dimension, recreate=recreate)

    engine = get_engine(settings.database_url)
    session_factory = get_session_factory(engine)
    indexed = 0
    last_paragraph_id = 0

    while True:
        # Keep database read transactions short: model inference and Qdrant
        # network writes can be much slower than fetching one PostgreSQL batch.
        with session_factory() as session:
            statement = (
                select(Paragraph, Case)
                .join(Case, Paragraph.case_id == Case.id)
                .where(Paragraph.id > last_paragraph_id)
                .order_by(Paragraph.id)
                .limit(batch_size)
            )
            rows = session.execute(statement).all()
        if not rows:
            break

        texts = [paragraph.text for paragraph, _case in rows]
        embeddings = provider.embed_documents(texts, batch_size=batch_size)
        records = [
            ParagraphVectorRecord(
                # PostgreSQL paragraph IDs are stable, globally unique, and
                # therefore make reruns overwrite rather than duplicate points.
                point_id=paragraph.id,
                vector=embedding,
                payload=paragraph_payload(paragraph, case),
            )
            for (paragraph, case), embedding in zip(rows, embeddings, strict=True)
        ]
        indexed += paragraph_index.upsert(records)
        last_paragraph_id = rows[-1][0].id
        print(f"Indexed {indexed} paragraphs", flush=True)

    return indexed


def main() -> int:
    args = parse_args()
    indexed = index_vectors(
        batch_size=args.batch_size,
        collection=args.collection,
        model=args.model,
        recreate=args.recreate,
    )
    print(f"Vector indexing complete: {indexed} paragraphs indexed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
