#!/usr/bin/env python3
"""Run a semantic paragraph query against the Qdrant corpus index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from legal_rag.config import get_settings  # noqa: E402
from legal_rag.embeddings import SentenceTransformerEmbeddingProvider  # noqa: E402
from legal_rag.retrieval import (  # noqa: E402
    RetrievalFilters,
    build_retrieval_filters,
    search_dense,
)
from legal_rag.vector import QdrantParagraphIndex  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print the legal paragraphs most similar to a textual query.",
    )
    parser.add_argument(
        "query",
        nargs="+",
        help="Natural-language search query",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of matches to print (default: %(default)s)",
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
    parser.add_argument("--court", default=None, help="Exact court metadata filter")
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Exact judgment calendar year filter",
    )
    parser.add_argument(
        "--case-number",
        default=None,
        help="Exact canonical case-number filter",
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


def search(
    query: str,
    *,
    top_k: int,
    collection: str | None,
    model: str | None,
    filters: RetrievalFilters | None = None,
) -> int:
    if top_k <= 0:
        raise ValueError("top_k must be positive")

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
    paragraph_index.validate_collection(provider.dimension)
    results = search_dense(
        paragraph_index,
        provider.embed_query(query),
        top_k=top_k,
        filters=filters,
    )

    if filters is not None:
        print(
            f"Filters: court={filters.court or '-'} year={filters.year or '-'} "
            f"case_number={filters.case_number or '-'}"
        )

    if not results:
        print("No matching paragraphs found.")
        return 0

    for result in results:
        judgment_date = result.judgment_date or "unknown date"
        case_number = result.case_number or "unknown case number"
        court = result.court or "Unknown court"
        page = result.page_number if result.page_number is not None else "unknown"
        print(f"{result.rank}. score={result.score:.4f}")
        print(
            f"   {result.title or 'Untitled case'} | {case_number} | "
            f"{court} | {judgment_date}"
        )
        print(
            f"   paragraph={result.paragraph_number} page={page} "
            f"uid={result.paragraph_uid}"
        )
        print(f"   {result.text}")
        if result.rank != len(results):
            print()
    return len(results)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    query = " ".join(args.query).strip()
    try:
        filters = build_retrieval_filters(
            court=args.court,
            year=args.year,
            case_number=args.case_number,
        )
    except (TypeError, ValueError) as error:
        raise SystemExit(f"Invalid metadata filters: {error}") from error
    search(
        query,
        top_k=args.top_k,
        collection=args.collection,
        model=args.model,
        filters=filters,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
